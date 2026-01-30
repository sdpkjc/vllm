# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GUI Grounding Benchmark for vLLM.

This module implements a benchmark for evaluating GUI grounding capabilities
using the OSWorld-G dataset. It tests a model's ability to predict click
coordinates given a screenshot and an instruction.

Usage:
    # Auto-download from HuggingFace (recommended):
    vllm bench gui-grounding \\
        --base-url http://localhost:8000 \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --num-samples 100 \\
        --output-file results.json

    # Use local dataset:
    vllm bench gui-grounding \\
        --base-url http://localhost:8000 \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --dataset-path /path/to/OSWorld-G_refined.json \\
        --image-dir /path/to/images \\
        --num-samples 100 \\
        --output-file results.json
"""

import argparse
import asyncio
import base64
import io
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
HUGGINGFACE_DATASET_NAME = "MMInstruction/OSWorld-G"


@dataclass
class OSWorldGSample:
    """A single sample from the OSWorld-G dataset."""

    id: str
    image_path: str
    image_size: tuple[int, int]
    instruction: str
    box_type: str  # "bbox", "polygon", or "refusal"
    box_coordinates: list[float]
    gui_types: list[str]
    # For HuggingFace dataset: store PIL image directly
    pil_image: Any = None  # PIL.Image.Image


@dataclass
class GroundingResult:
    """Result of a single grounding prediction."""

    sample_id: str
    instruction: str
    image_path: str
    box_type: str
    gui_types: list[str]
    ground_truth: list[float]
    predicted_coords: tuple[float, float] | None
    is_correct: bool
    raw_response: str
    latency: float
    error: str = ""


@dataclass
class GroundingMetrics:
    """Aggregated metrics for the grounding benchmark."""

    total_samples: int
    correct: int
    accuracy: float
    failed_parses: int
    api_errors: int
    breakdown_by_gui_type: dict[str, dict[str, Any]]
    breakdown_by_box_type: dict[str, dict[str, Any]]
    mean_latency: float
    results: list[GroundingResult] = field(default_factory=list)


class OSWorldGDataset:
    """Loader for the OSWorld-G dataset.

    Supports loading from:
    1. HuggingFace Hub (default, auto-downloads)
    2. Local JSON file with image directory
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        image_dir: str | None = None,
        random_seed: int = 0,
    ):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.random_seed = random_seed
        self.data: list[OSWorldGSample] = []
        self._use_huggingface = False

    def load_data(self) -> None:
        """Load data from local JSON file or HuggingFace."""
        if self.dataset_path and self.image_dir:
            self._load_from_local()
        else:
            self._load_from_huggingface()

    def _load_from_local(self) -> None:
        """Load data from a local JSON file."""
        with open(self.dataset_path) as f:
            raw_data = json.load(f)

        for item in raw_data:
            sample = OSWorldGSample(
                id=item["id"],
                image_path=item["image_path"],
                image_size=tuple(item["image_size"]),
                instruction=item["instruction"],
                box_type=item["box_type"],
                box_coordinates=item["box_coordinates"],
                gui_types=item.get("GUI_types", []),
            )
            # Verify image exists
            image_full_path = self.image_dir / sample.image_path
            if image_full_path.exists():
                self.data.append(sample)
            else:
                print(f"Warning: Image not found: {image_full_path}")

        print(f"Loaded {len(self.data)} samples from {self.dataset_path}")

    def _load_from_huggingface(self) -> None:
        """Load data from HuggingFace Hub."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for HuggingFace loading. "
                "Install it with: pip install datasets"
            )

        print(f"Downloading OSWorld-G dataset from HuggingFace ({HUGGINGFACE_DATASET_NAME})...")
        hf_dataset = load_dataset(HUGGINGFACE_DATASET_NAME, split="test")

        for item in hf_dataset:
            sample = OSWorldGSample(
                id=item["id"],
                image_path=item["image_path"],
                image_size=tuple(item["image_size"]),
                instruction=item["instruction"],
                box_type=item["box_type"],
                box_coordinates=list(item["box_coordinates"]),
                gui_types=list(item["GUI_types"]),
                pil_image=item["image"],
            )
            self.data.append(sample)

        self._use_huggingface = True
        print(f"Loaded {len(self.data)} samples from HuggingFace")

    def sample(self, num_samples: int | None = None) -> list[OSWorldGSample]:
        """Sample a subset of the dataset."""
        if num_samples is None or num_samples >= len(self.data):
            return self.data.copy()

        random.seed(self.random_seed)
        return random.sample(self.data, num_samples)

    def get_image_path(self, sample: OSWorldGSample) -> Path | None:
        """Get the full path to the image file (for local datasets only)."""
        if self._use_huggingface:
            return None
        return self.image_dir / sample.image_path


class GroundingEvaluator:
    """Evaluator for grounding predictions."""

    @staticmethod
    def point_in_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
        """Check if a point is inside a bounding box.

        Args:
            point: (x, y) coordinates of the predicted point
            bbox: [x_center, y_center, width, height] in absolute coordinates

        Returns:
            True if the point is inside the bbox
        """
        x, y = point
        cx, cy, w, h = bbox
        x_min = cx - w / 2
        x_max = cx + w / 2
        y_min = cy - h / 2
        y_max = cy + h / 2
        return x_min <= x <= x_max and y_min <= y <= y_max

    @staticmethod
    def point_in_polygon(point: tuple[float, float], polygon: list[float]) -> bool:
        """Check if a point is inside a polygon using the ray casting algorithm.

        Args:
            point: (x, y) coordinates of the predicted point
            polygon: [x1, y1, x2, y2, ..., xn, yn] vertices of the polygon

        Returns:
            True if the point is inside the polygon
        """
        x, y = point
        n = len(polygon) // 2
        inside = False

        # Extract x and y coordinates
        px = [polygon[i * 2] for i in range(n)]
        py = [polygon[i * 2 + 1] for i in range(n)]

        j = n - 1
        for i in range(n):
            if ((py[i] > y) != (py[j] > y)) and (
                x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]
            ):
                inside = not inside
            j = i

        return inside

    def evaluate(
        self, prediction: tuple[float, float] | None, sample: OSWorldGSample
    ) -> bool:
        """Evaluate a prediction against ground truth.

        Args:
            prediction: Predicted (x, y) coordinates, or None if parsing failed
            sample: The ground truth sample

        Returns:
            True if the prediction is correct
        """
        # Handle refusal samples: model should NOT return coordinates
        if sample.box_type == "refusal":
            # Correct if model refuses to give coordinates (returns None)
            return prediction is None

        if prediction is None:
            return False

        # Normalize coordinates if needed
        x, y = prediction
        img_w, img_h = sample.image_size

        # Detect and convert normalized coordinates
        # Case 1: Normalized (0-1 range)
        if x <= 1.0 and y <= 1.0:
            x = x * img_w
            y = y * img_h
        # Case 2: Per-mille (0-1000 range) - common in some models
        elif x <= 1000 and y <= 1000 and max(x, y) > 1.0:
            # Only convert if values suggest per-mille format
            # (both coordinates are small relative to typical screen sizes)
            if img_w > 1000 or img_h > 1000:
                x = x * img_w / 1000
                y = y * img_h / 1000

        prediction = (x, y)

        if sample.box_type == "bbox":
            return self.point_in_bbox(prediction, sample.box_coordinates)
        elif sample.box_type == "polygon":
            coords = sample.box_coordinates
            # Special case: 4 values means rectangle [x_min, y_min, x_max, y_max]
            if len(coords) == 4:
                x_min, y_min, x_max, y_max = coords
                return x_min <= x <= x_max and y_min <= y <= y_max
            else:
                return self.point_in_polygon(prediction, coords)
        else:
            raise ValueError(f"Unknown box type: {sample.box_type}")


def parse_coordinates(response: str) -> tuple[float, float] | None:
    """Parse coordinates from model output.

    Supports multiple formats:
    1. tool_call format: <tool_call>{"name": "computer_use", "arguments": {"coordinate": [x, y]}}</tool_call>
    2. Simple coordinate format: [x, y] or (x, y)
    3. JSON format with coordinate/coordinates key

    Args:
        response: The raw model response

    Returns:
        (x, y) tuple if parsing succeeds, None otherwise
    """
    # Try tool_call format first
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    match = re.search(tool_call_pattern, response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "arguments" in data and "coordinate" in data["arguments"]:
                coord = data["arguments"]["coordinate"]
                return (float(coord[0]), float(coord[1]))
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            pass

    # Try JSON block format
    json_block_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_block_pattern, response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "coordinate" in data:
                coord = data["coordinate"]
                return (float(coord[0]), float(coord[1]))
            if "coordinates" in data:
                coord = data["coordinates"]
                return (float(coord[0]), float(coord[1]))
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            pass

    # Try simple bracket format: [x, y] or (x, y)
    bracket_pattern = r"[\[\(]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)]"
    matches = re.findall(bracket_pattern, response)
    if matches:
        # Take the last match (usually the final answer)
        x, y = matches[-1]
        return (float(x), float(y))

    return None


def build_grounding_prompt(instruction: str) -> str:
    """Build the prompt for GUI grounding task.

    Args:
        instruction: The task instruction from the dataset

    Returns:
        The formatted prompt string
    """
    prompt = f"""You are a GUI agent. Given a screenshot and an instruction, predict the click coordinate to complete the task.

Instruction: {instruction}

Output your prediction in the following format:
<tool_call>{{"name": "computer_use", "arguments": {{"coordinate": [x, y]}}}}</tool_call>

Where x and y are the pixel coordinates of where to click on the screenshot."""

    return prompt


def encode_image_base64(image_path: Path | None = None, pil_image: Any = None) -> str:
    """Encode an image to base64 string.

    Args:
        image_path: Path to image file (for local datasets)
        pil_image: PIL Image object (for HuggingFace datasets)

    Returns:
        Base64 encoded string
    """
    if pil_image is not None:
        # Convert PIL image to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    elif image_path is not None:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    else:
        raise ValueError("Either image_path or pil_image must be provided")


def get_image_media_type(image_path: Path | None = None, pil_image: Any = None) -> str:
    """Get the media type for an image.

    Args:
        image_path: Path to image file (for local datasets)
        pil_image: PIL Image object (for HuggingFace datasets)

    Returns:
        Media type string
    """
    if pil_image is not None:
        # PIL images are converted to PNG
        return "image/png"
    elif image_path is not None:
        suffix = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_types.get(suffix, "image/png")
    else:
        return "image/png"


async def send_grounding_request(
    sample: OSWorldGSample,
    image_path: Path | None,
    api_url: str,
    model: str,
    session: aiohttp.ClientSession,
    max_tokens: int = 512,
) -> tuple[str, float, str]:
    """Send a grounding request to the vLLM server.

    Args:
        sample: The dataset sample (may contain pil_image for HuggingFace data)
        image_path: Full path to the image file (None for HuggingFace data)
        api_url: The API endpoint URL
        model: The model name
        session: aiohttp session
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple of (generated_text, latency, error_message)
    """
    prompt = build_grounding_prompt(sample.instruction)

    # Encode image (from file path or PIL image)
    image_base64 = encode_image_base64(image_path, sample.pil_image)
    media_type = get_image_media_type(image_path, sample.pil_image)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # Deterministic output
    }

    headers = {"Content-Type": "application/json"}

    start_time = time.perf_counter()
    try:
        async with session.post(
            api_url, json=payload, headers=headers, timeout=AIOHTTP_TIMEOUT
        ) as response:
            latency = time.perf_counter() - start_time
            if response.status == 200:
                data = await response.json()
                generated_text = data["choices"][0]["message"]["content"]
                return generated_text, latency, ""
            else:
                error_text = await response.text()
                return "", latency, f"HTTP {response.status}: {error_text}"
    except Exception as e:
        latency = time.perf_counter() - start_time
        return "", latency, str(e)


async def run_benchmark(
    dataset: OSWorldGDataset,
    samples: list[OSWorldGSample],
    api_url: str,
    model: str,
    max_concurrency: int = 8,
) -> list[GroundingResult]:
    """Run the grounding benchmark on all samples.

    Args:
        dataset: The loaded dataset
        samples: List of samples to evaluate
        api_url: The API endpoint URL
        model: The model name
        max_concurrency: Maximum concurrent requests

    Returns:
        List of grounding results
    """
    evaluator = GroundingEvaluator()
    results: list[GroundingResult] = []
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_sample(
        sample: OSWorldGSample, session: aiohttp.ClientSession, pbar: tqdm
    ) -> GroundingResult:
        async with semaphore:
            image_path = dataset.get_image_path(sample)
            response_text, latency, error = await send_grounding_request(
                sample, image_path, api_url, model, session
            )

            if error:
                result = GroundingResult(
                    sample_id=sample.id,
                    instruction=sample.instruction,
                    image_path=sample.image_path,
                    box_type=sample.box_type,
                    gui_types=sample.gui_types,
                    ground_truth=sample.box_coordinates,
                    predicted_coords=None,
                    is_correct=False,
                    raw_response=response_text,
                    latency=latency,
                    error=error,
                )
            else:
                predicted_coords = parse_coordinates(response_text)
                is_correct = evaluator.evaluate(predicted_coords, sample)
                result = GroundingResult(
                    sample_id=sample.id,
                    instruction=sample.instruction,
                    image_path=sample.image_path,
                    box_type=sample.box_type,
                    gui_types=sample.gui_types,
                    ground_truth=sample.box_coordinates,
                    predicted_coords=predicted_coords,
                    is_correct=is_correct,
                    raw_response=response_text,
                    latency=latency,
                )

            pbar.update(1)
            return result

    async with aiohttp.ClientSession() as session:
        with tqdm(total=len(samples), desc="Evaluating") as pbar:
            tasks = [process_sample(sample, session, pbar) for sample in samples]
            results = await asyncio.gather(*tasks)

    return list(results)


def calculate_metrics(results: list[GroundingResult]) -> GroundingMetrics:
    """Calculate aggregate metrics from results.

    Args:
        results: List of grounding results

    Returns:
        Aggregated metrics
    """
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    # Failed parses: no coordinates returned, no API error, and NOT a refusal sample
    failed_parses = sum(
        1 for r in results
        if r.predicted_coords is None and not r.error and r.box_type != "refusal"
    )
    api_errors = sum(1 for r in results if r.error)

    # Breakdown by GUI type
    gui_type_stats: dict[str, dict[str, int]] = {}
    for result in results:
        for gui_type in result.gui_types:
            if gui_type not in gui_type_stats:
                gui_type_stats[gui_type] = {"total": 0, "correct": 0}
            gui_type_stats[gui_type]["total"] += 1
            if result.is_correct:
                gui_type_stats[gui_type]["correct"] += 1

    breakdown_by_gui_type = {}
    for gui_type, stats in gui_type_stats.items():
        breakdown_by_gui_type[gui_type] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
        }

    # Breakdown by box type
    box_type_stats: dict[str, dict[str, int]] = {}
    for result in results:
        box_type = result.box_type
        if box_type not in box_type_stats:
            box_type_stats[box_type] = {"total": 0, "correct": 0}
        box_type_stats[box_type]["total"] += 1
        if result.is_correct:
            box_type_stats[box_type]["correct"] += 1

    breakdown_by_box_type = {}
    for box_type, stats in box_type_stats.items():
        breakdown_by_box_type[box_type] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
        }

    # Calculate mean latency
    latencies = [r.latency for r in results if r.latency > 0]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0

    return GroundingMetrics(
        total_samples=total,
        correct=correct,
        accuracy=correct / total if total > 0 else 0,
        failed_parses=failed_parses,
        api_errors=api_errors,
        breakdown_by_gui_type=breakdown_by_gui_type,
        breakdown_by_box_type=breakdown_by_box_type,
        mean_latency=mean_latency,
        results=results,
    )


def print_results(metrics: GroundingMetrics, model: str) -> None:
    """Print benchmark results to console."""
    print()
    print("{s:{c}^{n}}".format(s=" GUI Grounding Benchmark Result ", n=60, c="="))
    print("{:<40} {:<20}".format("Model:", model))
    print("{:<40} {:<20}".format("Total samples:", metrics.total_samples))
    print("{:<40} {:<20}".format("Correct predictions:", metrics.correct))
    print("{:<40} {:<20.4f}".format("Accuracy:", metrics.accuracy))
    print("{:<40} {:<20}".format("Failed parses:", metrics.failed_parses))
    print("{:<40} {:<20}".format("API errors:", metrics.api_errors))
    print("{:<40} {:<20.2f}".format("Mean latency (s):", metrics.mean_latency))
    print()

    if metrics.breakdown_by_box_type:
        print("{s:{c}^{n}}".format(s=" Breakdown by Box Type ", n=60, c="-"))
        for box_type, stats in sorted(metrics.breakdown_by_box_type.items()):
            print(
                "  {:<20} Total: {:>5}  Correct: {:>5}  Accuracy: {:.4f}".format(
                    box_type, stats["total"], stats["correct"], stats["accuracy"]
                )
            )
        print()

    if metrics.breakdown_by_gui_type:
        print("{s:{c}^{n}}".format(s=" Breakdown by GUI Type ", n=60, c="-"))
        for gui_type, stats in sorted(metrics.breakdown_by_gui_type.items()):
            print(
                "  {:<20} Total: {:>5}  Correct: {:>5}  Accuracy: {:.4f}".format(
                    gui_type, stats["total"], stats["correct"], stats["accuracy"]
                )
            )
        print()

    print("=" * 60)


def save_results(metrics: GroundingMetrics, model: str, output_file: str) -> None:
    """Save benchmark results to a JSON file."""
    output = {
        "model": model,
        "total_samples": metrics.total_samples,
        "correct": metrics.correct,
        "accuracy": metrics.accuracy,
        "failed_parses": metrics.failed_parses,
        "api_errors": metrics.api_errors,
        "mean_latency": metrics.mean_latency,
        "breakdown_by_gui_type": metrics.breakdown_by_gui_type,
        "breakdown_by_box_type": metrics.breakdown_by_box_type,
        "detailed_results": [
            {
                "sample_id": r.sample_id,
                "instruction": r.instruction,
                "image_path": r.image_path,
                "box_type": r.box_type,
                "gui_types": r.gui_types,
                "ground_truth": r.ground_truth,
                "predicted_coords": list(r.predicted_coords)
                if r.predicted_coords
                else None,
                "is_correct": r.is_correct,
                "raw_response": r.raw_response,
                "latency": r.latency,
                "error": r.error,
            }
            for r in metrics.results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_file}")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the GUI grounding benchmark."""
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the vLLM server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for inference",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local OSWorld-G JSON dataset file. "
        "If not provided, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing the dataset images. "
        "Required if --dataset-path is provided.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save detailed results as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling (default: 0)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent API requests (default: 8)",
    )


def main(args: argparse.Namespace) -> None:
    """Main entry point for the GUI grounding benchmark."""
    # Validate arguments
    if args.dataset_path and not args.image_dir:
        print("Error: --image-dir is required when using --dataset-path")
        return
    if args.image_dir and not args.dataset_path:
        print("Error: --dataset-path is required when using --image-dir")
        return

    # Construct API URL
    api_url = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    if args.dataset_path:
        print(f"Loading dataset from {args.dataset_path}")
        print(f"Image directory: {args.image_dir}")
    else:
        print("Loading dataset from HuggingFace (MMInstruction/OSWorld-G)...")

    # Load dataset
    dataset = OSWorldGDataset(
        dataset_path=args.dataset_path,
        image_dir=args.image_dir,
        random_seed=args.seed,
    )
    dataset.load_data()

    if len(dataset.data) == 0:
        print("Error: No valid samples found in dataset")
        return

    # Sample if needed
    samples = dataset.sample(args.num_samples)
    print(f"Evaluating {len(samples)} samples")

    # Run benchmark
    results = asyncio.run(
        run_benchmark(
            dataset=dataset,
            samples=samples,
            api_url=api_url,
            model=args.model,
            max_concurrency=args.max_concurrency,
        )
    )

    # Calculate and display metrics
    metrics = calculate_metrics(results)
    print_results(metrics, args.model)

    # Save results if output file specified
    if args.output_file:
        save_results(metrics, args.model, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GUI Grounding Benchmark using OSWorld-G dataset"
    )
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)
