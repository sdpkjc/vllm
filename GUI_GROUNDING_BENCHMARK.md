# GUI Grounding Benchmark

Benchmark for evaluating VL model GUI grounding capability using the OSWorld-G dataset.

## Quick Start

```bash
# 1. Start vLLM server
vllm serve Qwen/Qwen3-VL-8B-Instruct --trust-remote-code

# 2. Run benchmark (auto-downloads dataset from HuggingFace)
python benchmarks/benchmark_gui_grounding.py \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --num-samples 100 \
    --output-file results.json
```

Or use CLI:
```bash
vllm bench gui-grounding \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --num-samples 100
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--base-url` | vLLM server URL | `http://localhost:8000` |
| `--model` | Model name | Required |
| `--num-samples` | Number of samples to evaluate | All |
| `--output-file` | Path to save results JSON | None |
| `--dataset-path` | Local dataset JSON path | None (download from HF) |
| `--image-dir` | Local image directory | None |
| `--max-concurrency` | Max concurrent requests | 8 |
| `--seed` | Random seed | 0 |

## Using Local Dataset

```bash
python benchmarks/benchmark_gui_grounding.py \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --dataset-path /path/to/OSWorld-G_refined.json \
    --image-dir /path/to/images \
    --num-samples 100
```

## Output Example

The following is a baseline result from `Qwen/Qwen3-VL-8B-Instruct` on the full OSWorld-G dataset (510 samples). The expected baseline accuracy is approximately **57-58%**. If your result deviates significantly, check your setup.

```
============== GUI Grounding Benchmark Result ==============
Model:                                   Qwen/Qwen3-VL-8B-Instruct
Total samples:                           510                 
Correct predictions:                     290                 
Accuracy:                                0.5686              
Failed parses:                           3                   
API errors:                              0                   
Mean latency (s):                        2.14                

------------------ Breakdown by Box Type -------------------
  bbox                 Total:   470  Correct:   258  Accuracy: 0.5489
  polygon              Total:    40  Correct:    32  Accuracy: 0.8000

------------------ Breakdown by GUI Type -------------------
  Accordion/Collapsible Panel Total:     3  Correct:     2  Accuracy: 0.6667
  Banner/Notification  Total:    15  Correct:    14  Accuracy: 0.9333
  Button               Total:   188  Correct:   119  Accuracy: 0.6330
  Checkbox             Total:    26  Correct:    21  Accuracy: 0.8077
  Color Picker         Total:     6  Correct:     4  Accuracy: 0.6667
  Date Picker          Total:     3  Correct:     3  Accuracy: 1.0000
  Dialog/Modal         Total:    31  Correct:    24  Accuracy: 0.7742
  Divider              Total:     4  Correct:     0  Accuracy: 0.0000
  Drawer               Total:     1  Correct:     0  Accuracy: 0.0000
  Dropdown Menu        Total:    70  Correct:    47  Accuracy: 0.6714
  Grid                 Total:     4  Correct:     1  Accuracy: 0.2500
  Icon                 Total:   230  Correct:   110  Accuracy: 0.4783
  Image                Total:    12  Correct:     6  Accuracy: 0.5000
  Input Box            Total:     1  Correct:     1  Accuracy: 1.0000
  Label                Total:   240  Correct:   171  Accuracy: 0.7125
  List                 Total:    18  Correct:    11  Accuracy: 0.6111
  Menu Bar             Total:    21  Correct:    17  Accuracy: 0.8095
  Menu bar             Total:     1  Correct:     1  Accuracy: 1.0000
  Pagination Control   Total:     6  Correct:     2  Accuracy: 0.3333
  Panel/Container      Total:     3  Correct:     1  Accuracy: 0.3333
  Radio Button         Total:     3  Correct:     2  Accuracy: 0.6667
  Search Bar           Total:     2  Correct:     1  Accuracy: 0.5000
  Sidebar              Total:    31  Correct:    14  Accuracy: 0.4516
  Slider               Total:    21  Correct:     3  Accuracy: 0.1429
  Stepper              Total:    15  Correct:     7  Accuracy: 0.4667
  Tab                  Total:    11  Correct:     4  Accuracy: 0.3636
  Table                Total:    28  Correct:    14  Accuracy: 0.5000
  Text Field/Input Box Total:    23  Correct:    14  Accuracy: 0.6087
  Toggle/Switch        Total:     4  Correct:     3  Accuracy: 0.7500
  Toolbar              Total:    56  Correct:    23  Accuracy: 0.4107
  Tree View            Total:     3  Correct:     2  Accuracy: 0.6667
  Window Border        Total:     1  Correct:     0  Accuracy: 0.0000
  Window Hierarchy     Total:     1  Correct:     1  Accuracy: 1.0000

============================================================
```

## File Structure

```
benchmarks/benchmark_gui_grounding.py           # Standalone script
vllm/benchmarks/gui_grounding.py                # Core implementation
vllm/entrypoints/cli/benchmark/gui_grounding.py # CLI subcommand
```
