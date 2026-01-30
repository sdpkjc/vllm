

vllm serve  --trust-remote-code


vllm serve /workspace/osagent_cloud/ckpt/qwen3-vl-8b-instruct \
  --served-model-name "qwen3-vl-8b-instruct" \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 30000 \
  --max-model-len "10000" \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --gpu-memory-utilization 0.8 \
  --mm-processor-cache-gb 64




python benchmarks/benchmark_gui_grounding.py \
    --base-url http://localhost:30000 \
    --model qwen3-vl-8b-instruct \
    --num-samples 100 \
    --output-file results.json