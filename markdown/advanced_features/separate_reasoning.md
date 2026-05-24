# Reasoning Parser

SGLang supports parsing reasoning content out from "normal" content for reasoning models such as [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1).

## Supported Models & Parsers

| Model  |  Reasoning tags      | Parser | Notes |
|---------|-----------------------------|------------------|-------|
| [DeepSeek‑R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) | `<think>` … `</think>` | `deepseek-r1` | Supports all variants (R1, R1-0528, R1-Distill) |
| [DeepSeek‑V3 series](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) | `<think>` … `</think>` | `deepseek-v3` | Including [DeepSeek‑V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp). Supports `thinking` parameter |
| [Standard Qwen3 models](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | `<think>` … `</think>` | `qwen3` | Supports `enable_thinking` parameter |
| [Qwen3-Thinking models](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) | `<think>` … `</think>` | `qwen3` or `qwen3-thinking` | Always generates thinking content |
| [Kimi K2 Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) | `◁think▷` … `◁/think▷` | `kimi_k2` | Uses special thinking delimiters. Also requires `--tool-call-parser kimi_k2` for tool use. |
| [GPT OSS](https://huggingface.co/openai/gpt-oss-120b) | `<\|channel\|>analysis<\|message\|>` … `<\|end\|>` | `gpt-oss` | N/A |
### Model-Specific Behaviors

**DeepSeek-R1 Family:**
- DeepSeek-R1: No `<think>` start tag, jumps directly to thinking content
- DeepSeek-R1-0528: Generates both `<think>` start and `</think>` end tags
- Both are handled by the same `deepseek-r1` parser

**DeepSeek-V3 Family:**
- DeepSeek-V3.1/V3.2: Hybrid model supporting both thinking and non-thinking modes, use the `deepseek-v3` parser and `thinking` parameter (NOTE: not `enable_thinking`)

**Qwen3 Family:**
- Standard Qwen3 (e.g., Qwen3-2507): Use `qwen3` parser, supports `enable_thinking` in chat templates
- Qwen3-Thinking (e.g., Qwen3-235B-A22B-Thinking-2507): Use `qwen3` or `qwen3-thinking` parser, always thinks

**Kimi K2:**
- Kimi K2 Thinking: Uses special `◁think▷` and `◁/think▷` tags. For agentic tool use, also specify `--tool-call-parser kimi_k2`.

**GPT OSS:**
- GPT OSS: Uses special `<|channel|>analysis<|message|>` and `<|end|>` tags

## Usage

### Launching the Server

Specify the `--reasoning-parser` option.


```python
import requests
from openai import OpenAI
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --host 0.0.0.0 --reasoning-parser deepseek-r1 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:02<00:02,  2.31s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:04<00:00,  2.08s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:04<00:00,  2.11s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:28,  4.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:28,  4.72s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:55,  2.07s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:55,  2.07s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:06,  1.21s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:06,  1.21s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:43,  1.24it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:43,  1.24it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:30,  1.76it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:30,  1.76it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:22,  2.34it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:22,  2.34it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:16,  3.03it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:16,  3.03it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:15,  3.26it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:15,  3.26it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:13,  3.51it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:13,  3.51it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:12,  3.84it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:12,  3.84it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:12,  3.68it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:12,  3.68it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:12,  3.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:12,  3.71it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:11,  3.79it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:11,  3.79it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:11,  4.00it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:11,  4.00it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:09,  4.66it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:09,  4.66it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:07,  5.26it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:07,  5.26it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:06,  5.93it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:06,  5.93it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:06,  5.93it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:08<00:05,  7.34it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:08<00:05,  7.34it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:08<00:05,  7.34it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:08<00:04,  8.72it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:08<00:04,  8.72it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:08<00:04,  8.72it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:08<00:03, 10.63it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:08<00:03, 10.63it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:08<00:03, 10.63it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:08<00:02, 12.20it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:08<00:02, 12.20it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:08<00:02, 12.20it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:08<00:02, 12.20it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:08<00:02, 14.76it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:08<00:02, 14.76it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:08<00:02, 14.76it/s]

    Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:08<00:02, 14.76it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:08<00:01, 17.09it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:08<00:01, 17.09it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:08<00:01, 17.09it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:08<00:01, 17.09it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:08<00:01, 19.04it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:08<00:01, 19.04it/s]

    Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:08<00:01, 19.04it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:08<00:01, 19.04it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:08<00:00, 21.03it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:08<00:00, 21.03it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:08<00:00, 21.03it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:09<00:00, 21.03it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:09<00:00, 21.03it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:09<00:00, 23.93it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:09<00:00, 23.93it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:09<00:00, 23.93it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:09<00:00, 23.93it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:09<00:00, 24.91it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:09<00:00, 24.91it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:09<00:00, 24.91it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:09<00:00, 24.91it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:09<00:00, 24.91it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:09<00:00, 27.07it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:09<00:00, 27.07it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:09<00:00, 27.07it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:09<00:00, 27.07it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:09<00:00, 27.80it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:09<00:00, 27.80it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:09<00:00, 27.80it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:09<00:00, 27.80it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:09<00:00, 27.80it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:09<00:00, 29.82it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:09<00:00, 29.82it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:09<00:00, 29.82it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:09<00:00, 29.82it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  6.04it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=42.26 GB):   2%|▏         | 1/58 [00:00<00:31,  1.81it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.23 GB):   2%|▏         | 1/58 [00:00<00:31,  1.81it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.23 GB):   3%|▎         | 2/58 [00:00<00:26,  2.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.23 GB):   3%|▎         | 2/58 [00:00<00:26,  2.11it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.23 GB):   5%|▌         | 3/58 [00:01<00:23,  2.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.22 GB):   5%|▌         | 3/58 [00:01<00:23,  2.38it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.22 GB):   7%|▋         | 4/58 [00:01<00:20,  2.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.22 GB):   7%|▋         | 4/58 [00:01<00:20,  2.70it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.22 GB):   9%|▊         | 5/58 [00:01<00:16,  3.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.22 GB):   9%|▊         | 5/58 [00:01<00:16,  3.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.22 GB):  10%|█         | 6/58 [00:01<00:13,  3.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.22 GB):  10%|█         | 6/58 [00:01<00:13,  3.81it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.22 GB):  12%|█▏        | 7/58 [00:02<00:15,  3.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.25 GB):  12%|█▏        | 7/58 [00:02<00:15,  3.37it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=41.25 GB):  14%|█▍        | 8/58 [00:02<00:14,  3.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.19 GB):  14%|█▍        | 8/58 [00:02<00:14,  3.39it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.19 GB):  16%|█▌        | 9/58 [00:02<00:14,  3.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.37 GB):  16%|█▌        | 9/58 [00:02<00:14,  3.43it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=41.37 GB):  17%|█▋        | 10/58 [00:03<00:13,  3.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.19 GB):  17%|█▋        | 10/58 [00:03<00:13,  3.49it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.19 GB):  19%|█▉        | 11/58 [00:03<00:12,  3.68it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.42 GB):  19%|█▉        | 11/58 [00:03<00:12,  3.68it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=41.42 GB):  21%|██        | 12/58 [00:03<00:12,  3.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.18 GB):  21%|██        | 12/58 [00:03<00:12,  3.73it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=42.18 GB):  22%|██▏       | 13/58 [00:03<00:11,  3.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.47 GB):  22%|██▏       | 13/58 [00:03<00:11,  3.90it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=41.47 GB):  24%|██▍       | 14/58 [00:04<00:11,  4.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.18 GB):  24%|██▍       | 14/58 [00:04<00:11,  4.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.18 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.52 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.41it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=41.52 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.51 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.51 GB):  29%|██▉       | 17/58 [00:04<00:08,  4.80it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.57 GB):  29%|██▉       | 17/58 [00:04<00:08,  4.80it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=41.57 GB):  31%|███       | 18/58 [00:04<00:08,  4.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.61 GB):  31%|███       | 18/58 [00:04<00:08,  4.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.61 GB):  33%|███▎      | 19/58 [00:05<00:07,  5.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.07 GB):  33%|███▎      | 19/58 [00:05<00:07,  5.56it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=42.07 GB):  34%|███▍      | 20/58 [00:05<00:06,  5.79it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.61 GB):  34%|███▍      | 20/58 [00:05<00:06,  5.79it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.61 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.46it/s]Capturing num tokens (num_tokens=960 avail_mem=42.15 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.46it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=42.15 GB):  38%|███▊      | 22/58 [00:05<00:05,  7.09it/s]Capturing num tokens (num_tokens=896 avail_mem=41.65 GB):  38%|███▊      | 22/58 [00:05<00:05,  7.09it/s]Capturing num tokens (num_tokens=896 avail_mem=41.65 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.50it/s]Capturing num tokens (num_tokens=832 avail_mem=42.15 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.50it/s]

    Capturing num tokens (num_tokens=832 avail_mem=42.15 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.02it/s]Capturing num tokens (num_tokens=768 avail_mem=41.70 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.02it/s]Capturing num tokens (num_tokens=768 avail_mem=41.70 GB):  43%|████▎     | 25/58 [00:05<00:03,  8.36it/s]Capturing num tokens (num_tokens=704 avail_mem=42.14 GB):  43%|████▎     | 25/58 [00:05<00:03,  8.36it/s]Capturing num tokens (num_tokens=640 avail_mem=41.72 GB):  43%|████▎     | 25/58 [00:05<00:03,  8.36it/s]

    Capturing num tokens (num_tokens=640 avail_mem=41.72 GB):  47%|████▋     | 27/58 [00:05<00:03,  9.12it/s]Capturing num tokens (num_tokens=576 avail_mem=41.71 GB):  47%|████▋     | 27/58 [00:05<00:03,  9.12it/s]Capturing num tokens (num_tokens=512 avail_mem=42.12 GB):  47%|████▋     | 27/58 [00:06<00:03,  9.12it/s]Capturing num tokens (num_tokens=512 avail_mem=42.12 GB):  50%|█████     | 29/58 [00:06<00:02, 10.22it/s]Capturing num tokens (num_tokens=480 avail_mem=41.74 GB):  50%|█████     | 29/58 [00:06<00:02, 10.22it/s]

    Capturing num tokens (num_tokens=448 avail_mem=42.11 GB):  50%|█████     | 29/58 [00:06<00:02, 10.22it/s]Capturing num tokens (num_tokens=448 avail_mem=42.11 GB):  53%|█████▎    | 31/58 [00:06<00:02, 10.64it/s]Capturing num tokens (num_tokens=416 avail_mem=41.75 GB):  53%|█████▎    | 31/58 [00:06<00:02, 10.64it/s]Capturing num tokens (num_tokens=384 avail_mem=42.10 GB):  53%|█████▎    | 31/58 [00:06<00:02, 10.64it/s]

    Capturing num tokens (num_tokens=384 avail_mem=42.10 GB):  57%|█████▋    | 33/58 [00:06<00:02, 11.28it/s]Capturing num tokens (num_tokens=352 avail_mem=41.77 GB):  57%|█████▋    | 33/58 [00:06<00:02, 11.28it/s]Capturing num tokens (num_tokens=320 avail_mem=42.09 GB):  57%|█████▋    | 33/58 [00:06<00:02, 11.28it/s]Capturing num tokens (num_tokens=320 avail_mem=42.09 GB):  60%|██████    | 35/58 [00:06<00:01, 12.04it/s]Capturing num tokens (num_tokens=288 avail_mem=41.79 GB):  60%|██████    | 35/58 [00:06<00:01, 12.04it/s]

    Capturing num tokens (num_tokens=256 avail_mem=42.09 GB):  60%|██████    | 35/58 [00:06<00:01, 12.04it/s]Capturing num tokens (num_tokens=256 avail_mem=42.09 GB):  64%|██████▍   | 37/58 [00:06<00:01, 13.00it/s]Capturing num tokens (num_tokens=240 avail_mem=41.81 GB):  64%|██████▍   | 37/58 [00:06<00:01, 13.00it/s]Capturing num tokens (num_tokens=224 avail_mem=42.08 GB):  64%|██████▍   | 37/58 [00:06<00:01, 13.00it/s]Capturing num tokens (num_tokens=224 avail_mem=42.08 GB):  67%|██████▋   | 39/58 [00:06<00:01, 13.56it/s]Capturing num tokens (num_tokens=208 avail_mem=41.83 GB):  67%|██████▋   | 39/58 [00:06<00:01, 13.56it/s]

    Capturing num tokens (num_tokens=192 avail_mem=42.07 GB):  67%|██████▋   | 39/58 [00:06<00:01, 13.56it/s]Capturing num tokens (num_tokens=192 avail_mem=42.07 GB):  71%|███████   | 41/58 [00:06<00:01, 14.36it/s]Capturing num tokens (num_tokens=176 avail_mem=41.85 GB):  71%|███████   | 41/58 [00:06<00:01, 14.36it/s]Capturing num tokens (num_tokens=160 avail_mem=42.06 GB):  71%|███████   | 41/58 [00:07<00:01, 14.36it/s]Capturing num tokens (num_tokens=160 avail_mem=42.06 GB):  74%|███████▍  | 43/58 [00:07<00:00, 15.39it/s]Capturing num tokens (num_tokens=144 avail_mem=42.06 GB):  74%|███████▍  | 43/58 [00:07<00:00, 15.39it/s]

    Capturing num tokens (num_tokens=128 avail_mem=41.89 GB):  74%|███████▍  | 43/58 [00:07<00:00, 15.39it/s]Capturing num tokens (num_tokens=112 avail_mem=42.05 GB):  74%|███████▍  | 43/58 [00:07<00:00, 15.39it/s]Capturing num tokens (num_tokens=112 avail_mem=42.05 GB):  79%|███████▉  | 46/58 [00:07<00:00, 16.83it/s]Capturing num tokens (num_tokens=96 avail_mem=42.05 GB):  79%|███████▉  | 46/58 [00:07<00:00, 16.83it/s] Capturing num tokens (num_tokens=80 avail_mem=42.03 GB):  79%|███████▉  | 46/58 [00:07<00:00, 16.83it/s]Capturing num tokens (num_tokens=64 avail_mem=41.91 GB):  79%|███████▉  | 46/58 [00:07<00:00, 16.83it/s]

    Capturing num tokens (num_tokens=64 avail_mem=41.91 GB):  84%|████████▍ | 49/58 [00:07<00:00, 18.66it/s]Capturing num tokens (num_tokens=48 avail_mem=42.01 GB):  84%|████████▍ | 49/58 [00:07<00:00, 18.66it/s]Capturing num tokens (num_tokens=32 avail_mem=42.01 GB):  84%|████████▍ | 49/58 [00:07<00:00, 18.66it/s]Capturing num tokens (num_tokens=28 avail_mem=42.01 GB):  84%|████████▍ | 49/58 [00:07<00:00, 18.66it/s]Capturing num tokens (num_tokens=28 avail_mem=42.01 GB):  90%|████████▉ | 52/58 [00:07<00:00, 20.19it/s]Capturing num tokens (num_tokens=24 avail_mem=41.98 GB):  90%|████████▉ | 52/58 [00:07<00:00, 20.19it/s]Capturing num tokens (num_tokens=20 avail_mem=41.99 GB):  90%|████████▉ | 52/58 [00:07<00:00, 20.19it/s]

    Capturing num tokens (num_tokens=16 avail_mem=41.98 GB):  90%|████████▉ | 52/58 [00:07<00:00, 20.19it/s]Capturing num tokens (num_tokens=16 avail_mem=41.98 GB):  95%|█████████▍| 55/58 [00:07<00:00, 20.93it/s]Capturing num tokens (num_tokens=12 avail_mem=41.98 GB):  95%|█████████▍| 55/58 [00:07<00:00, 20.93it/s]Capturing num tokens (num_tokens=8 avail_mem=41.92 GB):  95%|█████████▍| 55/58 [00:07<00:00, 20.93it/s] Capturing num tokens (num_tokens=4 avail_mem=41.96 GB):  95%|█████████▍| 55/58 [00:07<00:00, 20.93it/s]Capturing num tokens (num_tokens=4 avail_mem=41.96 GB): 100%|██████████| 58/58 [00:07<00:00, 22.70it/s]Capturing num tokens (num_tokens=4 avail_mem=41.96 GB): 100%|██████████| 58/58 [00:07<00:00,  7.51it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


Note that `--reasoning-parser` defines the parser used to interpret responses.

### OpenAI Compatible API

Using the OpenAI compatible API, the contract follows the [DeepSeek API design](https://api-docs.deepseek.com/guides/reasoning_model) established with the release of DeepSeek-R1:

- `reasoning_content`: The content of the CoT.
- `content`: The content of the final answer.


```python
# Initialize OpenAI-like client
client = OpenAI(api_key="None", base_url=f"http://0.0.0.0:{port}/v1")
model_name = client.models.list().data[0].id

messages = [
    {
        "role": "user",
        "content": "What is 1+3?",
    }
]
```

#### Non-Streaming Request


```python
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=False,  # Non-streaming
    extra_body={"separate_reasoning": True},
)
print_highlight("==== Reasoning ====")
print_highlight(response_non_stream.choices[0].message.reasoning_content)

print_highlight("==== Text ====")
print_highlight(response_non_stream.choices[0].message.content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I arrive at the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Certainly! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Start with the number 1.**<br>2. **Add 3 to it.**<br>   <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>


#### Streaming Request


```python
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=True,  # Non-streaming
    extra_body={"separate_reasoning": True},
)

reasoning_content = ""
content = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content

print_highlight("==== Reasoning ====")
print_highlight(reasoning_content)

print_highlight("==== Text ====")
print_highlight(content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>I need to calculate the sum of 1 and 3.<br><br>First, I add the numbers together: 1 plus 3 equals 4.<br><br>Therefore, the answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>To find the sum of 1 and 3, follow these simple steps:<br><br>1. **Start with the first number:**  <br>   \[<br>   1<br>   \]<br><br>2. **Add the second number to it:**  <br>   \[<br>   1 + 3<br>   \]<br><br>3. **Calculate the total:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**  <br>\[<br>\boxed{4}<br>\]</strong>


Optionally, you can buffer the reasoning content to the last reasoning chunk (or the first chunk after the reasoning content).


```python
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=True,  # Non-streaming
    extra_body={"separate_reasoning": True, "stream_reasoning": False},
)

reasoning_content = ""
content = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content

print_highlight("==== Reasoning ====")
print_highlight(reasoning_content)

print_highlight("==== Text ====")
print_highlight(content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


The reasoning separation is enable by default when specify . 
**To disable it, set the `separate_reasoning` option to `False` in request.**


```python
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=False,  # Non-streaming
    extra_body={"separate_reasoning": False},
)

print_highlight("==== Original Output ====")
print_highlight(response_non_stream.choices[0].message.content)
```


<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>


### SGLang Native API 


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)

gen_url = f"http://localhost:{port}/generate"
gen_data = {
    "text": input,
    "sampling_params": {
        "skip_special_tokens": False,
        "max_new_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
    },
}
gen_response = requests.post(gen_url, json=gen_data).json()["text"]

print_highlight("==== Original Output ====")
print_highlight(gen_response)

parse_url = f"http://localhost:{port}/separate_reasoning"
separate_reasoning_data = {
    "text": gen_response,
    "reasoning_parser": "deepseek-r1",
}
separate_reasoning_response_json = requests.post(
    parse_url, json=separate_reasoning_data
).json()
print_highlight("==== Reasoning ====")
print_highlight(separate_reasoning_response_json["reasoning_text"])
print_highlight("==== Text ====")
print_highlight(separate_reasoning_response_json["text"])
```


<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I perform the addition of these two numbers: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Add the numbers:**<br>   <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>2. **Final Answer:**<br>   <br>   \[<br>   \boxed{4}<br>   \]</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I perform the addition of these two numbers: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Add the numbers:**<br>   <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>2. **Final Answer:**<br>   <br>   \[<br>   \boxed{4}<br>   \]</strong>



```python
terminate_process(server_process)
```

### Offline Engine API


```python
import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.utils import print_highlight

llm = sgl.Engine(model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
sampling_params = {
    "max_new_tokens": 1024,
    "skip_special_tokens": False,
    "temperature": 0.6,
    "top_p": 0.95,
}
result = llm.generate(prompt=input, sampling_params=sampling_params)

generated_text = result["text"]  # Assume there is only one prompt

print_highlight("==== Original Output ====")
print_highlight(generated_text)

parser = ReasoningParser("deepseek-r1")
reasoning_text, text = parser.parse_non_stream(generated_text)
print_highlight("==== Reasoning ====")
print_highlight(reasoning_text)
print_highlight("==== Text ====")
print_highlight(text)
```

    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.40s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:37,  4.87s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:37,  4.87s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<01:59,  2.13s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<01:59,  2.13s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:08,  1.25s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:08,  1.25s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:44,  1.21it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:44,  1.21it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:31,  1.71it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:31,  1.71it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:22,  2.27it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:22,  2.27it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:17,  2.95it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:17,  2.95it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.69it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.69it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.55it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.55it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.55it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.25it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.25it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.25it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.81it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.81it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.81it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.36it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.36it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.36it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.13it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.13it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.13it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:06<00:03, 11.13it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:02, 14.49it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:02, 14.49it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:02, 14.49it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:06<00:02, 14.49it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:06<00:02, 14.49it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:06<00:01, 20.03it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:06<00:01, 20.03it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:06<00:01, 20.03it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 20.03it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.03it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.03it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.03it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]

    Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.58it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.58it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.58it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.58it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.58it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.58it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.58it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.58it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.58it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 46.73it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 46.73it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 46.73it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 46.73it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 46.73it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 46.73it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 46.73it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 46.73it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 46.73it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:07<00:00, 46.73it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:07<00:00, 56.32it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:07<00:00, 56.32it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:07<00:00, 56.32it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:07<00:00, 56.32it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:07<00:00, 56.32it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:07<00:00, 56.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.20 GB):   2%|▏         | 1/58 [00:00<00:16,  3.39it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.16 GB):   2%|▏         | 1/58 [00:00<00:16,  3.39it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.16 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.16 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.16 GB):   5%|▌         | 3/58 [00:00<00:14,  3.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.16 GB):   5%|▌         | 3/58 [00:00<00:14,  3.78it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.16 GB):   7%|▋         | 4/58 [00:01<00:13,  4.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.16 GB):   7%|▋         | 4/58 [00:01<00:13,  4.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.16 GB):   9%|▊         | 5/58 [00:01<00:12,  4.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.16 GB):   9%|▊         | 5/58 [00:01<00:12,  4.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.16 GB):  10%|█         | 6/58 [00:01<00:10,  4.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.16 GB):  10%|█         | 6/58 [00:01<00:10,  4.74it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.16 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.16 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.16 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.16 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.63it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.16 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.16 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.16 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.16 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.59it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.16 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.16 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.16 GB):  21%|██        | 12/58 [00:02<00:05,  7.78it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.16 GB):  21%|██        | 12/58 [00:02<00:05,  7.78it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.16 GB):  21%|██        | 12/58 [00:02<00:05,  7.78it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.16 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.15 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.15 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.39it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.15 GB):  31%|███       | 18/58 [00:02<00:03, 12.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.14 GB):  31%|███       | 18/58 [00:02<00:03, 12.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.14 GB):  31%|███       | 18/58 [00:02<00:03, 12.00it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=43.14 GB):  34%|███▍      | 20/58 [00:02<00:03, 12.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.19 GB):  34%|███▍      | 20/58 [00:02<00:03, 12.14it/s]Capturing num tokens (num_tokens=960 avail_mem=42.19 GB):  34%|███▍      | 20/58 [00:02<00:03, 12.14it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=42.19 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.17it/s]Capturing num tokens (num_tokens=896 avail_mem=42.18 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.17it/s]Capturing num tokens (num_tokens=832 avail_mem=42.18 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.17it/s]Capturing num tokens (num_tokens=768 avail_mem=42.18 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.17it/s]Capturing num tokens (num_tokens=768 avail_mem=42.18 GB):  43%|████▎     | 25/58 [00:03<00:02, 13.80it/s]Capturing num tokens (num_tokens=704 avail_mem=42.17 GB):  43%|████▎     | 25/58 [00:03<00:02, 13.80it/s]Capturing num tokens (num_tokens=640 avail_mem=42.17 GB):  43%|████▎     | 25/58 [00:03<00:02, 13.80it/s]Capturing num tokens (num_tokens=576 avail_mem=42.17 GB):  43%|████▎     | 25/58 [00:03<00:02, 13.80it/s]

    Capturing num tokens (num_tokens=576 avail_mem=42.17 GB):  48%|████▊     | 28/58 [00:03<00:01, 17.24it/s]Capturing num tokens (num_tokens=512 avail_mem=42.16 GB):  48%|████▊     | 28/58 [00:03<00:01, 17.24it/s]Capturing num tokens (num_tokens=480 avail_mem=42.16 GB):  48%|████▊     | 28/58 [00:03<00:01, 17.24it/s]Capturing num tokens (num_tokens=448 avail_mem=42.16 GB):  48%|████▊     | 28/58 [00:03<00:01, 17.24it/s]Capturing num tokens (num_tokens=416 avail_mem=42.15 GB):  48%|████▊     | 28/58 [00:03<00:01, 17.24it/s]Capturing num tokens (num_tokens=416 avail_mem=42.15 GB):  55%|█████▌    | 32/58 [00:03<00:01, 21.66it/s]Capturing num tokens (num_tokens=384 avail_mem=42.15 GB):  55%|█████▌    | 32/58 [00:03<00:01, 21.66it/s]Capturing num tokens (num_tokens=352 avail_mem=42.14 GB):  55%|█████▌    | 32/58 [00:03<00:01, 21.66it/s]Capturing num tokens (num_tokens=320 avail_mem=42.14 GB):  55%|█████▌    | 32/58 [00:03<00:01, 21.66it/s]Capturing num tokens (num_tokens=288 avail_mem=42.15 GB):  55%|█████▌    | 32/58 [00:03<00:01, 21.66it/s]

    Capturing num tokens (num_tokens=288 avail_mem=42.15 GB):  62%|██████▏   | 36/58 [00:03<00:00, 25.64it/s]Capturing num tokens (num_tokens=256 avail_mem=42.14 GB):  62%|██████▏   | 36/58 [00:03<00:00, 25.64it/s]Capturing num tokens (num_tokens=240 avail_mem=42.14 GB):  62%|██████▏   | 36/58 [00:03<00:00, 25.64it/s]Capturing num tokens (num_tokens=224 avail_mem=42.13 GB):  62%|██████▏   | 36/58 [00:03<00:00, 25.64it/s]Capturing num tokens (num_tokens=208 avail_mem=42.13 GB):  62%|██████▏   | 36/58 [00:03<00:00, 25.64it/s]Capturing num tokens (num_tokens=208 avail_mem=42.13 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.17it/s]Capturing num tokens (num_tokens=192 avail_mem=42.13 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.17it/s]Capturing num tokens (num_tokens=176 avail_mem=42.12 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.17it/s]Capturing num tokens (num_tokens=160 avail_mem=42.12 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.17it/s]Capturing num tokens (num_tokens=144 avail_mem=42.11 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.17it/s]

    Capturing num tokens (num_tokens=144 avail_mem=42.11 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.89it/s]Capturing num tokens (num_tokens=128 avail_mem=42.12 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.89it/s]Capturing num tokens (num_tokens=112 avail_mem=42.11 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.89it/s]Capturing num tokens (num_tokens=96 avail_mem=42.11 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.89it/s] Capturing num tokens (num_tokens=80 avail_mem=42.10 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.89it/s]Capturing num tokens (num_tokens=64 avail_mem=42.10 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.89it/s]Capturing num tokens (num_tokens=64 avail_mem=42.10 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.69it/s]Capturing num tokens (num_tokens=48 avail_mem=42.10 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.69it/s]Capturing num tokens (num_tokens=32 avail_mem=42.09 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.69it/s]Capturing num tokens (num_tokens=28 avail_mem=42.09 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.69it/s]Capturing num tokens (num_tokens=24 avail_mem=42.09 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.69it/s]

    Capturing num tokens (num_tokens=20 avail_mem=42.08 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.69it/s]Capturing num tokens (num_tokens=20 avail_mem=42.08 GB):  93%|█████████▎| 54/58 [00:03<00:00, 36.74it/s]Capturing num tokens (num_tokens=16 avail_mem=42.08 GB):  93%|█████████▎| 54/58 [00:03<00:00, 36.74it/s]Capturing num tokens (num_tokens=12 avail_mem=42.08 GB):  93%|█████████▎| 54/58 [00:03<00:00, 36.74it/s]Capturing num tokens (num_tokens=8 avail_mem=42.07 GB):  93%|█████████▎| 54/58 [00:03<00:00, 36.74it/s] Capturing num tokens (num_tokens=4 avail_mem=42.07 GB):  93%|█████████▎| 54/58 [00:04<00:00, 36.74it/s]Capturing num tokens (num_tokens=4 avail_mem=42.07 GB): 100%|██████████| 58/58 [00:04<00:00, 37.05it/s]Capturing num tokens (num_tokens=4 avail_mem=42.07 GB): 100%|██████████| 58/58 [00:04<00:00, 14.35it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I need to identify the two numbers in the problem, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Finally, the sum of 1 and 3 is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of two numbers: 1 and 3.<br><br>1. **Identify the numbers:**<br>   - First number: 1<br>   - Second number: 3<br><br>2. **Add the numbers:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I need to identify the two numbers in the problem, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Finally, the sum of 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of two numbers: 1 and 3.<br><br>1. **Identify the numbers:**<br>   - First number: 1<br>   - Second number: 3<br><br>2. **Add the numbers:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
