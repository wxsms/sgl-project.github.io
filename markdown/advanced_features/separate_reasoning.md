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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.60s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.68s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.67s/it]


    2026-05-02 19:27:06,187 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 19:27:06] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:39,  4.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:39,  4.91s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<01:59,  2.14s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<01:59,  2.14s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:08,  1.25s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:08,  1.25s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:44,  1.21it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:44,  1.21it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:30,  1.71it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:30,  1.71it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:22,  2.29it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:22,  2.29it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:17,  2.98it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:17,  2.98it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.75it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.75it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.64it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.64it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.64it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.36it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.36it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.36it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.96it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.96it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.96it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.60it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.60it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.60it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.56it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.56it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.56it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:06<00:03, 11.56it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:02, 14.92it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:02, 14.92it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:02, 14.92it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:06<00:02, 14.92it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:06<00:02, 14.92it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:06<00:01, 20.46it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:06<00:01, 20.46it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:06<00:01, 20.46it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:06<00:01, 20.46it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.46it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.46it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 25.25it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 25.25it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 25.25it/s]

    Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 25.25it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:07<00:01, 24.25it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:07<00:01, 24.25it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:07<00:01, 24.25it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:07<00:01, 24.25it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 24.44it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 24.44it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 24.44it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 24.44it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:07<00:00, 25.66it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:07<00:00, 25.66it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:07<00:00, 25.66it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:07<00:00, 25.66it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:07<00:00, 26.61it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:07<00:00, 26.61it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:07<00:00, 26.61it/s]

    Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:07<00:00, 26.61it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 22.65it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 22.65it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 22.65it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 22.65it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 22.65it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:07<00:00, 24.75it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:07<00:00, 24.75it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:07<00:00, 24.75it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:07<00:00, 24.75it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:07<00:00, 25.71it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:07<00:00, 25.71it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:07<00:00, 25.71it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:08<00:00, 25.71it/s]

    Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:08<00:00, 25.71it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:08<00:00, 28.78it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:08<00:00, 28.78it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:08<00:00, 28.78it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:08<00:00, 28.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=23.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=23.17 GB):   2%|▏         | 1/58 [00:00<00:41,  1.39it/s]Capturing num tokens (num_tokens=7680 avail_mem=23.14 GB):   2%|▏         | 1/58 [00:00<00:41,  1.39it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=23.14 GB):   3%|▎         | 2/58 [00:01<00:34,  1.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=23.14 GB):   3%|▎         | 2/58 [00:01<00:34,  1.61it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=23.14 GB):   5%|▌         | 3/58 [00:01<00:30,  1.77it/s]Capturing num tokens (num_tokens=6656 avail_mem=23.14 GB):   5%|▌         | 3/58 [00:01<00:30,  1.77it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=23.14 GB):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=23.12 GB):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=23.12 GB):   9%|▊         | 5/58 [00:02<00:25,  2.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=23.12 GB):   9%|▊         | 5/58 [00:02<00:25,  2.10it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=23.12 GB):  10%|█         | 6/58 [00:03<00:24,  2.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=23.11 GB):  10%|█         | 6/58 [00:03<00:24,  2.16it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=23.11 GB):  12%|█▏        | 7/58 [00:03<00:23,  2.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=23.11 GB):  12%|█▏        | 7/58 [00:03<00:23,  2.20it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=23.11 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=23.11 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.61it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=23.11 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.22 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.22 GB):  17%|█▋        | 10/58 [00:04<00:13,  3.56it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.22 GB):  17%|█▋        | 10/58 [00:04<00:13,  3.56it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.22 GB):  19%|█▉        | 11/58 [00:04<00:10,  4.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.22 GB):  19%|█▉        | 11/58 [00:04<00:10,  4.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.22 GB):  21%|██        | 12/58 [00:04<00:08,  5.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.21 GB):  21%|██        | 12/58 [00:04<00:08,  5.23it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=42.21 GB):  21%|██        | 12/58 [00:04<00:08,  5.23it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.21 GB):  24%|██▍       | 14/58 [00:04<00:06,  6.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.24 GB):  24%|██▍       | 14/58 [00:04<00:06,  6.41it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=41.24 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.24 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.24 GB):  28%|██▊       | 16/58 [00:04<00:07,  5.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.17 GB):  28%|██▊       | 16/58 [00:04<00:07,  5.83it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=42.17 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.35 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.35 GB):  31%|███       | 18/58 [00:05<00:06,  5.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.35 GB):  31%|███       | 18/58 [00:05<00:06,  5.94it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=41.35 GB):  33%|███▎      | 19/58 [00:05<00:06,  6.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.17 GB):  33%|███▎      | 19/58 [00:05<00:06,  6.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.17 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.91it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.16 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.91it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=42.16 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.39it/s]Capturing num tokens (num_tokens=960 avail_mem=41.39 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.39it/s] Capturing num tokens (num_tokens=960 avail_mem=41.39 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.38it/s]Capturing num tokens (num_tokens=896 avail_mem=41.38 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.38it/s]

    Capturing num tokens (num_tokens=896 avail_mem=41.38 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.53it/s]Capturing num tokens (num_tokens=832 avail_mem=42.15 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.53it/s]Capturing num tokens (num_tokens=832 avail_mem=42.15 GB):  41%|████▏     | 24/58 [00:06<00:04,  7.79it/s]Capturing num tokens (num_tokens=768 avail_mem=41.43 GB):  41%|████▏     | 24/58 [00:06<00:04,  7.79it/s]

    Capturing num tokens (num_tokens=768 avail_mem=41.43 GB):  43%|████▎     | 25/58 [00:06<00:04,  7.88it/s]Capturing num tokens (num_tokens=704 avail_mem=41.42 GB):  43%|████▎     | 25/58 [00:06<00:04,  7.88it/s]Capturing num tokens (num_tokens=704 avail_mem=41.42 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.14it/s]Capturing num tokens (num_tokens=640 avail_mem=42.13 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.14it/s]

    Capturing num tokens (num_tokens=576 avail_mem=41.47 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.14it/s]Capturing num tokens (num_tokens=576 avail_mem=41.47 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.59it/s]Capturing num tokens (num_tokens=512 avail_mem=41.47 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.59it/s]

    Capturing num tokens (num_tokens=480 avail_mem=42.12 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.59it/s]Capturing num tokens (num_tokens=480 avail_mem=42.12 GB):  52%|█████▏    | 30/58 [00:06<00:02,  9.35it/s]Capturing num tokens (num_tokens=448 avail_mem=41.52 GB):  52%|█████▏    | 30/58 [00:06<00:02,  9.35it/s]

    Capturing num tokens (num_tokens=448 avail_mem=41.52 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.00it/s]Capturing num tokens (num_tokens=416 avail_mem=41.51 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.00it/s]Capturing num tokens (num_tokens=416 avail_mem=41.51 GB):  55%|█████▌    | 32/58 [00:06<00:02,  9.12it/s]Capturing num tokens (num_tokens=384 avail_mem=42.11 GB):  55%|█████▌    | 32/58 [00:06<00:02,  9.12it/s]

    Capturing num tokens (num_tokens=384 avail_mem=42.11 GB):  57%|█████▋    | 33/58 [00:06<00:02,  8.97it/s]Capturing num tokens (num_tokens=352 avail_mem=41.56 GB):  57%|█████▋    | 33/58 [00:06<00:02,  8.97it/s]Capturing num tokens (num_tokens=352 avail_mem=41.56 GB):  59%|█████▊    | 34/58 [00:07<00:02,  8.82it/s]Capturing num tokens (num_tokens=320 avail_mem=42.10 GB):  59%|█████▊    | 34/58 [00:07<00:02,  8.82it/s]

    Capturing num tokens (num_tokens=320 avail_mem=42.10 GB):  60%|██████    | 35/58 [00:07<00:02,  8.36it/s]Capturing num tokens (num_tokens=288 avail_mem=41.61 GB):  60%|██████    | 35/58 [00:07<00:02,  8.36it/s]Capturing num tokens (num_tokens=288 avail_mem=41.61 GB):  62%|██████▏   | 36/58 [00:07<00:02,  8.71it/s]Capturing num tokens (num_tokens=256 avail_mem=41.66 GB):  62%|██████▏   | 36/58 [00:07<00:02,  8.71it/s]

    Capturing num tokens (num_tokens=256 avail_mem=41.66 GB):  64%|██████▍   | 37/58 [00:07<00:02,  8.93it/s]Capturing num tokens (num_tokens=240 avail_mem=42.10 GB):  64%|██████▍   | 37/58 [00:07<00:02,  8.93it/s]Capturing num tokens (num_tokens=224 avail_mem=41.99 GB):  64%|██████▍   | 37/58 [00:07<00:02,  8.93it/s]Capturing num tokens (num_tokens=224 avail_mem=41.99 GB):  67%|██████▋   | 39/58 [00:07<00:01,  9.82it/s]Capturing num tokens (num_tokens=208 avail_mem=41.88 GB):  67%|██████▋   | 39/58 [00:07<00:01,  9.82it/s]

    Capturing num tokens (num_tokens=192 avail_mem=41.70 GB):  67%|██████▋   | 39/58 [00:07<00:01,  9.82it/s]Capturing num tokens (num_tokens=192 avail_mem=41.70 GB):  71%|███████   | 41/58 [00:07<00:01, 10.11it/s]Capturing num tokens (num_tokens=176 avail_mem=41.80 GB):  71%|███████   | 41/58 [00:07<00:01, 10.11it/s]Capturing num tokens (num_tokens=160 avail_mem=42.08 GB):  71%|███████   | 41/58 [00:07<00:01, 10.11it/s]

    Capturing num tokens (num_tokens=160 avail_mem=42.08 GB):  74%|███████▍  | 43/58 [00:07<00:01, 11.25it/s]Capturing num tokens (num_tokens=144 avail_mem=41.71 GB):  74%|███████▍  | 43/58 [00:07<00:01, 11.25it/s]Capturing num tokens (num_tokens=128 avail_mem=42.07 GB):  74%|███████▍  | 43/58 [00:08<00:01, 11.25it/s]Capturing num tokens (num_tokens=128 avail_mem=42.07 GB):  78%|███████▊  | 45/58 [00:08<00:01, 11.93it/s]Capturing num tokens (num_tokens=112 avail_mem=41.74 GB):  78%|███████▊  | 45/58 [00:08<00:01, 11.93it/s]

    Capturing num tokens (num_tokens=96 avail_mem=42.06 GB):  78%|███████▊  | 45/58 [00:08<00:01, 11.93it/s] Capturing num tokens (num_tokens=96 avail_mem=42.06 GB):  81%|████████  | 47/58 [00:08<00:00, 12.49it/s]Capturing num tokens (num_tokens=80 avail_mem=41.77 GB):  81%|████████  | 47/58 [00:08<00:00, 12.49it/s]Capturing num tokens (num_tokens=64 avail_mem=42.05 GB):  81%|████████  | 47/58 [00:08<00:00, 12.49it/s]

    Capturing num tokens (num_tokens=64 avail_mem=42.05 GB):  84%|████████▍ | 49/58 [00:08<00:00, 13.56it/s]Capturing num tokens (num_tokens=48 avail_mem=42.05 GB):  84%|████████▍ | 49/58 [00:08<00:00, 13.56it/s]Capturing num tokens (num_tokens=32 avail_mem=41.79 GB):  84%|████████▍ | 49/58 [00:08<00:00, 13.56it/s]Capturing num tokens (num_tokens=32 avail_mem=41.79 GB):  88%|████████▊ | 51/58 [00:08<00:00, 14.43it/s]Capturing num tokens (num_tokens=28 avail_mem=42.04 GB):  88%|████████▊ | 51/58 [00:08<00:00, 14.43it/s]Capturing num tokens (num_tokens=24 avail_mem=41.81 GB):  88%|████████▊ | 51/58 [00:08<00:00, 14.43it/s]

    Capturing num tokens (num_tokens=24 avail_mem=41.81 GB):  91%|█████████▏| 53/58 [00:08<00:00, 15.66it/s]Capturing num tokens (num_tokens=20 avail_mem=42.03 GB):  91%|█████████▏| 53/58 [00:08<00:00, 15.66it/s]Capturing num tokens (num_tokens=16 avail_mem=41.83 GB):  91%|█████████▏| 53/58 [00:08<00:00, 15.66it/s]Capturing num tokens (num_tokens=12 avail_mem=42.01 GB):  91%|█████████▏| 53/58 [00:08<00:00, 15.66it/s]Capturing num tokens (num_tokens=12 avail_mem=42.01 GB):  97%|█████████▋| 56/58 [00:08<00:00, 16.94it/s]Capturing num tokens (num_tokens=8 avail_mem=42.01 GB):  97%|█████████▋| 56/58 [00:08<00:00, 16.94it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=41.87 GB):  97%|█████████▋| 56/58 [00:08<00:00, 16.94it/s]Capturing num tokens (num_tokens=4 avail_mem=41.87 GB): 100%|██████████| 58/58 [00:08<00:00,  6.56it/s]


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



<strong style='color: #00008B;'>First, I need to identify the two numbers in the addition problem: 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Finally, the sum of 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I'll add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>To solve the problem \(1 + 3\), I start by identifying the numbers involved.<br><br>Next, I add the two numbers together.<br><br>Finally, I conclude that the sum is 4.</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**  <br>1. **Identify the numbers to add:**  <br>   We have the numbers 1 and 3.<br><br>2. **Add the numbers together:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**  <br>   \[<br>   \boxed{4}<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I identify the numbers to be added, which are 1 and 3.<br><br>Next, I add these two numbers together.<br><br>Finally, I arrive at the sum, which is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.28s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.25s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.25s/it]


    2026-05-02 19:27:49,381 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 19:27:49] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<01:59,  2.14s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<01:59,  2.14s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:08,  1.25s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:08,  1.25s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:44,  1.21it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:44,  1.21it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:30,  1.71it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:30,  1.71it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:22,  2.29it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:22,  2.29it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:17,  2.97it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:17,  2.97it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.73it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.73it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.61it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.61it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.61it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.32it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.32it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.32it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.91it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.91it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.91it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.53it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.53it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.53it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.49it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.49it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.49it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:06<00:03, 11.49it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:02, 14.83it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:02, 14.83it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:02, 14.83it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:06<00:02, 14.83it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:06<00:02, 14.83it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:06<00:01, 20.39it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:06<00:01, 20.39it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:06<00:01, 20.39it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:06<00:01, 20.39it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.39it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.39it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.39it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 29.43it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 29.43it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 29.43it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 29.43it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 29.43it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 29.43it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 29.43it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 37.01it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 37.01it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 37.01it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 37.01it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 37.01it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 37.01it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 37.01it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 37.01it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 37.01it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 47.11it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 47.11it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 47.11it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 47.11it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 47.11it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 47.11it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 47.11it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 47.11it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 47.11it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 55.67it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 55.67it/s]

    Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 55.67it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 55.67it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 55.67it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 55.67it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 55.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.81it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=45.99 GB):   2%|▏         | 1/58 [00:00<00:17,  3.30it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.95 GB):   2%|▏         | 1/58 [00:00<00:17,  3.30it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=45.95 GB):   3%|▎         | 2/58 [00:00<00:15,  3.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.95 GB):   3%|▎         | 2/58 [00:00<00:15,  3.57it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=45.95 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.16 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.16 GB):   7%|▋         | 4/58 [00:01<00:13,  4.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.16 GB):   7%|▋         | 4/58 [00:01<00:13,  4.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.16 GB):   9%|▊         | 5/58 [00:01<00:12,  4.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.16 GB):   9%|▊         | 5/58 [00:01<00:12,  4.36it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.16 GB):  10%|█         | 6/58 [00:01<00:10,  4.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.16 GB):  10%|█         | 6/58 [00:01<00:10,  4.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.16 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.16 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.15it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.16 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.16 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.16 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.13it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.16 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.13it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.16 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.16 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.16 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.16 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.18it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=43.16 GB):  21%|██        | 12/58 [00:02<00:05,  7.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.16 GB):  21%|██        | 12/58 [00:02<00:05,  7.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.16 GB):  21%|██        | 12/58 [00:02<00:05,  7.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.16 GB):  24%|██▍       | 14/58 [00:02<00:04,  9.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.15 GB):  24%|██▍       | 14/58 [00:02<00:04,  9.01it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=43.15 GB):  24%|██▍       | 14/58 [00:02<00:04,  9.01it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.15 GB):  31%|███       | 18/58 [00:02<00:03, 11.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.14 GB):  31%|███       | 18/58 [00:02<00:03, 11.95it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=43.14 GB):  31%|███       | 18/58 [00:02<00:03, 11.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.13 GB):  31%|███       | 18/58 [00:02<00:03, 11.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.13 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.94it/s]Capturing num tokens (num_tokens=960 avail_mem=43.13 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.94it/s] Capturing num tokens (num_tokens=896 avail_mem=43.12 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.94it/s]

    Capturing num tokens (num_tokens=896 avail_mem=43.12 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.55it/s]Capturing num tokens (num_tokens=832 avail_mem=43.12 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.55it/s]Capturing num tokens (num_tokens=768 avail_mem=43.11 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.55it/s]Capturing num tokens (num_tokens=704 avail_mem=43.11 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.55it/s]Capturing num tokens (num_tokens=704 avail_mem=43.11 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.66it/s]Capturing num tokens (num_tokens=640 avail_mem=43.11 GB):  45%|████▍     | 26/58 [00:02<00:01, 18.66it/s]Capturing num tokens (num_tokens=576 avail_mem=43.10 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Capturing num tokens (num_tokens=512 avail_mem=43.10 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]

    Capturing num tokens (num_tokens=480 avail_mem=43.10 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.66it/s]Capturing num tokens (num_tokens=480 avail_mem=43.10 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.58it/s]Capturing num tokens (num_tokens=448 avail_mem=43.09 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.58it/s]Capturing num tokens (num_tokens=416 avail_mem=43.09 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.58it/s]Capturing num tokens (num_tokens=384 avail_mem=43.09 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.58it/s]Capturing num tokens (num_tokens=352 avail_mem=43.08 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.58it/s]Capturing num tokens (num_tokens=352 avail_mem=43.08 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.63it/s]Capturing num tokens (num_tokens=320 avail_mem=43.08 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.63it/s]Capturing num tokens (num_tokens=288 avail_mem=43.08 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.63it/s]

    Capturing num tokens (num_tokens=256 avail_mem=43.08 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.63it/s]Capturing num tokens (num_tokens=240 avail_mem=43.07 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.63it/s]Capturing num tokens (num_tokens=240 avail_mem=43.07 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.82it/s]Capturing num tokens (num_tokens=224 avail_mem=43.07 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.82it/s]Capturing num tokens (num_tokens=208 avail_mem=43.07 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.82it/s]Capturing num tokens (num_tokens=192 avail_mem=43.06 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.82it/s]Capturing num tokens (num_tokens=176 avail_mem=43.06 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.82it/s]Capturing num tokens (num_tokens=176 avail_mem=43.06 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.48it/s]Capturing num tokens (num_tokens=160 avail_mem=43.06 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.48it/s]Capturing num tokens (num_tokens=144 avail_mem=43.05 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.48it/s]

    Capturing num tokens (num_tokens=128 avail_mem=43.05 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.48it/s]Capturing num tokens (num_tokens=112 avail_mem=43.05 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.48it/s]Capturing num tokens (num_tokens=112 avail_mem=43.05 GB):  79%|███████▉  | 46/58 [00:03<00:00, 33.39it/s]Capturing num tokens (num_tokens=96 avail_mem=43.05 GB):  79%|███████▉  | 46/58 [00:03<00:00, 33.39it/s] Capturing num tokens (num_tokens=80 avail_mem=43.04 GB):  79%|███████▉  | 46/58 [00:03<00:00, 33.39it/s]Capturing num tokens (num_tokens=64 avail_mem=43.04 GB):  79%|███████▉  | 46/58 [00:03<00:00, 33.39it/s]Capturing num tokens (num_tokens=48 avail_mem=43.03 GB):  79%|███████▉  | 46/58 [00:03<00:00, 33.39it/s]Capturing num tokens (num_tokens=48 avail_mem=43.03 GB):  86%|████████▌ | 50/58 [00:03<00:00, 35.14it/s]Capturing num tokens (num_tokens=32 avail_mem=43.03 GB):  86%|████████▌ | 50/58 [00:03<00:00, 35.14it/s]Capturing num tokens (num_tokens=28 avail_mem=43.03 GB):  86%|████████▌ | 50/58 [00:03<00:00, 35.14it/s]

    Capturing num tokens (num_tokens=24 avail_mem=43.03 GB):  86%|████████▌ | 50/58 [00:03<00:00, 35.14it/s]Capturing num tokens (num_tokens=20 avail_mem=43.02 GB):  86%|████████▌ | 50/58 [00:03<00:00, 35.14it/s]Capturing num tokens (num_tokens=16 avail_mem=43.02 GB):  86%|████████▌ | 50/58 [00:03<00:00, 35.14it/s]Capturing num tokens (num_tokens=16 avail_mem=43.02 GB):  95%|█████████▍| 55/58 [00:03<00:00, 36.56it/s]Capturing num tokens (num_tokens=12 avail_mem=43.01 GB):  95%|█████████▍| 55/58 [00:03<00:00, 36.56it/s]Capturing num tokens (num_tokens=8 avail_mem=43.01 GB):  95%|█████████▍| 55/58 [00:03<00:00, 36.56it/s] Capturing num tokens (num_tokens=4 avail_mem=43.01 GB):  95%|█████████▍| 55/58 [00:03<00:00, 36.56it/s]Capturing num tokens (num_tokens=4 avail_mem=43.01 GB): 100%|██████████| 58/58 [00:03<00:00, 15.02it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I arrive at the result.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I arrive at the result.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
