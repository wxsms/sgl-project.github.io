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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.48s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.36s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<01:58,  2.11s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<01:58,  2.11s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:08,  1.24s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:08,  1.24s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:49,  1.09it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:49,  1.09it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:30,  1.71it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:30,  1.71it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:25,  2.03it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:25,  2.03it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:21,  2.37it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:21,  2.37it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:17,  2.76it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:17,  2.76it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:15,  3.14it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:15,  3.14it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:13,  3.54it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:13,  3.54it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:11,  3.95it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:11,  3.95it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:10,  4.35it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:10,  4.35it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:09,  4.79it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:09,  4.79it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:08,  5.24it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:08,  5.24it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:07,  5.86it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:07,  5.86it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:08<00:06,  6.43it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:08<00:06,  6.43it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:08<00:06,  6.43it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:08<00:05,  7.73it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:08<00:05,  7.73it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:08<00:05,  7.73it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:08<00:04,  9.05it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:08<00:04,  9.05it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:08<00:04,  9.05it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:08<00:03, 10.82it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:08<00:03, 10.82it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:09<00:03, 10.82it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:09<00:02, 12.48it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:09<00:02, 12.48it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:09<00:02, 12.48it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:09<00:02, 14.08it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:09<00:02, 14.08it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:09<00:02, 14.08it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:09<00:02, 14.08it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:09<00:01, 16.88it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:09<00:01, 16.88it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:09<00:01, 16.88it/s]

    Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:09<00:01, 16.88it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:09<00:01, 19.22it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:09<00:01, 19.22it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:09<00:01, 19.22it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:09<00:01, 19.22it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:09<00:01, 20.79it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:09<00:01, 20.79it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:09<00:01, 20.79it/s]

    Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:09<00:01, 20.79it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:09<00:01, 20.79it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:09<00:00, 23.61it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:09<00:00, 23.61it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:09<00:00, 23.61it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:09<00:00, 23.61it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:09<00:00, 24.77it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:09<00:00, 24.77it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:09<00:00, 24.77it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:09<00:00, 24.77it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:09<00:00, 24.77it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:09<00:00, 27.11it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:09<00:00, 27.11it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:09<00:00, 27.11it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:09<00:00, 27.11it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:10<00:00, 27.11it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:10<00:00, 29.13it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:10<00:00, 29.13it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:10<00:00, 29.13it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:10<00:00, 29.13it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:10<00:00, 29.13it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:10<00:00, 29.13it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:10<00:00, 33.82it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:10<00:00, 33.82it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:10<00:00, 33.82it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=32.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=32.59 GB):   2%|▏         | 1/58 [00:00<00:18,  3.04it/s]Capturing num tokens (num_tokens=7680 avail_mem=31.58 GB):   2%|▏         | 1/58 [00:00<00:18,  3.04it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=31.58 GB):   3%|▎         | 2/58 [00:00<00:26,  2.13it/s]Capturing num tokens (num_tokens=7168 avail_mem=32.52 GB):   3%|▎         | 2/58 [00:00<00:26,  2.13it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=32.52 GB):   5%|▌         | 3/58 [00:01<00:26,  2.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=31.70 GB):   5%|▌         | 3/58 [00:01<00:26,  2.07it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=31.70 GB):   7%|▋         | 4/58 [00:01<00:24,  2.19it/s]Capturing num tokens (num_tokens=6144 avail_mem=31.75 GB):   7%|▋         | 4/58 [00:01<00:24,  2.19it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=31.75 GB):   9%|▊         | 5/58 [00:02<00:24,  2.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=32.52 GB):   9%|▊         | 5/58 [00:02<00:24,  2.19it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=32.52 GB):  10%|█         | 6/58 [00:02<00:22,  2.30it/s]Capturing num tokens (num_tokens=5120 avail_mem=31.99 GB):  10%|█         | 6/58 [00:02<00:22,  2.30it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=31.99 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.47it/s]Capturing num tokens (num_tokens=4608 avail_mem=31.86 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.47it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=31.86 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=31.92 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.66it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=31.92 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=32.52 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.85it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=32.52 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=31.97 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.02it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=31.97 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=32.02 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.33it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=32.02 GB):  21%|██        | 12/58 [00:04<00:12,  3.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=32.07 GB):  21%|██        | 12/58 [00:04<00:12,  3.64it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=32.07 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=32.51 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.87it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=32.51 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=32.51 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=32.51 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=32.12 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.49it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=32.12 GB):  28%|██▊       | 16/58 [00:05<00:08,  4.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=32.43 GB):  28%|██▊       | 16/58 [00:05<00:08,  4.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=32.43 GB):  29%|██▉       | 17/58 [00:05<00:08,  5.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=32.50 GB):  29%|██▉       | 17/58 [00:05<00:08,  5.05it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=32.50 GB):  31%|███       | 18/58 [00:05<00:07,  5.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=32.49 GB):  31%|███       | 18/58 [00:05<00:07,  5.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=32.49 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=32.18 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.97it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=32.20 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=32.20 GB):  36%|███▌      | 21/58 [00:05<00:04,  7.50it/s]Capturing num tokens (num_tokens=960 avail_mem=32.47 GB):  36%|███▌      | 21/58 [00:05<00:04,  7.50it/s] Capturing num tokens (num_tokens=960 avail_mem=32.47 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.94it/s]Capturing num tokens (num_tokens=896 avail_mem=32.47 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.94it/s]

    Capturing num tokens (num_tokens=832 avail_mem=32.24 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.94it/s]Capturing num tokens (num_tokens=832 avail_mem=32.24 GB):  41%|████▏     | 24/58 [00:06<00:03,  9.63it/s]Capturing num tokens (num_tokens=768 avail_mem=32.45 GB):  41%|████▏     | 24/58 [00:06<00:03,  9.63it/s]Capturing num tokens (num_tokens=704 avail_mem=32.45 GB):  41%|████▏     | 24/58 [00:06<00:03,  9.63it/s]

    Capturing num tokens (num_tokens=704 avail_mem=32.45 GB):  45%|████▍     | 26/58 [00:06<00:02, 10.74it/s]Capturing num tokens (num_tokens=640 avail_mem=32.28 GB):  45%|████▍     | 26/58 [00:06<00:02, 10.74it/s]Capturing num tokens (num_tokens=576 avail_mem=32.30 GB):  45%|████▍     | 26/58 [00:06<00:02, 10.74it/s]Capturing num tokens (num_tokens=576 avail_mem=32.30 GB):  48%|████▊     | 28/58 [00:06<00:02, 12.21it/s]Capturing num tokens (num_tokens=512 avail_mem=32.43 GB):  48%|████▊     | 28/58 [00:06<00:02, 12.21it/s]Capturing num tokens (num_tokens=480 avail_mem=32.41 GB):  48%|████▊     | 28/58 [00:06<00:02, 12.21it/s]

    Capturing num tokens (num_tokens=480 avail_mem=32.41 GB):  52%|█████▏    | 30/58 [00:06<00:02, 13.55it/s]Capturing num tokens (num_tokens=448 avail_mem=32.41 GB):  52%|█████▏    | 30/58 [00:06<00:02, 13.55it/s]Capturing num tokens (num_tokens=416 avail_mem=32.40 GB):  52%|█████▏    | 30/58 [00:06<00:02, 13.55it/s]Capturing num tokens (num_tokens=416 avail_mem=32.40 GB):  55%|█████▌    | 32/58 [00:06<00:01, 14.41it/s]Capturing num tokens (num_tokens=384 avail_mem=32.39 GB):  55%|█████▌    | 32/58 [00:06<00:01, 14.41it/s]Capturing num tokens (num_tokens=352 avail_mem=32.30 GB):  55%|█████▌    | 32/58 [00:06<00:01, 14.41it/s]

    Capturing num tokens (num_tokens=320 avail_mem=32.38 GB):  55%|█████▌    | 32/58 [00:06<00:01, 14.41it/s]Capturing num tokens (num_tokens=320 avail_mem=32.38 GB):  60%|██████    | 35/58 [00:06<00:01, 16.98it/s]Capturing num tokens (num_tokens=288 avail_mem=32.38 GB):  60%|██████    | 35/58 [00:06<00:01, 16.98it/s]Capturing num tokens (num_tokens=256 avail_mem=32.37 GB):  60%|██████    | 35/58 [00:06<00:01, 16.98it/s]Capturing num tokens (num_tokens=240 avail_mem=32.29 GB):  60%|██████    | 35/58 [00:06<00:01, 16.98it/s]Capturing num tokens (num_tokens=240 avail_mem=32.29 GB):  66%|██████▌   | 38/58 [00:06<00:01, 18.62it/s]Capturing num tokens (num_tokens=224 avail_mem=32.12 GB):  66%|██████▌   | 38/58 [00:06<00:01, 18.62it/s]

    Capturing num tokens (num_tokens=208 avail_mem=31.96 GB):  66%|██████▌   | 38/58 [00:06<00:01, 18.62it/s]Capturing num tokens (num_tokens=192 avail_mem=31.75 GB):  66%|██████▌   | 38/58 [00:06<00:01, 18.62it/s]Capturing num tokens (num_tokens=192 avail_mem=31.75 GB):  71%|███████   | 41/58 [00:06<00:00, 20.14it/s]Capturing num tokens (num_tokens=176 avail_mem=31.59 GB):  71%|███████   | 41/58 [00:06<00:00, 20.14it/s]Capturing num tokens (num_tokens=160 avail_mem=31.45 GB):  71%|███████   | 41/58 [00:06<00:00, 20.14it/s]Capturing num tokens (num_tokens=144 avail_mem=31.27 GB):  71%|███████   | 41/58 [00:06<00:00, 20.14it/s]

    Capturing num tokens (num_tokens=144 avail_mem=31.27 GB):  76%|███████▌  | 44/58 [00:07<00:00, 20.86it/s]Capturing num tokens (num_tokens=128 avail_mem=31.11 GB):  76%|███████▌  | 44/58 [00:07<00:00, 20.86it/s]Capturing num tokens (num_tokens=112 avail_mem=31.00 GB):  76%|███████▌  | 44/58 [00:07<00:00, 20.86it/s]Capturing num tokens (num_tokens=96 avail_mem=30.98 GB):  76%|███████▌  | 44/58 [00:07<00:00, 20.86it/s] Capturing num tokens (num_tokens=96 avail_mem=30.98 GB):  81%|████████  | 47/58 [00:07<00:00, 22.00it/s]Capturing num tokens (num_tokens=80 avail_mem=30.97 GB):  81%|████████  | 47/58 [00:07<00:00, 22.00it/s]Capturing num tokens (num_tokens=64 avail_mem=30.96 GB):  81%|████████  | 47/58 [00:07<00:00, 22.00it/s]Capturing num tokens (num_tokens=48 avail_mem=30.93 GB):  81%|████████  | 47/58 [00:07<00:00, 22.00it/s]

    Capturing num tokens (num_tokens=48 avail_mem=30.93 GB):  86%|████████▌ | 50/58 [00:07<00:00, 23.36it/s]Capturing num tokens (num_tokens=32 avail_mem=30.83 GB):  86%|████████▌ | 50/58 [00:07<00:00, 23.36it/s]Capturing num tokens (num_tokens=28 avail_mem=30.83 GB):  86%|████████▌ | 50/58 [00:07<00:00, 23.36it/s]Capturing num tokens (num_tokens=24 avail_mem=30.78 GB):  86%|████████▌ | 50/58 [00:07<00:00, 23.36it/s]Capturing num tokens (num_tokens=20 avail_mem=30.77 GB):  86%|████████▌ | 50/58 [00:07<00:00, 23.36it/s]Capturing num tokens (num_tokens=20 avail_mem=30.77 GB):  93%|█████████▎| 54/58 [00:07<00:00, 25.39it/s]Capturing num tokens (num_tokens=16 avail_mem=30.76 GB):  93%|█████████▎| 54/58 [00:07<00:00, 25.39it/s]Capturing num tokens (num_tokens=12 avail_mem=30.23 GB):  93%|█████████▎| 54/58 [00:07<00:00, 25.39it/s]

    Capturing num tokens (num_tokens=8 avail_mem=29.76 GB):  93%|█████████▎| 54/58 [00:07<00:00, 25.39it/s] Capturing num tokens (num_tokens=8 avail_mem=29.76 GB):  98%|█████████▊| 57/58 [00:07<00:00, 25.58it/s]Capturing num tokens (num_tokens=4 avail_mem=28.45 GB):  98%|█████████▊| 57/58 [00:07<00:00, 25.58it/s]Capturing num tokens (num_tokens=4 avail_mem=28.45 GB): 100%|██████████| 58/58 [00:07<00:00,  7.67it/s]


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Finally, I conclude that the answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I identify the numbers to be added, which are 1 and 3.<br><br>Next, I perform the addition of these two numbers.<br><br>Finally, I arrive at the sum, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \text{ and } 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Finally, I conclude that the result of 1 plus 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **State the final answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>


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



<strong style='color: #00008B;'>To solve the problem 1 + 3, I start by identifying the two numbers involved.<br><br>Next, I add these numbers together.<br><br>Finally, I arrive at the sum, which is 4.<br></think><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**  <br>1. Start with the number **1**.<br>2. Add the number **3** to it.<br>3. Calculate the sum:  <br>   \(1 + 3 = 4\).<br><br>**Final Answer:**  <br>\(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I'll add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I'll add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.42s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.36s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.37s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:49,  5.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:49,  5.08s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:04,  2.22s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:04,  2.22s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:11,  1.29s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:11,  1.29s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:31,  1.66it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:23,  2.21it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:23,  2.21it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.87it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.87it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.62it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.62it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.48it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.48it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.48it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.15it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.15it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.15it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:05,  8.37it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:05,  8.37it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:05,  8.17it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:05,  8.17it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:05,  8.15it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:05,  8.15it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:05,  8.15it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:07<00:03, 10.21it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:07<00:03, 10.21it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:07<00:03, 10.21it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:07<00:03, 10.21it/s] 

    Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:07<00:03, 10.21it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:02, 16.05it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:02, 16.05it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:02, 16.05it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:07<00:02, 16.05it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:07<00:02, 16.05it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:07<00:02, 16.05it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:07<00:01, 23.58it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:07<00:01, 23.58it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:07<00:01, 23.58it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:07<00:01, 23.58it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:07<00:01, 23.58it/s]

    Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:07<00:01, 23.58it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:07<00:01, 23.58it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:07<00:00, 32.56it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:07<00:00, 32.56it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:07<00:00, 32.56it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:07<00:00, 32.56it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:07<00:00, 32.56it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:07<00:00, 32.56it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:07<00:00, 32.56it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:07<00:00, 32.56it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:07<00:00, 32.56it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:07<00:00, 43.50it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:07<00:00, 43.50it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:07<00:00, 43.50it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:07<00:00, 43.50it/s]

    Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:07<00:00, 43.50it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:07<00:00, 43.50it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:07<00:00, 43.50it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:07<00:00, 47.73it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:07<00:00, 47.73it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:07<00:00, 47.73it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:07<00:00, 47.73it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:07<00:00, 47.73it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:07<00:00, 47.73it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:07<00:00, 47.73it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:07<00:00, 47.73it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:07<00:00, 47.73it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:07<00:00, 47.73it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:07<00:00, 47.73it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00, 61.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=28.49 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=28.49 GB):   2%|▏         | 1/58 [00:00<00:16,  3.41it/s]Capturing num tokens (num_tokens=7680 avail_mem=28.46 GB):   2%|▏         | 1/58 [00:00<00:16,  3.41it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=28.46 GB):   3%|▎         | 2/58 [00:00<00:15,  3.59it/s]Capturing num tokens (num_tokens=7168 avail_mem=28.46 GB):   3%|▎         | 2/58 [00:00<00:15,  3.59it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=28.46 GB):   5%|▌         | 3/58 [00:00<00:14,  3.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.67 GB):   5%|▌         | 3/58 [00:00<00:14,  3.75it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.67 GB):   7%|▋         | 4/58 [00:01<00:14,  3.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.67 GB):   7%|▋         | 4/58 [00:01<00:14,  3.83it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.67 GB):   9%|▊         | 5/58 [00:01<00:17,  3.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.67 GB):   9%|▊         | 5/58 [00:01<00:17,  3.02it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.67 GB):  10%|█         | 6/58 [00:01<00:18,  2.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.67 GB):  10%|█         | 6/58 [00:01<00:18,  2.82it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.67 GB):  12%|█▏        | 7/58 [00:02<00:18,  2.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.67 GB):  12%|█▏        | 7/58 [00:02<00:18,  2.75it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=25.67 GB):  14%|█▍        | 8/58 [00:02<00:17,  2.84it/s]Capturing num tokens (num_tokens=4096 avail_mem=25.67 GB):  14%|█▍        | 8/58 [00:02<00:17,  2.84it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=25.67 GB):  16%|█▌        | 9/58 [00:02<00:16,  2.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.67 GB):  16%|█▌        | 9/58 [00:02<00:16,  2.96it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=25.67 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.07it/s]Capturing num tokens (num_tokens=3584 avail_mem=25.67 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.07it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=25.67 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=25.67 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.19it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=25.67 GB):  21%|██        | 12/58 [00:03<00:13,  3.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=25.66 GB):  21%|██        | 12/58 [00:03<00:13,  3.38it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=25.66 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.55it/s]Capturing num tokens (num_tokens=2816 avail_mem=25.66 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.55it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=25.66 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=25.66 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=25.66 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=25.66 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.19it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=25.66 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.72 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.34it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=24.72 GB):  29%|██▉       | 17/58 [00:04<00:10,  4.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=24.72 GB):  29%|██▉       | 17/58 [00:04<00:10,  4.07it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=24.72 GB):  31%|███       | 18/58 [00:05<00:09,  4.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=24.71 GB):  31%|███       | 18/58 [00:05<00:09,  4.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=24.71 GB):  33%|███▎      | 19/58 [00:05<00:07,  4.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=24.71 GB):  33%|███▎      | 19/58 [00:05<00:07,  4.88it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=24.71 GB):  34%|███▍      | 20/58 [00:05<00:06,  5.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.70 GB):  34%|███▍      | 20/58 [00:05<00:06,  5.74it/s]Capturing num tokens (num_tokens=960 avail_mem=24.70 GB):  34%|███▍      | 20/58 [00:05<00:06,  5.74it/s] Capturing num tokens (num_tokens=960 avail_mem=24.70 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.88it/s]Capturing num tokens (num_tokens=896 avail_mem=24.69 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.88it/s]

    Capturing num tokens (num_tokens=832 avail_mem=24.69 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.88it/s]Capturing num tokens (num_tokens=832 avail_mem=24.69 GB):  41%|████▏     | 24/58 [00:05<00:03,  9.65it/s]Capturing num tokens (num_tokens=768 avail_mem=24.68 GB):  41%|████▏     | 24/58 [00:05<00:03,  9.65it/s]Capturing num tokens (num_tokens=704 avail_mem=24.68 GB):  41%|████▏     | 24/58 [00:05<00:03,  9.65it/s]Capturing num tokens (num_tokens=704 avail_mem=24.68 GB):  45%|████▍     | 26/58 [00:05<00:02, 11.57it/s]Capturing num tokens (num_tokens=640 avail_mem=24.67 GB):  45%|████▍     | 26/58 [00:05<00:02, 11.57it/s]

    Capturing num tokens (num_tokens=576 avail_mem=24.67 GB):  45%|████▍     | 26/58 [00:05<00:02, 11.57it/s]Capturing num tokens (num_tokens=576 avail_mem=24.67 GB):  48%|████▊     | 28/58 [00:05<00:02, 13.21it/s]Capturing num tokens (num_tokens=512 avail_mem=24.66 GB):  48%|████▊     | 28/58 [00:05<00:02, 13.21it/s]Capturing num tokens (num_tokens=480 avail_mem=24.66 GB):  48%|████▊     | 28/58 [00:05<00:02, 13.21it/s]Capturing num tokens (num_tokens=448 avail_mem=24.66 GB):  48%|████▊     | 28/58 [00:05<00:02, 13.21it/s]Capturing num tokens (num_tokens=448 avail_mem=24.66 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.00it/s]Capturing num tokens (num_tokens=416 avail_mem=24.66 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.00it/s]

    Capturing num tokens (num_tokens=384 avail_mem=24.65 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.00it/s]Capturing num tokens (num_tokens=352 avail_mem=24.65 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.00it/s]Capturing num tokens (num_tokens=320 avail_mem=24.64 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.00it/s]Capturing num tokens (num_tokens=320 avail_mem=24.64 GB):  60%|██████    | 35/58 [00:06<00:01, 21.20it/s]Capturing num tokens (num_tokens=288 avail_mem=24.65 GB):  60%|██████    | 35/58 [00:06<00:01, 21.20it/s]Capturing num tokens (num_tokens=256 avail_mem=24.64 GB):  60%|██████    | 35/58 [00:06<00:01, 21.20it/s]Capturing num tokens (num_tokens=240 avail_mem=24.64 GB):  60%|██████    | 35/58 [00:06<00:01, 21.20it/s]Capturing num tokens (num_tokens=224 avail_mem=24.64 GB):  60%|██████    | 35/58 [00:06<00:01, 21.20it/s]Capturing num tokens (num_tokens=224 avail_mem=24.64 GB):  67%|██████▋   | 39/58 [00:06<00:00, 25.63it/s]Capturing num tokens (num_tokens=208 avail_mem=24.63 GB):  67%|██████▋   | 39/58 [00:06<00:00, 25.63it/s]

    Capturing num tokens (num_tokens=192 avail_mem=24.63 GB):  67%|██████▋   | 39/58 [00:06<00:00, 25.63it/s]Capturing num tokens (num_tokens=176 avail_mem=24.63 GB):  67%|██████▋   | 39/58 [00:06<00:00, 25.63it/s]Capturing num tokens (num_tokens=160 avail_mem=24.62 GB):  67%|██████▋   | 39/58 [00:06<00:00, 25.63it/s]Capturing num tokens (num_tokens=160 avail_mem=24.62 GB):  74%|███████▍  | 43/58 [00:06<00:00, 29.24it/s]Capturing num tokens (num_tokens=144 avail_mem=24.62 GB):  74%|███████▍  | 43/58 [00:06<00:00, 29.24it/s]Capturing num tokens (num_tokens=128 avail_mem=23.48 GB):  74%|███████▍  | 43/58 [00:06<00:00, 29.24it/s]

    Capturing num tokens (num_tokens=112 avail_mem=23.48 GB):  74%|███████▍  | 43/58 [00:06<00:00, 29.24it/s]Capturing num tokens (num_tokens=96 avail_mem=23.48 GB):  74%|███████▍  | 43/58 [00:06<00:00, 29.24it/s] Capturing num tokens (num_tokens=96 avail_mem=23.48 GB):  81%|████████  | 47/58 [00:06<00:00, 19.49it/s]Capturing num tokens (num_tokens=80 avail_mem=24.57 GB):  81%|████████  | 47/58 [00:06<00:00, 19.49it/s]

    Capturing num tokens (num_tokens=64 avail_mem=23.58 GB):  81%|████████  | 47/58 [00:06<00:00, 19.49it/s]Capturing num tokens (num_tokens=48 avail_mem=23.58 GB):  81%|████████  | 47/58 [00:06<00:00, 19.49it/s]

    Capturing num tokens (num_tokens=48 avail_mem=23.58 GB):  86%|████████▌ | 50/58 [00:06<00:00, 16.27it/s]Capturing num tokens (num_tokens=32 avail_mem=23.58 GB):  86%|████████▌ | 50/58 [00:06<00:00, 16.27it/s]Capturing num tokens (num_tokens=28 avail_mem=24.56 GB):  86%|████████▌ | 50/58 [00:07<00:00, 16.27it/s]

    Capturing num tokens (num_tokens=24 avail_mem=23.64 GB):  86%|████████▌ | 50/58 [00:07<00:00, 16.27it/s]Capturing num tokens (num_tokens=24 avail_mem=23.64 GB):  91%|█████████▏| 53/58 [00:07<00:00, 11.62it/s]Capturing num tokens (num_tokens=20 avail_mem=23.63 GB):  91%|█████████▏| 53/58 [00:07<00:00, 11.62it/s]

    Capturing num tokens (num_tokens=16 avail_mem=24.55 GB):  91%|█████████▏| 53/58 [00:07<00:00, 11.62it/s]Capturing num tokens (num_tokens=16 avail_mem=24.55 GB):  95%|█████████▍| 55/58 [00:07<00:00,  9.68it/s]Capturing num tokens (num_tokens=12 avail_mem=23.69 GB):  95%|█████████▍| 55/58 [00:07<00:00,  9.68it/s]

    Capturing num tokens (num_tokens=8 avail_mem=23.69 GB):  95%|█████████▍| 55/58 [00:07<00:00,  9.68it/s] Capturing num tokens (num_tokens=8 avail_mem=23.69 GB):  98%|█████████▊| 57/58 [00:08<00:00,  8.38it/s]Capturing num tokens (num_tokens=4 avail_mem=24.54 GB):  98%|█████████▊| 57/58 [00:08<00:00,  8.38it/s]

    Capturing num tokens (num_tokens=4 avail_mem=24.54 GB): 100%|██████████| 58/58 [00:08<00:00,  7.04it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the numbers involved.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum to find the result.<br></think><br><br>**Solution:**<br><br>To find the sum of 1 and 3, follow these simple steps:<br><br>1. **Identify the numbers to add:**<br>   <br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br><br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**<br><br>   \[<br>   \boxed{4}<br>   \]</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the numbers involved.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum to find the result.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>To find the sum of 1 and 3, follow these simple steps:<br><br>1. **Identify the numbers to add:**<br>   <br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br><br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**<br><br>   \[<br>   \boxed{4}<br>   \]</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
