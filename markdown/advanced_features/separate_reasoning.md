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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.10s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.04s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.05s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:57,  5.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:57,  5.22s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:15,  2.42s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:15,  2.42s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:22,  1.51s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:22,  1.51s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:57,  1.07s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:57,  1.07s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:42,  1.24it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:42,  1.24it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:33,  1.56it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:33,  1.56it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:26,  1.92it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:26,  1.92it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:21,  2.34it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:21,  2.34it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:17,  2.74it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:17,  2.74it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:15,  3.01it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:15,  3.01it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:14,  3.29it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:14,  3.29it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:12,  3.64it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:12,  3.64it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:10,  4.48it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:10,  4.48it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:08,  5.32it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:08,  5.32it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:08<00:08,  5.32it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:05,  7.58it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:05,  7.58it/s]

    Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:09<00:05,  7.58it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:04,  9.32it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:04,  9.32it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:09<00:04,  9.32it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:03, 11.08it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:03, 11.08it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:03, 11.08it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:09<00:03, 11.08it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:09<00:02, 15.19it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:09<00:02, 15.19it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:09<00:02, 15.19it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:09<00:02, 15.19it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:09<00:02, 15.19it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:09<00:02, 15.19it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:09<00:01, 23.01it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:09<00:01, 23.01it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:09<00:01, 23.01it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:09<00:01, 23.01it/s]

    Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:09<00:01, 23.01it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:09<00:01, 23.01it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:09<00:01, 23.01it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:09<00:00, 31.58it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:09<00:00, 31.58it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:09<00:00, 31.58it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:09<00:00, 31.58it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:09<00:00, 31.58it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:09<00:00, 31.58it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:09<00:00, 31.58it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:09<00:00, 31.58it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:09<00:00, 41.54it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:09<00:00, 41.54it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:09<00:00, 41.54it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:09<00:00, 41.54it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:09<00:00, 41.54it/s]

    Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:09<00:00, 41.54it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:09<00:00, 41.54it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:09<00:00, 41.54it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:09<00:00, 41.54it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:09<00:00, 50.08it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:09<00:00, 50.08it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:09<00:00, 50.08it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:09<00:00, 50.08it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:09<00:00, 50.08it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:09<00:00, 50.08it/s]

    Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:09<00:00, 50.08it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:09<00:00, 42.65it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:09<00:00, 42.65it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:09<00:00, 42.65it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:09<00:00, 42.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  5.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.23 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=42.23 GB):   2%|▏         | 1/58 [00:00<00:34,  1.64it/s]Capturing num tokens (num_tokens=7680 avail_mem=41.42 GB):   2%|▏         | 1/58 [00:00<00:34,  1.64it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=41.42 GB):   3%|▎         | 2/58 [00:01<00:32,  1.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=41.48 GB):   3%|▎         | 2/58 [00:01<00:32,  1.74it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=41.48 GB):   5%|▌         | 3/58 [00:01<00:29,  1.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=41.53 GB):   5%|▌         | 3/58 [00:01<00:29,  1.86it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=41.53 GB):   7%|▋         | 4/58 [00:02<00:26,  2.00it/s]Capturing num tokens (num_tokens=6144 avail_mem=41.58 GB):   7%|▋         | 4/58 [00:02<00:26,  2.00it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=41.58 GB):   9%|▊         | 5/58 [00:02<00:24,  2.13it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.64 GB):   9%|▊         | 5/58 [00:02<00:24,  2.13it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=41.64 GB):  10%|█         | 6/58 [00:02<00:22,  2.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.69 GB):  10%|█         | 6/58 [00:02<00:22,  2.35it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=41.69 GB):  12%|█▏        | 7/58 [00:03<00:19,  2.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.75 GB):  12%|█▏        | 7/58 [00:03<00:19,  2.56it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=41.75 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.77 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.83it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=41.77 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.80 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.09it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=41.80 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.35it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.82 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.35it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=41.82 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.85 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.62it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=41.85 GB):  21%|██        | 12/58 [00:04<00:11,  3.94it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.87 GB):  21%|██        | 12/58 [00:04<00:11,  3.94it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.87 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.27it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.89 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.27it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=41.89 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.91 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.91 GB):  26%|██▌       | 15/58 [00:04<00:08,  5.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.94 GB):  26%|██▌       | 15/58 [00:04<00:08,  5.18it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=41.94 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.96 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.96 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.98 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.34it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=41.98 GB):  31%|███       | 18/58 [00:05<00:05,  7.06it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.14 GB):  31%|███       | 18/58 [00:05<00:05,  7.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.13 GB):  31%|███       | 18/58 [00:05<00:05,  7.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.13 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.11 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.50it/s]

    Capturing num tokens (num_tokens=960 avail_mem=42.10 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.50it/s] Capturing num tokens (num_tokens=960 avail_mem=42.10 GB):  38%|███▊      | 22/58 [00:05<00:03,  9.97it/s]Capturing num tokens (num_tokens=896 avail_mem=42.09 GB):  38%|███▊      | 22/58 [00:05<00:03,  9.97it/s]Capturing num tokens (num_tokens=832 avail_mem=42.09 GB):  38%|███▊      | 22/58 [00:05<00:03,  9.97it/s]

    Capturing num tokens (num_tokens=832 avail_mem=42.09 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.44it/s]Capturing num tokens (num_tokens=768 avail_mem=42.08 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.44it/s]Capturing num tokens (num_tokens=704 avail_mem=42.07 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.44it/s]Capturing num tokens (num_tokens=704 avail_mem=42.07 GB):  45%|████▍     | 26/58 [00:05<00:02, 12.98it/s]Capturing num tokens (num_tokens=640 avail_mem=42.07 GB):  45%|████▍     | 26/58 [00:05<00:02, 12.98it/s]Capturing num tokens (num_tokens=576 avail_mem=42.06 GB):  45%|████▍     | 26/58 [00:05<00:02, 12.98it/s]

    Capturing num tokens (num_tokens=576 avail_mem=42.06 GB):  48%|████▊     | 28/58 [00:05<00:02, 14.53it/s]Capturing num tokens (num_tokens=512 avail_mem=42.01 GB):  48%|████▊     | 28/58 [00:05<00:02, 14.53it/s]Capturing num tokens (num_tokens=480 avail_mem=42.04 GB):  48%|████▊     | 28/58 [00:05<00:02, 14.53it/s]Capturing num tokens (num_tokens=448 avail_mem=42.04 GB):  48%|████▊     | 28/58 [00:05<00:02, 14.53it/s]Capturing num tokens (num_tokens=448 avail_mem=42.04 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.81it/s]Capturing num tokens (num_tokens=416 avail_mem=42.03 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.81it/s]Capturing num tokens (num_tokens=384 avail_mem=42.03 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.81it/s]

    Capturing num tokens (num_tokens=352 avail_mem=42.02 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.81it/s]Capturing num tokens (num_tokens=352 avail_mem=42.02 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.61it/s]Capturing num tokens (num_tokens=320 avail_mem=42.01 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.61it/s]Capturing num tokens (num_tokens=288 avail_mem=41.98 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.61it/s]Capturing num tokens (num_tokens=256 avail_mem=42.00 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.61it/s]Capturing num tokens (num_tokens=256 avail_mem=42.00 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.79it/s]Capturing num tokens (num_tokens=240 avail_mem=42.00 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.79it/s]Capturing num tokens (num_tokens=224 avail_mem=41.99 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.79it/s]

    Capturing num tokens (num_tokens=208 avail_mem=41.98 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.79it/s]Capturing num tokens (num_tokens=208 avail_mem=41.98 GB):  69%|██████▉   | 40/58 [00:06<00:00, 22.22it/s]Capturing num tokens (num_tokens=192 avail_mem=41.98 GB):  69%|██████▉   | 40/58 [00:06<00:00, 22.22it/s]Capturing num tokens (num_tokens=176 avail_mem=41.97 GB):  69%|██████▉   | 40/58 [00:06<00:00, 22.22it/s]Capturing num tokens (num_tokens=160 avail_mem=41.96 GB):  69%|██████▉   | 40/58 [00:06<00:00, 22.22it/s]Capturing num tokens (num_tokens=160 avail_mem=41.96 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.44it/s]Capturing num tokens (num_tokens=144 avail_mem=41.93 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.44it/s]Capturing num tokens (num_tokens=128 avail_mem=41.93 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.44it/s]

    Capturing num tokens (num_tokens=112 avail_mem=41.94 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.44it/s]Capturing num tokens (num_tokens=112 avail_mem=41.94 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.89it/s]Capturing num tokens (num_tokens=96 avail_mem=41.94 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.89it/s] Capturing num tokens (num_tokens=80 avail_mem=41.93 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.89it/s]Capturing num tokens (num_tokens=64 avail_mem=41.93 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.89it/s]Capturing num tokens (num_tokens=64 avail_mem=41.93 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.62it/s]Capturing num tokens (num_tokens=48 avail_mem=41.92 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.62it/s]Capturing num tokens (num_tokens=32 avail_mem=41.91 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.62it/s]

    Capturing num tokens (num_tokens=28 avail_mem=41.91 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.62it/s]Capturing num tokens (num_tokens=28 avail_mem=41.91 GB):  90%|████████▉ | 52/58 [00:06<00:00, 26.37it/s]Capturing num tokens (num_tokens=24 avail_mem=41.90 GB):  90%|████████▉ | 52/58 [00:06<00:00, 26.37it/s]Capturing num tokens (num_tokens=20 avail_mem=41.89 GB):  90%|████████▉ | 52/58 [00:06<00:00, 26.37it/s]Capturing num tokens (num_tokens=16 avail_mem=41.86 GB):  90%|████████▉ | 52/58 [00:06<00:00, 26.37it/s]Capturing num tokens (num_tokens=16 avail_mem=41.86 GB):  95%|█████████▍| 55/58 [00:06<00:00, 27.28it/s]Capturing num tokens (num_tokens=12 avail_mem=41.88 GB):  95%|█████████▍| 55/58 [00:06<00:00, 27.28it/s]Capturing num tokens (num_tokens=8 avail_mem=41.87 GB):  95%|█████████▍| 55/58 [00:06<00:00, 27.28it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=41.86 GB):  95%|█████████▍| 55/58 [00:07<00:00, 27.28it/s]Capturing num tokens (num_tokens=4 avail_mem=41.86 GB): 100%|██████████| 58/58 [00:07<00:00, 26.00it/s]Capturing num tokens (num_tokens=4 avail_mem=41.86 GB): 100%|██████████| 58/58 [00:07<00:00,  8.21it/s]


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



<strong style='color: #00008B;'>First, I need to identify the two numbers in the addition problem, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Finally, I'll calculate the sum to find the result.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>I see that the problem is to add 1 and 3.<br><br>First, I'll identify the two numbers: 1 and 3.<br><br>Next, I'll add them together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I recognize that I need to add the numbers 1 and 3 together.<br><br>Adding 1 and 3 gives me a sum of 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Certainly! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Present the final answer:**<br>   \[<br>   \boxed{4}<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I identify the two numbers that need to be added: 1 and 3.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum to determine the result.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum, which is 4.<br></think><br><br>To solve the addition problem \(1 + 3\), follow these simple steps:<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Present the final answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>To solve the addition problem \(1 + 3\), follow these simple steps:<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Present the final answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.35s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.23s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.25s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:04,  2.22s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:04,  2.22s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:11,  1.29s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:11,  1.29s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:31,  1.66it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:23,  2.22it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:23,  2.22it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.89it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.89it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.63it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.63it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.50it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.50it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.50it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.18it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.18it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.18it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.74it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.74it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.74it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.24it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.24it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.24it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.25it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.25it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.25it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.25it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.83it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.83it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.83it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.83it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.83it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.83it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]

    Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 33.67it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 33.67it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 33.67it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 33.67it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 33.67it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 33.67it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 33.67it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 33.67it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:07<00:00, 41.84it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:07<00:00, 41.84it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:07<00:00, 41.84it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:07<00:00, 41.84it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:07<00:00, 41.84it/s]

    Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:07<00:00, 41.84it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:07<00:00, 41.84it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:07<00:00, 41.84it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:07<00:00, 41.84it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:07<00:00, 50.33it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:07<00:00, 50.33it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 50.33it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 50.33it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:07<00:00, 50.33it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:07<00:00, 50.33it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:07<00:00, 50.33it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:07<00:00, 50.33it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:07<00:00, 50.33it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.20 GB):   2%|▏         | 1/58 [00:00<00:17,  3.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.88 GB):   2%|▏         | 1/58 [00:00<00:17,  3.34it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.88 GB):   3%|▎         | 2/58 [00:00<00:25,  2.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.23 GB):   3%|▎         | 2/58 [00:00<00:25,  2.22it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.23 GB):   5%|▌         | 3/58 [00:01<00:19,  2.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.23 GB):   5%|▌         | 3/58 [00:01<00:19,  2.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.23 GB):   7%|▋         | 4/58 [00:01<00:16,  3.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.23 GB):   7%|▋         | 4/58 [00:01<00:16,  3.29it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.23 GB):   9%|▊         | 5/58 [00:01<00:14,  3.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.23 GB):   9%|▊         | 5/58 [00:01<00:14,  3.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.23 GB):  10%|█         | 6/58 [00:01<00:12,  4.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.22 GB):  10%|█         | 6/58 [00:01<00:12,  4.21it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.22 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.23 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.23 GB):  14%|█▍        | 8/58 [00:02<00:09,  5.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.23 GB):  14%|█▍        | 8/58 [00:02<00:09,  5.22it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.23 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.23 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.23 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.22 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.27it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.22 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.22 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.22 GB):  21%|██        | 12/58 [00:02<00:06,  7.49it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.22 GB):  21%|██        | 12/58 [00:02<00:06,  7.49it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=42.22 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.22 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.22 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.22 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.21 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.48it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=42.21 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.21 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.21 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.37it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=42.21 GB):  31%|███       | 18/58 [00:03<00:04,  8.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.20 GB):  31%|███       | 18/58 [00:03<00:04,  8.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.20 GB):  33%|███▎      | 19/58 [00:03<00:05,  7.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.20 GB):  33%|███▎      | 19/58 [00:03<00:05,  7.55it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=42.20 GB):  34%|███▍      | 20/58 [00:03<00:05,  7.27it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.19 GB):  34%|███▍      | 20/58 [00:03<00:05,  7.27it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.19 GB):  36%|███▌      | 21/58 [00:03<00:04,  7.56it/s]Capturing num tokens (num_tokens=960 avail_mem=42.19 GB):  36%|███▌      | 21/58 [00:03<00:04,  7.56it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=42.19 GB):  38%|███▊      | 22/58 [00:03<00:04,  7.49it/s]Capturing num tokens (num_tokens=896 avail_mem=42.18 GB):  38%|███▊      | 22/58 [00:03<00:04,  7.49it/s]Capturing num tokens (num_tokens=896 avail_mem=42.18 GB):  40%|███▉      | 23/58 [00:03<00:04,  7.53it/s]Capturing num tokens (num_tokens=832 avail_mem=42.18 GB):  40%|███▉      | 23/58 [00:03<00:04,  7.53it/s]

    Capturing num tokens (num_tokens=832 avail_mem=42.18 GB):  41%|████▏     | 24/58 [00:03<00:04,  7.76it/s]Capturing num tokens (num_tokens=768 avail_mem=42.18 GB):  41%|████▏     | 24/58 [00:03<00:04,  7.76it/s]Capturing num tokens (num_tokens=768 avail_mem=42.18 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.80it/s]Capturing num tokens (num_tokens=704 avail_mem=42.17 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.80it/s]

    Capturing num tokens (num_tokens=704 avail_mem=42.17 GB):  45%|████▍     | 26/58 [00:04<00:03,  8.06it/s]Capturing num tokens (num_tokens=640 avail_mem=42.17 GB):  45%|████▍     | 26/58 [00:04<00:03,  8.06it/s]Capturing num tokens (num_tokens=640 avail_mem=42.17 GB):  47%|████▋     | 27/58 [00:04<00:03,  8.10it/s]Capturing num tokens (num_tokens=576 avail_mem=42.17 GB):  47%|████▋     | 27/58 [00:04<00:03,  8.10it/s]

    Capturing num tokens (num_tokens=576 avail_mem=42.17 GB):  48%|████▊     | 28/58 [00:04<00:03,  8.33it/s]Capturing num tokens (num_tokens=512 avail_mem=42.16 GB):  48%|████▊     | 28/58 [00:04<00:03,  8.33it/s]Capturing num tokens (num_tokens=512 avail_mem=42.16 GB):  50%|█████     | 29/58 [00:04<00:03,  8.53it/s]Capturing num tokens (num_tokens=480 avail_mem=42.16 GB):  50%|█████     | 29/58 [00:04<00:03,  8.53it/s]

    Capturing num tokens (num_tokens=480 avail_mem=42.16 GB):  52%|█████▏    | 30/58 [00:04<00:03,  8.91it/s]Capturing num tokens (num_tokens=448 avail_mem=42.16 GB):  52%|█████▏    | 30/58 [00:04<00:03,  8.91it/s]Capturing num tokens (num_tokens=448 avail_mem=42.16 GB):  53%|█████▎    | 31/58 [00:04<00:03,  8.82it/s]Capturing num tokens (num_tokens=416 avail_mem=42.15 GB):  53%|█████▎    | 31/58 [00:04<00:03,  8.82it/s]

    Capturing num tokens (num_tokens=416 avail_mem=42.15 GB):  55%|█████▌    | 32/58 [00:04<00:03,  8.46it/s]Capturing num tokens (num_tokens=384 avail_mem=42.15 GB):  55%|█████▌    | 32/58 [00:04<00:03,  8.46it/s]Capturing num tokens (num_tokens=384 avail_mem=42.15 GB):  57%|█████▋    | 33/58 [00:05<00:02,  8.42it/s]Capturing num tokens (num_tokens=352 avail_mem=42.14 GB):  57%|█████▋    | 33/58 [00:05<00:02,  8.42it/s]

    Capturing num tokens (num_tokens=352 avail_mem=42.14 GB):  59%|█████▊    | 34/58 [00:05<00:02,  8.28it/s]Capturing num tokens (num_tokens=320 avail_mem=42.14 GB):  59%|█████▊    | 34/58 [00:05<00:02,  8.28it/s]

    Capturing num tokens (num_tokens=320 avail_mem=42.14 GB):  60%|██████    | 35/58 [00:05<00:03,  6.63it/s]Capturing num tokens (num_tokens=288 avail_mem=42.15 GB):  60%|██████    | 35/58 [00:05<00:03,  6.63it/s]

    Capturing num tokens (num_tokens=288 avail_mem=42.15 GB):  62%|██████▏   | 36/58 [00:05<00:04,  4.66it/s]Capturing num tokens (num_tokens=256 avail_mem=42.14 GB):  62%|██████▏   | 36/58 [00:05<00:04,  4.66it/s]Capturing num tokens (num_tokens=256 avail_mem=42.14 GB):  64%|██████▍   | 37/58 [00:05<00:04,  4.93it/s]Capturing num tokens (num_tokens=240 avail_mem=42.14 GB):  64%|██████▍   | 37/58 [00:05<00:04,  4.93it/s]

    Capturing num tokens (num_tokens=224 avail_mem=42.13 GB):  64%|██████▍   | 37/58 [00:06<00:04,  4.93it/s]Capturing num tokens (num_tokens=224 avail_mem=42.13 GB):  67%|██████▋   | 39/58 [00:06<00:02,  6.91it/s]Capturing num tokens (num_tokens=208 avail_mem=42.13 GB):  67%|██████▋   | 39/58 [00:06<00:02,  6.91it/s]Capturing num tokens (num_tokens=192 avail_mem=42.13 GB):  67%|██████▋   | 39/58 [00:06<00:02,  6.91it/s]Capturing num tokens (num_tokens=192 avail_mem=42.13 GB):  71%|███████   | 41/58 [00:06<00:01,  9.23it/s]Capturing num tokens (num_tokens=176 avail_mem=42.12 GB):  71%|███████   | 41/58 [00:06<00:01,  9.23it/s]

    Capturing num tokens (num_tokens=160 avail_mem=42.12 GB):  71%|███████   | 41/58 [00:06<00:01,  9.23it/s]Capturing num tokens (num_tokens=160 avail_mem=42.12 GB):  74%|███████▍  | 43/58 [00:06<00:01, 11.46it/s]Capturing num tokens (num_tokens=144 avail_mem=42.11 GB):  74%|███████▍  | 43/58 [00:06<00:01, 11.46it/s]Capturing num tokens (num_tokens=128 avail_mem=42.12 GB):  74%|███████▍  | 43/58 [00:06<00:01, 11.46it/s]Capturing num tokens (num_tokens=112 avail_mem=42.11 GB):  74%|███████▍  | 43/58 [00:06<00:01, 11.46it/s]Capturing num tokens (num_tokens=112 avail_mem=42.11 GB):  79%|███████▉  | 46/58 [00:06<00:00, 14.85it/s]Capturing num tokens (num_tokens=96 avail_mem=42.11 GB):  79%|███████▉  | 46/58 [00:06<00:00, 14.85it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=42.10 GB):  79%|███████▉  | 46/58 [00:06<00:00, 14.85it/s]Capturing num tokens (num_tokens=64 avail_mem=42.10 GB):  79%|███████▉  | 46/58 [00:06<00:00, 14.85it/s]Capturing num tokens (num_tokens=64 avail_mem=42.10 GB):  84%|████████▍ | 49/58 [00:06<00:00, 17.59it/s]Capturing num tokens (num_tokens=48 avail_mem=42.09 GB):  84%|████████▍ | 49/58 [00:06<00:00, 17.59it/s]Capturing num tokens (num_tokens=32 avail_mem=42.09 GB):  84%|████████▍ | 49/58 [00:06<00:00, 17.59it/s]Capturing num tokens (num_tokens=32 avail_mem=42.09 GB):  88%|████████▊ | 51/58 [00:06<00:00, 18.12it/s]Capturing num tokens (num_tokens=28 avail_mem=42.09 GB):  88%|████████▊ | 51/58 [00:06<00:00, 18.12it/s]

    Capturing num tokens (num_tokens=24 avail_mem=42.08 GB):  88%|████████▊ | 51/58 [00:06<00:00, 18.12it/s]Capturing num tokens (num_tokens=20 avail_mem=42.08 GB):  88%|████████▊ | 51/58 [00:06<00:00, 18.12it/s]Capturing num tokens (num_tokens=16 avail_mem=42.08 GB):  88%|████████▊ | 51/58 [00:06<00:00, 18.12it/s]Capturing num tokens (num_tokens=16 avail_mem=42.08 GB):  95%|█████████▍| 55/58 [00:06<00:00, 23.58it/s]Capturing num tokens (num_tokens=12 avail_mem=42.07 GB):  95%|█████████▍| 55/58 [00:06<00:00, 23.58it/s]Capturing num tokens (num_tokens=8 avail_mem=42.07 GB):  95%|█████████▍| 55/58 [00:06<00:00, 23.58it/s] Capturing num tokens (num_tokens=4 avail_mem=42.07 GB):  95%|█████████▍| 55/58 [00:06<00:00, 23.58it/s]Capturing num tokens (num_tokens=4 avail_mem=42.07 GB): 100%|██████████| 58/58 [00:06<00:00,  8.52it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>To find the sum of 1 and 3, I start by identifying the two numbers involved in the addition.<br><br>Next, I add the two numbers together to get the total.<br><br>Finally, I conclude that the sum of 1 and 3 is 4.<br></think><br><br>**Solution:**<br><br>We need to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the sum of \(1\) and \(3\) is \(\boxed{4}\).</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>To find the sum of 1 and 3, I start by identifying the two numbers involved in the addition.<br><br>Next, I add the two numbers together to get the total.<br><br>Finally, I conclude that the sum of 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We need to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the sum of \(1\) and \(3\) is \(\boxed{4}\).</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
