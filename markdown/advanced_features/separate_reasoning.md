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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.14s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.06s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.07s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:01,  2.17s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:01,  2.17s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:09,  1.27s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:09,  1.27s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:31,  1.67it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:23,  2.24it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:23,  2.24it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.91it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.91it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.66it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.66it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.52it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.52it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.52it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.21it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.21it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.21it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.37it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.37it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.37it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.32it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.32it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.32it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:06<00:03, 11.32it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:02, 14.62it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:02, 14.62it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:02, 14.62it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.62it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.62it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.95it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.53it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.53it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.53it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.53it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.53it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.53it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.53it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.53it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.53it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 46.83it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 46.83it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 55.65it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 55.65it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 55.65it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 55.65it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 55.65it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 55.65it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 55.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.28 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=45.28 GB):   2%|▏         | 1/58 [00:00<00:24,  2.36it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.15 GB):   2%|▏         | 1/58 [00:00<00:24,  2.36it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.15 GB):   3%|▎         | 2/58 [00:00<00:22,  2.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=44.15 GB):   3%|▎         | 2/58 [00:00<00:22,  2.51it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=44.15 GB):   5%|▌         | 3/58 [00:01<00:17,  3.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.15 GB):   5%|▌         | 3/58 [00:01<00:17,  3.07it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=44.15 GB):   7%|▋         | 4/58 [00:01<00:15,  3.56it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.15 GB):   7%|▋         | 4/58 [00:01<00:15,  3.56it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.15 GB):   9%|▊         | 5/58 [00:01<00:13,  3.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.15 GB):   9%|▊         | 5/58 [00:01<00:13,  3.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.15 GB):  10%|█         | 6/58 [00:01<00:11,  4.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.15 GB):  10%|█         | 6/58 [00:01<00:11,  4.45it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=44.15 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.15 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.15 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.15 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.45it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=44.15 GB):  16%|█▌        | 9/58 [00:02<00:08,  6.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.15 GB):  16%|█▌        | 9/58 [00:02<00:08,  6.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.15 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=44.15 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.53it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=44.15 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.14 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.11it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=44.14 GB):  21%|██        | 12/58 [00:02<00:07,  6.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=44.14 GB):  21%|██        | 12/58 [00:02<00:07,  6.23it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=44.14 GB):  22%|██▏       | 13/58 [00:02<00:08,  5.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=44.14 GB):  22%|██▏       | 13/58 [00:02<00:08,  5.39it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=44.14 GB):  24%|██▍       | 14/58 [00:02<00:08,  5.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.14 GB):  24%|██▍       | 14/58 [00:02<00:08,  5.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.14 GB):  26%|██▌       | 15/58 [00:03<00:08,  5.02it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.14 GB):  26%|██▌       | 15/58 [00:03<00:08,  5.02it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=44.14 GB):  28%|██▊       | 16/58 [00:03<00:08,  5.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.13 GB):  28%|██▊       | 16/58 [00:03<00:08,  5.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.13 GB):  29%|██▉       | 17/58 [00:03<00:07,  5.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.13 GB):  29%|██▉       | 17/58 [00:03<00:07,  5.14it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=44.13 GB):  31%|███       | 18/58 [00:03<00:07,  5.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.13 GB):  31%|███       | 18/58 [00:03<00:07,  5.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.13 GB):  33%|███▎      | 19/58 [00:03<00:06,  5.63it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.13 GB):  33%|███▎      | 19/58 [00:03<00:06,  5.63it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=44.13 GB):  34%|███▍      | 20/58 [00:04<00:06,  5.92it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.11 GB):  34%|███▍      | 20/58 [00:04<00:06,  5.92it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.11 GB):  36%|███▌      | 21/58 [00:04<00:05,  6.39it/s]Capturing num tokens (num_tokens=960 avail_mem=44.11 GB):  36%|███▌      | 21/58 [00:04<00:05,  6.39it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=44.11 GB):  38%|███▊      | 22/58 [00:04<00:05,  6.73it/s]Capturing num tokens (num_tokens=896 avail_mem=44.11 GB):  38%|███▊      | 22/58 [00:04<00:05,  6.73it/s]Capturing num tokens (num_tokens=896 avail_mem=44.11 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.03it/s]Capturing num tokens (num_tokens=832 avail_mem=44.10 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.03it/s]

    Capturing num tokens (num_tokens=832 avail_mem=44.10 GB):  41%|████▏     | 24/58 [00:04<00:04,  7.21it/s]Capturing num tokens (num_tokens=768 avail_mem=44.10 GB):  41%|████▏     | 24/58 [00:04<00:04,  7.21it/s]Capturing num tokens (num_tokens=768 avail_mem=44.10 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.62it/s]Capturing num tokens (num_tokens=704 avail_mem=44.09 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.62it/s]

    Capturing num tokens (num_tokens=704 avail_mem=44.09 GB):  45%|████▍     | 26/58 [00:04<00:04,  7.90it/s]Capturing num tokens (num_tokens=640 avail_mem=44.09 GB):  45%|████▍     | 26/58 [00:04<00:04,  7.90it/s]Capturing num tokens (num_tokens=640 avail_mem=44.09 GB):  47%|████▋     | 27/58 [00:04<00:03,  8.23it/s]Capturing num tokens (num_tokens=576 avail_mem=44.09 GB):  47%|████▋     | 27/58 [00:04<00:03,  8.23it/s]

    Capturing num tokens (num_tokens=576 avail_mem=44.09 GB):  48%|████▊     | 28/58 [00:05<00:03,  8.33it/s]Capturing num tokens (num_tokens=512 avail_mem=44.08 GB):  48%|████▊     | 28/58 [00:05<00:03,  8.33it/s]Capturing num tokens (num_tokens=480 avail_mem=44.08 GB):  48%|████▊     | 28/58 [00:05<00:03,  8.33it/s]Capturing num tokens (num_tokens=480 avail_mem=44.08 GB):  52%|█████▏    | 30/58 [00:05<00:02,  9.35it/s]Capturing num tokens (num_tokens=448 avail_mem=44.08 GB):  52%|█████▏    | 30/58 [00:05<00:02,  9.35it/s]

    Capturing num tokens (num_tokens=448 avail_mem=44.08 GB):  53%|█████▎    | 31/58 [00:05<00:02,  9.16it/s]Capturing num tokens (num_tokens=416 avail_mem=44.08 GB):  53%|█████▎    | 31/58 [00:05<00:02,  9.16it/s]Capturing num tokens (num_tokens=384 avail_mem=44.07 GB):  53%|█████▎    | 31/58 [00:05<00:02,  9.16it/s]Capturing num tokens (num_tokens=384 avail_mem=44.07 GB):  57%|█████▋    | 33/58 [00:05<00:02, 10.33it/s]Capturing num tokens (num_tokens=352 avail_mem=44.07 GB):  57%|█████▋    | 33/58 [00:05<00:02, 10.33it/s]

    Capturing num tokens (num_tokens=320 avail_mem=44.06 GB):  57%|█████▋    | 33/58 [00:05<00:02, 10.33it/s]Capturing num tokens (num_tokens=320 avail_mem=44.06 GB):  60%|██████    | 35/58 [00:05<00:01, 11.67it/s]Capturing num tokens (num_tokens=288 avail_mem=44.07 GB):  60%|██████    | 35/58 [00:05<00:01, 11.67it/s]Capturing num tokens (num_tokens=256 avail_mem=44.06 GB):  60%|██████    | 35/58 [00:05<00:01, 11.67it/s]Capturing num tokens (num_tokens=256 avail_mem=44.06 GB):  64%|██████▍   | 37/58 [00:05<00:01, 12.81it/s]Capturing num tokens (num_tokens=240 avail_mem=44.06 GB):  64%|██████▍   | 37/58 [00:05<00:01, 12.81it/s]

    Capturing num tokens (num_tokens=224 avail_mem=44.06 GB):  64%|██████▍   | 37/58 [00:05<00:01, 12.81it/s]Capturing num tokens (num_tokens=224 avail_mem=44.06 GB):  67%|██████▋   | 39/58 [00:05<00:01, 14.43it/s]Capturing num tokens (num_tokens=208 avail_mem=44.05 GB):  67%|██████▋   | 39/58 [00:05<00:01, 14.43it/s]Capturing num tokens (num_tokens=192 avail_mem=44.05 GB):  67%|██████▋   | 39/58 [00:05<00:01, 14.43it/s]Capturing num tokens (num_tokens=192 avail_mem=44.05 GB):  71%|███████   | 41/58 [00:05<00:01, 15.82it/s]Capturing num tokens (num_tokens=176 avail_mem=44.04 GB):  71%|███████   | 41/58 [00:05<00:01, 15.82it/s]Capturing num tokens (num_tokens=160 avail_mem=44.04 GB):  71%|███████   | 41/58 [00:05<00:01, 15.82it/s]

    Capturing num tokens (num_tokens=144 avail_mem=44.03 GB):  71%|███████   | 41/58 [00:06<00:01, 15.82it/s]Capturing num tokens (num_tokens=144 avail_mem=44.03 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.54it/s]Capturing num tokens (num_tokens=128 avail_mem=44.04 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.54it/s]Capturing num tokens (num_tokens=112 avail_mem=44.03 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.54it/s]Capturing num tokens (num_tokens=96 avail_mem=44.03 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.54it/s] Capturing num tokens (num_tokens=96 avail_mem=44.03 GB):  81%|████████  | 47/58 [00:06<00:00, 20.01it/s]Capturing num tokens (num_tokens=80 avail_mem=44.02 GB):  81%|████████  | 47/58 [00:06<00:00, 20.01it/s]

    Capturing num tokens (num_tokens=64 avail_mem=44.02 GB):  81%|████████  | 47/58 [00:06<00:00, 20.01it/s]Capturing num tokens (num_tokens=48 avail_mem=44.01 GB):  81%|████████  | 47/58 [00:06<00:00, 20.01it/s]Capturing num tokens (num_tokens=48 avail_mem=44.01 GB):  86%|████████▌ | 50/58 [00:06<00:00, 22.41it/s]Capturing num tokens (num_tokens=32 avail_mem=44.01 GB):  86%|████████▌ | 50/58 [00:06<00:00, 22.41it/s]Capturing num tokens (num_tokens=28 avail_mem=44.01 GB):  86%|████████▌ | 50/58 [00:06<00:00, 22.41it/s]Capturing num tokens (num_tokens=24 avail_mem=44.01 GB):  86%|████████▌ | 50/58 [00:06<00:00, 22.41it/s]Capturing num tokens (num_tokens=20 avail_mem=44.00 GB):  86%|████████▌ | 50/58 [00:06<00:00, 22.41it/s]Capturing num tokens (num_tokens=16 avail_mem=44.00 GB):  86%|████████▌ | 50/58 [00:06<00:00, 22.41it/s]Capturing num tokens (num_tokens=16 avail_mem=44.00 GB):  95%|█████████▍| 55/58 [00:06<00:00, 28.19it/s]Capturing num tokens (num_tokens=12 avail_mem=43.99 GB):  95%|█████████▍| 55/58 [00:06<00:00, 28.19it/s]Capturing num tokens (num_tokens=8 avail_mem=43.99 GB):  95%|█████████▍| 55/58 [00:06<00:00, 28.19it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=43.99 GB):  95%|█████████▍| 55/58 [00:06<00:00, 28.19it/s]Capturing num tokens (num_tokens=4 avail_mem=43.99 GB): 100%|██████████| 58/58 [00:06<00:00,  8.96it/s]


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I arrive at the conclusion that 1 plus 3 equals 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Start with the number 1.**<br>2. **Add 3 to it.**<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Final Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Finally, I conclude that the result of 1 plus 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the result is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the numbers involved.<br><br>Next, I add the two numbers together to find the sum.<br><br>Finally, I conclude that the result of 1 plus 3 is 4.<br></think><br><br>**Solution:**<br><br>To find the sum of 1 and 3, follow these steps:<br><br>1. **Start with the number 1.**<br>2. **Add the number 3 to it.**<br>3. **Calculate the total.**<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Final Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I'll identify the two numbers involved in the addition: 1 and 3.<br><br>Then, I'll perform the addition operation by adding these two numbers together.<br><br>Finally, I'll calculate the result, which is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = \boxed{4}<br>\]<br><br>**Step-by-Step Explanation:**<br><br>1. **Identify the Numbers:**<br>   - The first number is **1**.<br>   - The second number is **3**.<br><br>2. **Add the Numbers:**<br>   - Start with the first number: **1**.<br>   - Add the second number: \(1 + 3\).<br><br>3. **Calculate the Sum:**<br>   - \(1 + 3 = 4\).<br><br>**Final Answer:** \(\boxed{4}\)</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I'll identify the two numbers involved in the addition: 1 and 3.<br><br>Then, I'll perform the addition operation by adding these two numbers together.<br><br>Finally, I'll calculate the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = \boxed{4}<br>\]<br><br>**Step-by-Step Explanation:**<br><br>1. **Identify the Numbers:**<br>   - The first number is **1**.<br>   - The second number is **3**.<br><br>2. **Add the Numbers:**<br>   - Start with the first number: **1**.<br>   - Add the second number: \(1 + 3\).<br><br>3. **Calculate the Sum:**<br>   - \(1 + 3 = 4\).<br><br>**Final Answer:** \(\boxed{4}\)</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.19s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.11s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.12s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:16,  2.44s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:16,  2.44s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:23,  1.52s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:23,  1.52s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:58,  1.08s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:58,  1.08s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:43,  1.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:43,  1.23it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:33,  1.54it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:33,  1.54it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:27,  1.89it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:27,  1.89it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:22,  2.24it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:22,  2.24it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:18,  2.66it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:18,  2.66it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:15,  3.12it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:15,  3.12it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:13,  3.51it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:13,  3.51it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:11,  3.98it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:11,  3.98it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:09,  4.86it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:09,  4.86it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:08,  5.33it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:08,  5.33it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:07,  5.96it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:07,  5.96it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:06,  6.61it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:06,  6.61it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:09<00:06,  6.61it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:09<00:04,  7.95it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:09<00:04,  7.95it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:09<00:04,  7.95it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:09<00:03,  9.28it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:09<00:03,  9.28it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:09<00:03,  9.28it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:09<00:03, 11.13it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:09<00:03, 11.13it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:09<00:03, 11.13it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:09<00:02, 12.57it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:09<00:02, 12.57it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:09<00:02, 12.57it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:09<00:02, 14.14it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:09<00:02, 14.14it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:10<00:02, 14.14it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:10<00:02, 14.14it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:10<00:01, 16.83it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:10<00:01, 16.83it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:10<00:01, 16.83it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:10<00:01, 16.83it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:10<00:01, 18.28it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:10<00:01, 18.28it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:10<00:01, 18.28it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:10<00:01, 18.28it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:10<00:01, 20.15it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:10<00:01, 20.15it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:10<00:01, 20.15it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:10<00:01, 20.15it/s]

    Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:10<00:01, 20.15it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:10<00:00, 23.53it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:10<00:00, 23.53it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:10<00:00, 23.53it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:10<00:00, 23.53it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:10<00:00, 24.50it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:10<00:00, 24.50it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:10<00:00, 24.50it/s]

    Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:10<00:00, 24.50it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:10<00:00, 25.69it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:10<00:00, 25.69it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:10<00:00, 25.69it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:10<00:00, 25.69it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:10<00:00, 26.46it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:10<00:00, 26.46it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:10<00:00, 26.46it/s]

    Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:10<00:00, 26.46it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:10<00:00, 26.46it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:10<00:00, 28.40it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:10<00:00, 28.40it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:10<00:00, 28.40it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:10<00:00, 28.40it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:11<00:00, 28.40it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:11<00:00, 31.15it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:11<00:00, 31.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  5.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.32 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.32 GB):   2%|▏         | 1/58 [00:00<00:36,  1.58it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.29 GB):   2%|▏         | 1/58 [00:00<00:36,  1.58it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.29 GB):   3%|▎         | 2/58 [00:01<00:32,  1.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.29 GB):   3%|▎         | 2/58 [00:01<00:32,  1.73it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.29 GB):   5%|▌         | 3/58 [00:01<00:29,  1.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.29 GB):   5%|▌         | 3/58 [00:01<00:29,  1.86it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.29 GB):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.29 GB):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.29 GB):   9%|▊         | 5/58 [00:02<00:25,  2.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.29 GB):   9%|▊         | 5/58 [00:02<00:25,  2.11it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.29 GB):  10%|█         | 6/58 [00:02<00:22,  2.29it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.28 GB):  10%|█         | 6/58 [00:02<00:22,  2.29it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.28 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.29 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.48it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.29 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.29 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.90it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.29 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.29 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.11it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.29 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.28 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.17it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.28 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.28 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.28it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=43.28 GB):  21%|██        | 12/58 [00:04<00:13,  3.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.28 GB):  21%|██        | 12/58 [00:04<00:13,  3.40it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=43.28 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.28 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.56it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.28 GB):  24%|██▍       | 14/58 [00:05<00:11,  3.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.28 GB):  24%|██▍       | 14/58 [00:05<00:11,  3.73it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=43.28 GB):  26%|██▌       | 15/58 [00:05<00:10,  4.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.27 GB):  26%|██▌       | 15/58 [00:05<00:10,  4.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.27 GB):  28%|██▊       | 16/58 [00:05<00:09,  4.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.27 GB):  28%|██▊       | 16/58 [00:05<00:09,  4.30it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=43.27 GB):  29%|██▉       | 17/58 [00:05<00:09,  4.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.27 GB):  29%|██▉       | 17/58 [00:05<00:09,  4.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.27 GB):  31%|███       | 18/58 [00:05<00:08,  4.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.26 GB):  31%|███       | 18/58 [00:05<00:08,  4.83it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=43.26 GB):  33%|███▎      | 19/58 [00:05<00:07,  5.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.26 GB):  33%|███▎      | 19/58 [00:05<00:07,  5.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.26 GB):  34%|███▍      | 20/58 [00:06<00:06,  5.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.25 GB):  34%|███▍      | 20/58 [00:06<00:06,  5.86it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=43.25 GB):  36%|███▌      | 21/58 [00:06<00:05,  6.49it/s]Capturing num tokens (num_tokens=960 avail_mem=43.25 GB):  36%|███▌      | 21/58 [00:06<00:05,  6.49it/s] Capturing num tokens (num_tokens=960 avail_mem=43.25 GB):  38%|███▊      | 22/58 [00:06<00:05,  7.02it/s]Capturing num tokens (num_tokens=896 avail_mem=43.25 GB):  38%|███▊      | 22/58 [00:06<00:05,  7.02it/s]

    Capturing num tokens (num_tokens=896 avail_mem=43.25 GB):  40%|███▉      | 23/58 [00:06<00:04,  7.52it/s]Capturing num tokens (num_tokens=832 avail_mem=43.24 GB):  40%|███▉      | 23/58 [00:06<00:04,  7.52it/s]Capturing num tokens (num_tokens=832 avail_mem=43.24 GB):  41%|████▏     | 24/58 [00:06<00:04,  7.90it/s]Capturing num tokens (num_tokens=768 avail_mem=43.24 GB):  41%|████▏     | 24/58 [00:06<00:04,  7.90it/s]

    Capturing num tokens (num_tokens=768 avail_mem=43.24 GB):  43%|████▎     | 25/58 [00:06<00:03,  8.34it/s]Capturing num tokens (num_tokens=704 avail_mem=43.23 GB):  43%|████▎     | 25/58 [00:06<00:03,  8.34it/s]Capturing num tokens (num_tokens=704 avail_mem=43.23 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.69it/s]Capturing num tokens (num_tokens=640 avail_mem=43.23 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.69it/s]

    Capturing num tokens (num_tokens=576 avail_mem=43.23 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.69it/s]Capturing num tokens (num_tokens=576 avail_mem=43.23 GB):  48%|████▊     | 28/58 [00:06<00:03,  9.25it/s]Capturing num tokens (num_tokens=512 avail_mem=43.22 GB):  48%|████▊     | 28/58 [00:06<00:03,  9.25it/s]Capturing num tokens (num_tokens=480 avail_mem=43.22 GB):  48%|████▊     | 28/58 [00:07<00:03,  9.25it/s]

    Capturing num tokens (num_tokens=480 avail_mem=43.22 GB):  52%|█████▏    | 30/58 [00:07<00:02,  9.90it/s]Capturing num tokens (num_tokens=448 avail_mem=43.22 GB):  52%|█████▏    | 30/58 [00:07<00:02,  9.90it/s]Capturing num tokens (num_tokens=416 avail_mem=43.21 GB):  52%|█████▏    | 30/58 [00:07<00:02,  9.90it/s]Capturing num tokens (num_tokens=384 avail_mem=43.21 GB):  52%|█████▏    | 30/58 [00:07<00:02,  9.90it/s]Capturing num tokens (num_tokens=384 avail_mem=43.21 GB):  57%|█████▋    | 33/58 [00:07<00:01, 13.63it/s]Capturing num tokens (num_tokens=352 avail_mem=43.20 GB):  57%|█████▋    | 33/58 [00:07<00:01, 13.63it/s]Capturing num tokens (num_tokens=320 avail_mem=43.20 GB):  57%|█████▋    | 33/58 [00:07<00:01, 13.63it/s]Capturing num tokens (num_tokens=288 avail_mem=43.21 GB):  57%|█████▋    | 33/58 [00:07<00:01, 13.63it/s]

    Capturing num tokens (num_tokens=256 avail_mem=43.20 GB):  57%|█████▋    | 33/58 [00:07<00:01, 13.63it/s]Capturing num tokens (num_tokens=256 avail_mem=43.20 GB):  64%|██████▍   | 37/58 [00:07<00:01, 15.88it/s]Capturing num tokens (num_tokens=240 avail_mem=61.55 GB):  64%|██████▍   | 37/58 [00:07<00:01, 15.88it/s]Capturing num tokens (num_tokens=224 avail_mem=61.55 GB):  64%|██████▍   | 37/58 [00:07<00:01, 15.88it/s]Capturing num tokens (num_tokens=208 avail_mem=61.54 GB):  64%|██████▍   | 37/58 [00:07<00:01, 15.88it/s]Capturing num tokens (num_tokens=192 avail_mem=61.54 GB):  64%|██████▍   | 37/58 [00:07<00:01, 15.88it/s]

    Capturing num tokens (num_tokens=192 avail_mem=61.54 GB):  71%|███████   | 41/58 [00:07<00:00, 20.80it/s]Capturing num tokens (num_tokens=176 avail_mem=61.54 GB):  71%|███████   | 41/58 [00:07<00:00, 20.80it/s]Capturing num tokens (num_tokens=160 avail_mem=61.53 GB):  71%|███████   | 41/58 [00:07<00:00, 20.80it/s]Capturing num tokens (num_tokens=144 avail_mem=61.53 GB):  71%|███████   | 41/58 [00:07<00:00, 20.80it/s]Capturing num tokens (num_tokens=128 avail_mem=61.53 GB):  71%|███████   | 41/58 [00:07<00:00, 20.80it/s]Capturing num tokens (num_tokens=128 avail_mem=61.53 GB):  78%|███████▊  | 45/58 [00:07<00:00, 25.09it/s]Capturing num tokens (num_tokens=112 avail_mem=61.53 GB):  78%|███████▊  | 45/58 [00:07<00:00, 25.09it/s]Capturing num tokens (num_tokens=96 avail_mem=61.52 GB):  78%|███████▊  | 45/58 [00:07<00:00, 25.09it/s] Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  78%|███████▊  | 45/58 [00:07<00:00, 25.09it/s]Capturing num tokens (num_tokens=64 avail_mem=61.51 GB):  78%|███████▊  | 45/58 [00:07<00:00, 25.09it/s]

    Capturing num tokens (num_tokens=48 avail_mem=61.51 GB):  78%|███████▊  | 45/58 [00:07<00:00, 25.09it/s]Capturing num tokens (num_tokens=48 avail_mem=61.51 GB):  86%|████████▌ | 50/58 [00:07<00:00, 29.45it/s]Capturing num tokens (num_tokens=32 avail_mem=61.51 GB):  86%|████████▌ | 50/58 [00:07<00:00, 29.45it/s]Capturing num tokens (num_tokens=28 avail_mem=61.51 GB):  86%|████████▌ | 50/58 [00:07<00:00, 29.45it/s]Capturing num tokens (num_tokens=24 avail_mem=61.50 GB):  86%|████████▌ | 50/58 [00:07<00:00, 29.45it/s]Capturing num tokens (num_tokens=20 avail_mem=61.50 GB):  86%|████████▌ | 50/58 [00:07<00:00, 29.45it/s]Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  86%|████████▌ | 50/58 [00:07<00:00, 29.45it/s]Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  95%|█████████▍| 55/58 [00:07<00:00, 32.78it/s]Capturing num tokens (num_tokens=12 avail_mem=61.49 GB):  95%|█████████▍| 55/58 [00:07<00:00, 32.78it/s]Capturing num tokens (num_tokens=8 avail_mem=61.49 GB):  95%|█████████▍| 55/58 [00:07<00:00, 32.78it/s] Capturing num tokens (num_tokens=4 avail_mem=61.49 GB):  95%|█████████▍| 55/58 [00:07<00:00, 32.78it/s]

    Capturing num tokens (num_tokens=4 avail_mem=61.49 GB): 100%|██████████| 58/58 [00:08<00:00,  7.25it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>To find the sum of 1 and 3, I start by identifying the two numbers involved in the addition.<br><br>Next, I add these two numbers together to get the total.<br><br>Finally, I conclude that the result of 1 plus 3 is 4.<br></think><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Problem:**<br>\[ 1 + 3 \]<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>To find the sum of 1 and 3, I start by identifying the two numbers involved in the addition.<br><br>Next, I add these two numbers together to get the total.<br><br>Finally, I conclude that the result of 1 plus 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Problem:**<br>\[ 1 + 3 \]<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
