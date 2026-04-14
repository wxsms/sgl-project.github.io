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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-04-14 01:06:00] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-14 01:06:00] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-14 01:06:00] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-14 01:06:00] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-04-14 01:06:01] Ignore import error when loading sglang.srt.models.minimax_m2: cannot import name 'get_bool_env_var' from 'sglang.srt.distributed' (/actions-runner/_work/sglang/sglang/python/sglang/srt/distributed/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:00<00:00,  1.17it/s]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.28s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.22s/it]


    2026-04-14 01:06:05,210 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 01:06:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:03,  3.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:03,  3.22s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.08it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.08it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:34,  1.58it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:34,  1.58it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.18it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.18it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.82it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.82it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.56it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.56it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.36it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.36it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.26it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.26it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.26it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.97it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.97it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.97it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.52it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.52it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.52it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04, 10.11it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04, 10.11it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04, 10.11it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 12.05it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 12.05it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 12.05it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 12.05it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.47it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.47it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.47it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.47it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.47it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 19.11it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 19.11it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 19.11it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 19.11it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 19.00it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 21.07it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 21.07it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 21.07it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 21.07it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:06<00:01, 21.83it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:06<00:01, 21.83it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:06<00:01, 21.83it/s]

    Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:06<00:01, 21.83it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:06<00:00, 22.70it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:06<00:00, 22.70it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:06<00:00, 22.70it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:06<00:00, 22.70it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:06<00:00, 23.29it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:06<00:00, 23.29it/s]

    Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:06<00:00, 23.29it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:06<00:00, 23.29it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 22.93it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 22.93it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 22.93it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 22.93it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:06<00:00, 23.26it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:06<00:00, 23.26it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:06<00:00, 23.26it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:06<00:00, 23.26it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 23.37it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 23.37it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 23.37it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 23.37it/s]

    Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 24.63it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 24.63it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 24.63it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:06<00:00, 24.63it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 25.63it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 25.63it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=93.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=93.07 GB):   2%|▏         | 1/58 [00:00<00:36,  1.56it/s]Capturing num tokens (num_tokens=7680 avail_mem=93.04 GB):   2%|▏         | 1/58 [00:00<00:36,  1.56it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=93.04 GB):   3%|▎         | 2/58 [00:01<00:32,  1.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=93.04 GB):   3%|▎         | 2/58 [00:01<00:32,  1.70it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=93.04 GB):   5%|▌         | 3/58 [00:01<00:29,  1.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=93.04 GB):   5%|▌         | 3/58 [00:01<00:29,  1.84it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=93.04 GB):   7%|▋         | 4/58 [00:02<00:27,  1.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=93.04 GB):   7%|▋         | 4/58 [00:02<00:27,  1.98it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=93.04 GB):   9%|▊         | 5/58 [00:02<00:25,  2.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=93.05 GB):   9%|▊         | 5/58 [00:02<00:25,  2.09it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=93.05 GB):  10%|█         | 6/58 [00:02<00:22,  2.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=93.05 GB):  10%|█         | 6/58 [00:02<00:22,  2.32it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=93.05 GB):  12%|█▏        | 7/58 [00:03<00:19,  2.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=93.05 GB):  12%|█▏        | 7/58 [00:03<00:19,  2.62it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=93.05 GB):  14%|█▍        | 8/58 [00:03<00:16,  3.02it/s]Capturing num tokens (num_tokens=4096 avail_mem=93.06 GB):  14%|█▍        | 8/58 [00:03<00:16,  3.02it/s]Capturing num tokens (num_tokens=4096 avail_mem=93.06 GB):  16%|█▌        | 9/58 [00:03<00:14,  3.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=92.92 GB):  16%|█▌        | 9/58 [00:03<00:14,  3.48it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=92.92 GB):  17%|█▋        | 10/58 [00:03<00:12,  3.91it/s]Capturing num tokens (num_tokens=3584 avail_mem=92.51 GB):  17%|█▋        | 10/58 [00:03<00:12,  3.91it/s]Capturing num tokens (num_tokens=3584 avail_mem=92.51 GB):  19%|█▉        | 11/58 [00:03<00:10,  4.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=92.13 GB):  19%|█▉        | 11/58 [00:03<00:10,  4.39it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=92.13 GB):  21%|██        | 12/58 [00:04<00:08,  5.16it/s]Capturing num tokens (num_tokens=3072 avail_mem=91.77 GB):  21%|██        | 12/58 [00:04<00:08,  5.16it/s]Capturing num tokens (num_tokens=3072 avail_mem=91.77 GB):  22%|██▏       | 13/58 [00:04<00:07,  5.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=91.74 GB):  22%|██▏       | 13/58 [00:04<00:07,  5.67it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=91.74 GB):  22%|██▏       | 13/58 [00:04<00:07,  5.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=91.74 GB):  26%|██▌       | 15/58 [00:04<00:05,  7.29it/s]Capturing num tokens (num_tokens=2304 avail_mem=91.25 GB):  26%|██▌       | 15/58 [00:04<00:05,  7.29it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=91.25 GB):  28%|██▊       | 16/58 [00:04<00:06,  6.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=90.55 GB):  28%|██▊       | 16/58 [00:04<00:06,  6.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=90.55 GB):  29%|██▉       | 17/58 [00:04<00:06,  6.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=90.74 GB):  29%|██▉       | 17/58 [00:04<00:06,  6.16it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=90.74 GB):  31%|███       | 18/58 [00:04<00:06,  6.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=91.55 GB):  31%|███       | 18/58 [00:04<00:06,  6.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=91.55 GB):  33%|███▎      | 19/58 [00:05<00:05,  6.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=90.73 GB):  33%|███▎      | 19/58 [00:05<00:05,  6.52it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=90.73 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=90.73 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=90.73 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.02it/s]Capturing num tokens (num_tokens=960 avail_mem=91.03 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.02it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=91.03 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.69it/s]Capturing num tokens (num_tokens=896 avail_mem=91.02 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.69it/s]Capturing num tokens (num_tokens=896 avail_mem=91.02 GB):  40%|███▉      | 23/58 [00:05<00:04,  8.11it/s]Capturing num tokens (num_tokens=832 avail_mem=90.26 GB):  40%|███▉      | 23/58 [00:05<00:04,  8.11it/s]

    Capturing num tokens (num_tokens=832 avail_mem=90.26 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.99it/s]Capturing num tokens (num_tokens=768 avail_mem=90.25 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.99it/s]Capturing num tokens (num_tokens=768 avail_mem=90.25 GB):  43%|████▎     | 25/58 [00:05<00:04,  8.02it/s]Capturing num tokens (num_tokens=704 avail_mem=89.16 GB):  43%|████▎     | 25/58 [00:05<00:04,  8.02it/s]

    Capturing num tokens (num_tokens=640 avail_mem=88.43 GB):  43%|████▎     | 25/58 [00:05<00:04,  8.02it/s]Capturing num tokens (num_tokens=640 avail_mem=88.43 GB):  47%|████▋     | 27/58 [00:05<00:03,  8.59it/s]Capturing num tokens (num_tokens=576 avail_mem=88.42 GB):  47%|████▋     | 27/58 [00:05<00:03,  8.59it/s]

    Capturing num tokens (num_tokens=576 avail_mem=88.42 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.56it/s]Capturing num tokens (num_tokens=512 avail_mem=88.42 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.56it/s]Capturing num tokens (num_tokens=480 avail_mem=89.13 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.56it/s]Capturing num tokens (num_tokens=480 avail_mem=89.13 GB):  52%|█████▏    | 30/58 [00:06<00:02,  9.42it/s]Capturing num tokens (num_tokens=448 avail_mem=88.47 GB):  52%|█████▏    | 30/58 [00:06<00:02,  9.42it/s]

    Capturing num tokens (num_tokens=448 avail_mem=88.47 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.32it/s]Capturing num tokens (num_tokens=416 avail_mem=88.47 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.32it/s]Capturing num tokens (num_tokens=384 avail_mem=89.12 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.32it/s]Capturing num tokens (num_tokens=384 avail_mem=89.12 GB):  57%|█████▋    | 33/58 [00:06<00:02,  9.85it/s]Capturing num tokens (num_tokens=352 avail_mem=88.52 GB):  57%|█████▋    | 33/58 [00:06<00:02,  9.85it/s]

    Capturing num tokens (num_tokens=352 avail_mem=88.52 GB):  59%|█████▊    | 34/58 [00:06<00:02,  9.68it/s]Capturing num tokens (num_tokens=320 avail_mem=88.51 GB):  59%|█████▊    | 34/58 [00:06<00:02,  9.68it/s]Capturing num tokens (num_tokens=288 avail_mem=89.12 GB):  59%|█████▊    | 34/58 [00:06<00:02,  9.68it/s]Capturing num tokens (num_tokens=288 avail_mem=89.12 GB):  62%|██████▏   | 36/58 [00:06<00:02, 10.45it/s]Capturing num tokens (num_tokens=256 avail_mem=88.56 GB):  62%|██████▏   | 36/58 [00:06<00:02, 10.45it/s]

    Capturing num tokens (num_tokens=240 avail_mem=89.12 GB):  62%|██████▏   | 36/58 [00:06<00:02, 10.45it/s]Capturing num tokens (num_tokens=240 avail_mem=89.12 GB):  66%|██████▌   | 38/58 [00:07<00:01, 11.22it/s]Capturing num tokens (num_tokens=224 avail_mem=89.11 GB):  66%|██████▌   | 38/58 [00:07<00:01, 11.22it/s]Capturing num tokens (num_tokens=208 avail_mem=88.61 GB):  66%|██████▌   | 38/58 [00:07<00:01, 11.22it/s]

    Capturing num tokens (num_tokens=208 avail_mem=88.61 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.43it/s]Capturing num tokens (num_tokens=192 avail_mem=89.11 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.43it/s]Capturing num tokens (num_tokens=176 avail_mem=88.66 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.43it/s]Capturing num tokens (num_tokens=176 avail_mem=88.66 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.74it/s]Capturing num tokens (num_tokens=160 avail_mem=88.66 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.74it/s]

    Capturing num tokens (num_tokens=144 avail_mem=89.10 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.74it/s]Capturing num tokens (num_tokens=144 avail_mem=89.10 GB):  76%|███████▌  | 44/58 [00:07<00:01, 11.97it/s]Capturing num tokens (num_tokens=128 avail_mem=88.69 GB):  76%|███████▌  | 44/58 [00:07<00:01, 11.97it/s]Capturing num tokens (num_tokens=112 avail_mem=89.13 GB):  76%|███████▌  | 44/58 [00:07<00:01, 11.97it/s]

    Capturing num tokens (num_tokens=112 avail_mem=89.13 GB):  79%|███████▉  | 46/58 [00:07<00:00, 13.23it/s]Capturing num tokens (num_tokens=96 avail_mem=89.10 GB):  79%|███████▉  | 46/58 [00:07<00:00, 13.23it/s] Capturing num tokens (num_tokens=80 avail_mem=88.71 GB):  79%|███████▉  | 46/58 [00:07<00:00, 13.23it/s]Capturing num tokens (num_tokens=80 avail_mem=88.71 GB):  83%|████████▎ | 48/58 [00:07<00:00, 13.15it/s]Capturing num tokens (num_tokens=64 avail_mem=89.09 GB):  83%|████████▎ | 48/58 [00:07<00:00, 13.15it/s]

    Capturing num tokens (num_tokens=48 avail_mem=88.73 GB):  83%|████████▎ | 48/58 [00:07<00:00, 13.15it/s]Capturing num tokens (num_tokens=48 avail_mem=88.73 GB):  86%|████████▌ | 50/58 [00:07<00:00, 13.34it/s]Capturing num tokens (num_tokens=32 avail_mem=89.08 GB):  86%|████████▌ | 50/58 [00:07<00:00, 13.34it/s]Capturing num tokens (num_tokens=28 avail_mem=88.75 GB):  86%|████████▌ | 50/58 [00:08<00:00, 13.34it/s]

    Capturing num tokens (num_tokens=28 avail_mem=88.75 GB):  90%|████████▉ | 52/58 [00:08<00:00, 13.32it/s]Capturing num tokens (num_tokens=24 avail_mem=89.08 GB):  90%|████████▉ | 52/58 [00:08<00:00, 13.32it/s]

    Capturing num tokens (num_tokens=20 avail_mem=88.77 GB):  90%|████████▉ | 52/58 [00:08<00:00, 13.32it/s]Capturing num tokens (num_tokens=20 avail_mem=88.77 GB):  93%|█████████▎| 54/58 [00:08<00:00, 10.86it/s]Capturing num tokens (num_tokens=16 avail_mem=89.07 GB):  93%|█████████▎| 54/58 [00:08<00:00, 10.86it/s]Capturing num tokens (num_tokens=12 avail_mem=88.79 GB):  93%|█████████▎| 54/58 [00:08<00:00, 10.86it/s]Capturing num tokens (num_tokens=12 avail_mem=88.79 GB):  97%|█████████▋| 56/58 [00:08<00:00, 12.00it/s]Capturing num tokens (num_tokens=8 avail_mem=89.06 GB):  97%|█████████▋| 56/58 [00:08<00:00, 12.00it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=89.10 GB):  97%|█████████▋| 56/58 [00:08<00:00, 12.00it/s]Capturing num tokens (num_tokens=4 avail_mem=89.10 GB): 100%|██████████| 58/58 [00:08<00:00, 13.28it/s]Capturing num tokens (num_tokens=4 avail_mem=89.10 GB): 100%|██████████| 58/58 [00:08<00:00,  6.76it/s]


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



<strong style='color: #00008B;'>To solve the problem \(1 + 3\), I start by identifying the two numbers involved, which are 1 and 3.<br><br>Next, I add these two numbers together.<br><br>Finally, the sum of 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>To solve the problem \(1 + 3\), follow these easy steps:<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Present the final answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Finally, I present the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that I need to add the numbers 1 and 3.<br><br>Next, I perform the addition: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>


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



<strong style='color: #00008B;'>I need to calculate the sum of 1 and 3.<br><br>First, I identify the two numbers involved: 1 and 3.<br><br>Next, I add these numbers together.<br><br>Finally, the result of the addition is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\(\boxed{4}\)</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\(\boxed{4}\)</strong>



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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:00<00:00,  1.16it/s]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.19s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.14s/it]


    2026-04-14 01:06:57,532 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 01:06:57] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:09,  3.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:09,  3.32s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:29,  1.60s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:29,  1.60s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:52,  1.04it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:52,  1.04it/s]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:35,  1.53it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:35,  1.53it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:25,  2.11it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:25,  2.11it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.74it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.74it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.46it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.46it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.25it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.25it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.14it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.14it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.14it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:07,  6.71it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:07,  6.71it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:07,  6.71it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:05,  8.28it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:05,  8.28it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.28it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.87it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.87it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.87it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.81it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.81it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.81it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.81it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.24it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.24it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.24it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.24it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.24it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.81it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.81it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.81it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.81it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.81it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.81it/s]

    Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:01, 20.81it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 39.91it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 39.91it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 39.91it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 39.91it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 39.91it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 39.91it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 39.91it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 39.91it/s]

    Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 47.69it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 47.69it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 47.69it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 47.69it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 47.69it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 47.69it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 47.69it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 47.69it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 53.76it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 53.76it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 53.76it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 53.76it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 53.76it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 53.76it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 53.76it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 53.76it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=86.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=86.22 GB):   2%|▏         | 1/58 [00:00<00:37,  1.52it/s]Capturing num tokens (num_tokens=7680 avail_mem=86.19 GB):   2%|▏         | 1/58 [00:00<00:37,  1.52it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=86.19 GB):   3%|▎         | 2/58 [00:01<00:34,  1.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=86.19 GB):   3%|▎         | 2/58 [00:01<00:34,  1.64it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=86.19 GB):   5%|▌         | 3/58 [00:01<00:31,  1.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=86.19 GB):   5%|▌         | 3/58 [00:01<00:31,  1.76it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=86.19 GB):   7%|▋         | 4/58 [00:02<00:33,  1.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=85.19 GB):   7%|▋         | 4/58 [00:02<00:33,  1.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=85.19 GB):   9%|▊         | 5/58 [00:03<00:33,  1.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=85.20 GB):   9%|▊         | 5/58 [00:03<00:33,  1.56it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=85.20 GB):  10%|█         | 6/58 [00:03<00:32,  1.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=85.20 GB):  10%|█         | 6/58 [00:03<00:32,  1.60it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=85.20 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=85.20 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.67it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=85.20 GB):  14%|█▍        | 8/58 [00:04<00:28,  1.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=85.21 GB):  14%|█▍        | 8/58 [00:04<00:28,  1.77it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=85.21 GB):  16%|█▌        | 9/58 [00:05<00:26,  1.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=85.21 GB):  16%|█▌        | 9/58 [00:05<00:26,  1.88it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=85.21 GB):  17%|█▋        | 10/58 [00:05<00:23,  2.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=85.21 GB):  17%|█▋        | 10/58 [00:05<00:23,  2.01it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=85.21 GB):  19%|█▉        | 11/58 [00:06<00:21,  2.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=85.21 GB):  19%|█▉        | 11/58 [00:06<00:21,  2.18it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=85.21 GB):  21%|██        | 12/58 [00:06<00:19,  2.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=85.21 GB):  21%|██        | 12/58 [00:06<00:19,  2.38it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=85.21 GB):  22%|██▏       | 13/58 [00:06<00:18,  2.43it/s]Capturing num tokens (num_tokens=2816 avail_mem=85.21 GB):  22%|██▏       | 13/58 [00:06<00:18,  2.43it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=85.21 GB):  24%|██▍       | 14/58 [00:07<00:17,  2.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=85.20 GB):  24%|██▍       | 14/58 [00:07<00:17,  2.53it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=85.20 GB):  26%|██▌       | 15/58 [00:07<00:15,  2.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=85.20 GB):  26%|██▌       | 15/58 [00:07<00:15,  2.85it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=85.20 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=85.20 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=85.20 GB):  29%|██▉       | 17/58 [00:07<00:10,  3.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=85.20 GB):  29%|██▉       | 17/58 [00:07<00:10,  3.82it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=85.20 GB):  31%|███       | 18/58 [00:07<00:09,  4.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=84.01 GB):  31%|███       | 18/58 [00:07<00:09,  4.29it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=84.01 GB):  33%|███▎      | 19/58 [00:08<00:09,  4.29it/s]Capturing num tokens (num_tokens=1280 avail_mem=84.01 GB):  33%|███▎      | 19/58 [00:08<00:09,  4.29it/s]Capturing num tokens (num_tokens=1280 avail_mem=84.01 GB):  34%|███▍      | 20/58 [00:08<00:08,  4.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=84.01 GB):  34%|███▍      | 20/58 [00:08<00:08,  4.50it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=84.01 GB):  36%|███▌      | 21/58 [00:08<00:07,  5.00it/s]Capturing num tokens (num_tokens=960 avail_mem=85.16 GB):  36%|███▌      | 21/58 [00:08<00:07,  5.00it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=85.16 GB):  38%|███▊      | 22/58 [00:08<00:07,  4.63it/s]Capturing num tokens (num_tokens=896 avail_mem=84.17 GB):  38%|███▊      | 22/58 [00:08<00:07,  4.63it/s]

    Capturing num tokens (num_tokens=896 avail_mem=84.17 GB):  40%|███▉      | 23/58 [00:08<00:08,  4.29it/s]Capturing num tokens (num_tokens=832 avail_mem=85.15 GB):  40%|███▉      | 23/58 [00:08<00:08,  4.29it/s]Capturing num tokens (num_tokens=832 avail_mem=85.15 GB):  41%|████▏     | 24/58 [00:09<00:07,  4.83it/s]Capturing num tokens (num_tokens=768 avail_mem=84.23 GB):  41%|████▏     | 24/58 [00:09<00:07,  4.83it/s]

    Capturing num tokens (num_tokens=768 avail_mem=84.23 GB):  43%|████▎     | 25/58 [00:09<00:06,  4.86it/s]Capturing num tokens (num_tokens=704 avail_mem=85.15 GB):  43%|████▎     | 25/58 [00:09<00:06,  4.86it/s]Capturing num tokens (num_tokens=704 avail_mem=85.15 GB):  45%|████▍     | 26/58 [00:09<00:05,  5.70it/s]Capturing num tokens (num_tokens=640 avail_mem=84.29 GB):  45%|████▍     | 26/58 [00:09<00:05,  5.70it/s]

    Capturing num tokens (num_tokens=640 avail_mem=84.29 GB):  47%|████▋     | 27/58 [00:09<00:04,  6.30it/s]Capturing num tokens (num_tokens=576 avail_mem=84.28 GB):  47%|████▋     | 27/58 [00:09<00:04,  6.30it/s]Capturing num tokens (num_tokens=576 avail_mem=84.28 GB):  48%|████▊     | 28/58 [00:09<00:04,  7.03it/s]Capturing num tokens (num_tokens=512 avail_mem=85.14 GB):  48%|████▊     | 28/58 [00:09<00:04,  7.03it/s]Capturing num tokens (num_tokens=480 avail_mem=84.35 GB):  48%|████▊     | 28/58 [00:09<00:04,  7.03it/s]

    Capturing num tokens (num_tokens=480 avail_mem=84.35 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.18it/s]Capturing num tokens (num_tokens=448 avail_mem=84.35 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.18it/s]Capturing num tokens (num_tokens=448 avail_mem=84.35 GB):  53%|█████▎    | 31/58 [00:09<00:03,  7.61it/s]Capturing num tokens (num_tokens=416 avail_mem=84.10 GB):  53%|█████▎    | 31/58 [00:10<00:03,  7.61it/s]

    Capturing num tokens (num_tokens=416 avail_mem=84.10 GB):  55%|█████▌    | 32/58 [00:10<00:04,  5.47it/s]Capturing num tokens (num_tokens=384 avail_mem=84.37 GB):  55%|█████▌    | 32/58 [00:10<00:04,  5.47it/s]Capturing num tokens (num_tokens=384 avail_mem=84.37 GB):  57%|█████▋    | 33/58 [00:10<00:04,  5.77it/s]Capturing num tokens (num_tokens=352 avail_mem=85.10 GB):  57%|█████▋    | 33/58 [00:10<00:04,  5.77it/s]

    Capturing num tokens (num_tokens=352 avail_mem=85.10 GB):  59%|█████▊    | 34/58 [00:10<00:04,  5.93it/s]Capturing num tokens (num_tokens=320 avail_mem=83.61 GB):  59%|█████▊    | 34/58 [00:10<00:04,  5.93it/s]Capturing num tokens (num_tokens=320 avail_mem=83.61 GB):  60%|██████    | 35/58 [00:10<00:04,  5.65it/s]Capturing num tokens (num_tokens=288 avail_mem=83.61 GB):  60%|██████    | 35/58 [00:10<00:04,  5.65it/s]

    Capturing num tokens (num_tokens=288 avail_mem=83.61 GB):  62%|██████▏   | 36/58 [00:10<00:03,  5.89it/s]Capturing num tokens (num_tokens=256 avail_mem=85.09 GB):  62%|██████▏   | 36/58 [00:10<00:03,  5.89it/s]Capturing num tokens (num_tokens=256 avail_mem=85.09 GB):  64%|██████▍   | 37/58 [00:11<00:03,  6.21it/s]Capturing num tokens (num_tokens=240 avail_mem=84.49 GB):  64%|██████▍   | 37/58 [00:11<00:03,  6.21it/s]

    Capturing num tokens (num_tokens=240 avail_mem=84.49 GB):  66%|██████▌   | 38/58 [00:11<00:03,  6.38it/s]Capturing num tokens (num_tokens=224 avail_mem=84.32 GB):  66%|██████▌   | 38/58 [00:11<00:03,  6.38it/s]Capturing num tokens (num_tokens=224 avail_mem=84.32 GB):  67%|██████▋   | 39/58 [00:11<00:03,  6.15it/s]Capturing num tokens (num_tokens=208 avail_mem=83.79 GB):  67%|██████▋   | 39/58 [00:11<00:03,  6.15it/s]

    Capturing num tokens (num_tokens=208 avail_mem=83.79 GB):  69%|██████▉   | 40/58 [00:11<00:02,  6.39it/s]Capturing num tokens (num_tokens=192 avail_mem=85.09 GB):  69%|██████▉   | 40/58 [00:11<00:02,  6.39it/s]Capturing num tokens (num_tokens=192 avail_mem=85.09 GB):  71%|███████   | 41/58 [00:11<00:02,  6.54it/s]Capturing num tokens (num_tokens=176 avail_mem=83.88 GB):  71%|███████   | 41/58 [00:11<00:02,  6.54it/s]

    Capturing num tokens (num_tokens=176 avail_mem=83.88 GB):  72%|███████▏  | 42/58 [00:11<00:02,  6.44it/s]Capturing num tokens (num_tokens=160 avail_mem=84.37 GB):  72%|███████▏  | 42/58 [00:11<00:02,  6.44it/s]Capturing num tokens (num_tokens=160 avail_mem=84.37 GB):  74%|███████▍  | 43/58 [00:12<00:02,  6.30it/s]Capturing num tokens (num_tokens=144 avail_mem=83.90 GB):  74%|███████▍  | 43/58 [00:12<00:02,  6.30it/s]

    Capturing num tokens (num_tokens=144 avail_mem=83.90 GB):  76%|███████▌  | 44/58 [00:12<00:02,  6.59it/s]Capturing num tokens (num_tokens=128 avail_mem=85.09 GB):  76%|███████▌  | 44/58 [00:12<00:02,  6.59it/s]Capturing num tokens (num_tokens=128 avail_mem=85.09 GB):  78%|███████▊  | 45/58 [00:12<00:01,  6.76it/s]Capturing num tokens (num_tokens=112 avail_mem=84.00 GB):  78%|███████▊  | 45/58 [00:12<00:01,  6.76it/s]

    Capturing num tokens (num_tokens=112 avail_mem=84.00 GB):  79%|███████▉  | 46/58 [00:12<00:01,  6.70it/s]Capturing num tokens (num_tokens=96 avail_mem=84.42 GB):  79%|███████▉  | 46/58 [00:12<00:01,  6.70it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=84.42 GB):  81%|████████  | 47/58 [00:12<00:01,  5.76it/s]Capturing num tokens (num_tokens=80 avail_mem=84.68 GB):  81%|████████  | 47/58 [00:12<00:01,  5.76it/s]Capturing num tokens (num_tokens=80 avail_mem=84.68 GB):  83%|████████▎ | 48/58 [00:12<00:01,  6.23it/s]Capturing num tokens (num_tokens=64 avail_mem=84.47 GB):  83%|████████▎ | 48/58 [00:12<00:01,  6.23it/s]

    Capturing num tokens (num_tokens=64 avail_mem=84.47 GB):  84%|████████▍ | 49/58 [00:13<00:01,  5.58it/s]Capturing num tokens (num_tokens=48 avail_mem=84.76 GB):  84%|████████▍ | 49/58 [00:13<00:01,  5.58it/s]Capturing num tokens (num_tokens=32 avail_mem=84.99 GB):  84%|████████▍ | 49/58 [00:13<00:01,  5.58it/s]

    Capturing num tokens (num_tokens=32 avail_mem=84.99 GB):  88%|████████▊ | 51/58 [00:13<00:01,  6.58it/s]Capturing num tokens (num_tokens=28 avail_mem=84.53 GB):  88%|████████▊ | 51/58 [00:13<00:01,  6.58it/s]Capturing num tokens (num_tokens=28 avail_mem=84.53 GB):  90%|████████▉ | 52/58 [00:13<00:00,  6.82it/s]Capturing num tokens (num_tokens=24 avail_mem=84.77 GB):  90%|████████▉ | 52/58 [00:13<00:00,  6.82it/s]

    Capturing num tokens (num_tokens=24 avail_mem=84.77 GB):  91%|█████████▏| 53/58 [00:13<00:00,  7.41it/s]Capturing num tokens (num_tokens=20 avail_mem=84.57 GB):  91%|█████████▏| 53/58 [00:13<00:00,  7.41it/s]Capturing num tokens (num_tokens=20 avail_mem=84.57 GB):  93%|█████████▎| 54/58 [00:13<00:00,  7.16it/s]Capturing num tokens (num_tokens=16 avail_mem=84.57 GB):  93%|█████████▎| 54/58 [00:13<00:00,  7.16it/s]

    Capturing num tokens (num_tokens=16 avail_mem=84.57 GB):  95%|█████████▍| 55/58 [00:13<00:00,  7.43it/s]Capturing num tokens (num_tokens=12 avail_mem=85.06 GB):  95%|█████████▍| 55/58 [00:13<00:00,  7.43it/s]Capturing num tokens (num_tokens=12 avail_mem=85.06 GB):  97%|█████████▋| 56/58 [00:13<00:00,  7.73it/s]Capturing num tokens (num_tokens=8 avail_mem=84.42 GB):  97%|█████████▋| 56/58 [00:13<00:00,  7.73it/s] Capturing num tokens (num_tokens=4 avail_mem=85.12 GB):  97%|█████████▋| 56/58 [00:14<00:00,  7.73it/s]

    Capturing num tokens (num_tokens=4 avail_mem=85.12 GB): 100%|██████████| 58/58 [00:14<00:00,  8.88it/s]Capturing num tokens (num_tokens=4 avail_mem=85.12 GB): 100%|██████████| 58/58 [00:14<00:00,  4.11it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
