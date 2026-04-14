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


    [2026-04-14 00:21:15] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-14 00:21:15] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-14 00:21:15] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-14 00:21:15] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-14 00:21:15] Ignore import error when loading sglang.srt.models.minimax_m2: cannot import name 'get_bool_env_var' from 'sglang.srt.distributed' (/actions-runner/_work/sglang/sglang/python/sglang/srt/distributed/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.27s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.36s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]


    2026-04-14 00:21:19,754 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 00:21:19] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:58,  3.13s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:58,  3.13s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:24,  1.51s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:24,  1.51s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:49,  1.10it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:49,  1.10it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.20it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.20it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.85it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.85it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.51it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.51it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.19it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.19it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  4.96it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  4.96it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:08,  5.78it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:08,  5.78it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:07,  6.51it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:07,  6.51it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:07,  6.51it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.32it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.32it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.32it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04, 10.14it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04, 10.14it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04, 10.14it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 12.13it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 12.13it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 12.13it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 12.13it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.09it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.09it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.09it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.09it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:01, 18.04it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:01, 18.04it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:01, 18.04it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:01, 18.04it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 19.98it/s]

    Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 23.94it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 23.94it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 23.94it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 23.94it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 23.94it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 27.00it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 27.00it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 27.00it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 27.00it/s]

    Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 27.00it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 30.26it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 30.26it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 30.26it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 30.26it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 30.26it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:06<00:00, 30.26it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:06<00:00, 30.26it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 33.12it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 33.12it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 33.12it/s]

    Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 33.12it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:06<00:00, 33.12it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:06<00:00, 33.12it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 36.43it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 36.43it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 36.43it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 36.43it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:06<00:00, 36.43it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:06<00:00, 36.43it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:06<00:00, 36.43it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:06<00:00, 36.43it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:06<00:00, 36.43it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 46.98it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 46.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=103.23 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=103.23 GB):   2%|▏         | 1/58 [00:00<00:17,  3.25it/s]Capturing num tokens (num_tokens=7680 avail_mem=103.20 GB):   2%|▏         | 1/58 [00:00<00:17,  3.25it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=103.20 GB):   3%|▎         | 2/58 [00:00<00:15,  3.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=103.20 GB):   3%|▎         | 2/58 [00:00<00:15,  3.54it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=103.20 GB):   5%|▌         | 3/58 [00:00<00:14,  3.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=103.20 GB):   5%|▌         | 3/58 [00:00<00:14,  3.82it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=103.20 GB):   7%|▋         | 4/58 [00:01<00:13,  4.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=103.20 GB):   7%|▋         | 4/58 [00:01<00:13,  4.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=103.20 GB):   9%|▊         | 5/58 [00:01<00:12,  4.31it/s]Capturing num tokens (num_tokens=5632 avail_mem=103.20 GB):   9%|▊         | 5/58 [00:01<00:12,  4.31it/s]Capturing num tokens (num_tokens=5632 avail_mem=103.20 GB):  10%|█         | 6/58 [00:01<00:11,  4.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=103.20 GB):  10%|█         | 6/58 [00:01<00:11,  4.73it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=103.20 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=103.21 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=103.21 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=103.18 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.59it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=103.18 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=103.19 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=103.19 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=103.19 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.61it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=103.19 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=103.19 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.89it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=103.19 GB):  21%|██        | 12/58 [00:02<00:07,  5.78it/s]Capturing num tokens (num_tokens=3072 avail_mem=103.18 GB):  21%|██        | 12/58 [00:02<00:07,  5.78it/s]Capturing num tokens (num_tokens=3072 avail_mem=103.18 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.57it/s]Capturing num tokens (num_tokens=2816 avail_mem=103.19 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=103.19 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.57it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=103.19 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=103.19 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=103.18 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=103.18 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=103.19 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.95it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=103.18 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=103.18 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.70it/s]Capturing num tokens (num_tokens=1280 avail_mem=103.18 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.70it/s]Capturing num tokens (num_tokens=1024 avail_mem=103.18 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.70it/s]Capturing num tokens (num_tokens=960 avail_mem=103.18 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.70it/s] Capturing num tokens (num_tokens=960 avail_mem=103.18 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.68it/s]Capturing num tokens (num_tokens=896 avail_mem=103.17 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.68it/s]

    Capturing num tokens (num_tokens=832 avail_mem=103.17 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.68it/s]Capturing num tokens (num_tokens=768 avail_mem=103.17 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.68it/s]Capturing num tokens (num_tokens=768 avail_mem=103.17 GB):  43%|████▎     | 25/58 [00:03<00:01, 16.62it/s]Capturing num tokens (num_tokens=704 avail_mem=103.16 GB):  43%|████▎     | 25/58 [00:03<00:01, 16.62it/s]Capturing num tokens (num_tokens=640 avail_mem=103.16 GB):  43%|████▎     | 25/58 [00:03<00:01, 16.62it/s]Capturing num tokens (num_tokens=576 avail_mem=103.16 GB):  43%|████▎     | 25/58 [00:03<00:01, 16.62it/s]

    Capturing num tokens (num_tokens=576 avail_mem=103.16 GB):  48%|████▊     | 28/58 [00:03<00:01, 17.42it/s]Capturing num tokens (num_tokens=512 avail_mem=103.15 GB):  48%|████▊     | 28/58 [00:03<00:01, 17.42it/s]Capturing num tokens (num_tokens=480 avail_mem=103.15 GB):  48%|████▊     | 28/58 [00:03<00:01, 17.42it/s]Capturing num tokens (num_tokens=480 avail_mem=103.15 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.68it/s]

    Capturing num tokens (num_tokens=448 avail_mem=103.15 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.68it/s]Capturing num tokens (num_tokens=416 avail_mem=103.14 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.68it/s]Capturing num tokens (num_tokens=416 avail_mem=103.14 GB):  55%|█████▌    | 32/58 [00:03<00:01, 13.34it/s]Capturing num tokens (num_tokens=384 avail_mem=103.14 GB):  55%|█████▌    | 32/58 [00:03<00:01, 13.34it/s]

    Capturing num tokens (num_tokens=352 avail_mem=103.13 GB):  55%|█████▌    | 32/58 [00:03<00:01, 13.34it/s]Capturing num tokens (num_tokens=352 avail_mem=103.13 GB):  59%|█████▊    | 34/58 [00:03<00:02, 11.51it/s]Capturing num tokens (num_tokens=320 avail_mem=103.13 GB):  59%|█████▊    | 34/58 [00:03<00:02, 11.51it/s]

    Capturing num tokens (num_tokens=288 avail_mem=103.13 GB):  59%|█████▊    | 34/58 [00:04<00:02, 11.51it/s]Capturing num tokens (num_tokens=288 avail_mem=103.13 GB):  62%|██████▏   | 36/58 [00:04<00:02, 10.63it/s]Capturing num tokens (num_tokens=256 avail_mem=103.12 GB):  62%|██████▏   | 36/58 [00:04<00:02, 10.63it/s]Capturing num tokens (num_tokens=240 avail_mem=103.12 GB):  62%|██████▏   | 36/58 [00:04<00:02, 10.63it/s]Capturing num tokens (num_tokens=240 avail_mem=103.12 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.14it/s]Capturing num tokens (num_tokens=224 avail_mem=103.12 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.14it/s]Capturing num tokens (num_tokens=208 avail_mem=103.11 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.14it/s]

    Capturing num tokens (num_tokens=192 avail_mem=103.11 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.14it/s]Capturing num tokens (num_tokens=176 avail_mem=103.10 GB):  66%|██████▌   | 38/58 [00:04<00:01, 12.14it/s]Capturing num tokens (num_tokens=176 avail_mem=103.10 GB):  72%|███████▏  | 42/58 [00:04<00:00, 17.38it/s]Capturing num tokens (num_tokens=160 avail_mem=103.10 GB):  72%|███████▏  | 42/58 [00:04<00:00, 17.38it/s]Capturing num tokens (num_tokens=144 avail_mem=103.10 GB):  72%|███████▏  | 42/58 [00:04<00:00, 17.38it/s]Capturing num tokens (num_tokens=128 avail_mem=103.11 GB):  72%|███████▏  | 42/58 [00:04<00:00, 17.38it/s]Capturing num tokens (num_tokens=112 avail_mem=103.10 GB):  72%|███████▏  | 42/58 [00:04<00:00, 17.38it/s]Capturing num tokens (num_tokens=112 avail_mem=103.10 GB):  79%|███████▉  | 46/58 [00:04<00:00, 21.25it/s]Capturing num tokens (num_tokens=96 avail_mem=103.10 GB):  79%|███████▉  | 46/58 [00:04<00:00, 21.25it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=103.09 GB):  79%|███████▉  | 46/58 [00:04<00:00, 21.25it/s]Capturing num tokens (num_tokens=64 avail_mem=103.09 GB):  79%|███████▉  | 46/58 [00:04<00:00, 21.25it/s]Capturing num tokens (num_tokens=48 avail_mem=103.09 GB):  79%|███████▉  | 46/58 [00:04<00:00, 21.25it/s]Capturing num tokens (num_tokens=48 avail_mem=103.09 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.70it/s]Capturing num tokens (num_tokens=32 avail_mem=103.08 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.70it/s]Capturing num tokens (num_tokens=28 avail_mem=103.08 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.70it/s]Capturing num tokens (num_tokens=24 avail_mem=103.08 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.70it/s]Capturing num tokens (num_tokens=20 avail_mem=103.07 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.70it/s]

    Capturing num tokens (num_tokens=20 avail_mem=103.07 GB):  93%|█████████▎| 54/58 [00:04<00:00, 27.37it/s]Capturing num tokens (num_tokens=16 avail_mem=103.07 GB):  93%|█████████▎| 54/58 [00:04<00:00, 27.37it/s]Capturing num tokens (num_tokens=12 avail_mem=103.07 GB):  93%|█████████▎| 54/58 [00:04<00:00, 27.37it/s]Capturing num tokens (num_tokens=8 avail_mem=103.06 GB):  93%|█████████▎| 54/58 [00:04<00:00, 27.37it/s] Capturing num tokens (num_tokens=4 avail_mem=103.06 GB):  93%|█████████▎| 54/58 [00:04<00:00, 27.37it/s]Capturing num tokens (num_tokens=4 avail_mem=103.06 GB): 100%|██████████| 58/58 [00:04<00:00, 29.29it/s]Capturing num tokens (num_tokens=4 avail_mem=103.06 GB): 100%|██████████| 58/58 [00:04<00:00, 11.98it/s]


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



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I identify the two numbers that need to be added: 1 and 3.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum to determine the result.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the addition step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**  <br>1. **Identify the numbers to add:**  <br>   We have the numbers **1** and **3**.<br><br>2. **Perform the addition:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**  <br>   \[<br>   \boxed{4}<br>   \]<br><br>**Answer:**  <br>\(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I start by identifying the two numbers involved in the addition: 1 and 3.<br><br>Next, I perform the addition by combining these two numbers.<br><br>Finally, I arrive at the conclusion that 1 plus 3 equals 4.</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the addition step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**  <br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>I need to calculate the sum of 1 and 3. <br><br>First, I identify the two numbers involved in the addition: 1 and 3.<br><br>Next, I add these two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>Sure! Let's solve the addition step by step.<br><br>**Problem:**<br>\[ 1 + 3 \]<br><br>**Step 1:** Identify the numbers to add.<br>- Number 1<br>- Number 3<br><br>**Step 2:** Add the two numbers together.<br>\[ 1 + 3 = 4 \]<br><br>**Final Answer:**<br>\[ \boxed{4} \]</strong>


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



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved.<br><br>Next, I perform the addition by combining these two numbers together.<br><br>Finally, I calculate the sum to find the result of 1 plus 3.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved.<br><br>Next, I perform the addition by combining these two numbers together.<br><br>Finally, I calculate the sum to find the result of 1 plus 3.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.29s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.36s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]


    2026-04-14 00:22:06,866 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 00:22:06] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:56,  3.09s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:56,  3.09s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:24,  1.51s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:24,  1.51s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:49,  1.10it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:49,  1.10it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.21it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.21it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.84it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.84it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.57it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.57it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.35it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.35it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.24it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.24it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.24it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.90it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.90it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.90it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.44it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.44it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.44it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04, 10.02it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04, 10.02it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04, 10.02it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.95it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.95it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.95it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.95it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.37it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.37it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.37it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.37it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.37it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.90it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.90it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.90it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.90it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.90it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.90it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:01, 20.90it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 40.09it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 40.09it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 40.09it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 40.09it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 40.09it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 40.09it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 40.09it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 40.09it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 40.09it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 49.73it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 49.73it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 49.73it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 49.73it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 49.73it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 49.73it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 49.73it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 49.73it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 49.73it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 56.01it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 56.01it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 56.01it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 56.01it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 56.01it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 56.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=122.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=122.24 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=122.21 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=122.21 GB):   3%|▎         | 2/58 [00:00<00:15,  3.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=122.21 GB):   3%|▎         | 2/58 [00:00<00:15,  3.52it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=122.21 GB):   5%|▌         | 3/58 [00:00<00:14,  3.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=122.22 GB):   5%|▌         | 3/58 [00:00<00:14,  3.74it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=122.22 GB):   7%|▋         | 4/58 [00:01<00:13,  4.01it/s]Capturing num tokens (num_tokens=6144 avail_mem=122.22 GB):   7%|▋         | 4/58 [00:01<00:13,  4.01it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=122.22 GB):   9%|▊         | 5/58 [00:01<00:12,  4.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=122.22 GB):   9%|▊         | 5/58 [00:01<00:12,  4.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=122.22 GB):  10%|█         | 6/58 [00:01<00:11,  4.62it/s]Capturing num tokens (num_tokens=5120 avail_mem=122.22 GB):  10%|█         | 6/58 [00:01<00:11,  4.62it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=122.22 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=122.23 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=122.23 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=122.23 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.20it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=122.23 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=122.24 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=122.24 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=122.24 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.25it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=122.24 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=122.24 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=122.24 GB):  21%|██        | 12/58 [00:02<00:06,  7.43it/s]Capturing num tokens (num_tokens=3072 avail_mem=122.23 GB):  21%|██        | 12/58 [00:02<00:06,  7.43it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=122.23 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=122.24 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=122.24 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=122.24 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.44it/s]Capturing num tokens (num_tokens=2304 avail_mem=122.24 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.44it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=122.23 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=122.23 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=122.24 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=122.23 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=122.23 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=122.23 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.73it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=122.23 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.73it/s]Capturing num tokens (num_tokens=960 avail_mem=122.23 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.73it/s] Capturing num tokens (num_tokens=960 avail_mem=122.23 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.01it/s]Capturing num tokens (num_tokens=896 avail_mem=122.22 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.01it/s]Capturing num tokens (num_tokens=832 avail_mem=122.22 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.01it/s]Capturing num tokens (num_tokens=768 avail_mem=122.22 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.01it/s]Capturing num tokens (num_tokens=768 avail_mem=122.22 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.06it/s]Capturing num tokens (num_tokens=704 avail_mem=122.21 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.06it/s]

    Capturing num tokens (num_tokens=640 avail_mem=122.21 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.06it/s]Capturing num tokens (num_tokens=576 avail_mem=122.21 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.06it/s]Capturing num tokens (num_tokens=512 avail_mem=122.20 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.06it/s]Capturing num tokens (num_tokens=512 avail_mem=122.20 GB):  50%|█████     | 29/58 [00:03<00:01, 22.80it/s]Capturing num tokens (num_tokens=480 avail_mem=122.20 GB):  50%|█████     | 29/58 [00:03<00:01, 22.80it/s]Capturing num tokens (num_tokens=448 avail_mem=122.20 GB):  50%|█████     | 29/58 [00:03<00:01, 22.80it/s]Capturing num tokens (num_tokens=416 avail_mem=122.19 GB):  50%|█████     | 29/58 [00:03<00:01, 22.80it/s]Capturing num tokens (num_tokens=384 avail_mem=122.19 GB):  50%|█████     | 29/58 [00:03<00:01, 22.80it/s]

    Capturing num tokens (num_tokens=384 avail_mem=122.19 GB):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Capturing num tokens (num_tokens=352 avail_mem=122.18 GB):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Capturing num tokens (num_tokens=320 avail_mem=122.18 GB):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Capturing num tokens (num_tokens=288 avail_mem=122.18 GB):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Capturing num tokens (num_tokens=256 avail_mem=122.17 GB):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Capturing num tokens (num_tokens=256 avail_mem=122.17 GB):  64%|██████▍   | 37/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=240 avail_mem=122.17 GB):  64%|██████▍   | 37/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=224 avail_mem=122.17 GB):  64%|██████▍   | 37/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=208 avail_mem=122.16 GB):  64%|██████▍   | 37/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=192 avail_mem=122.16 GB):  64%|██████▍   | 37/58 [00:03<00:00, 29.25it/s]

    Capturing num tokens (num_tokens=192 avail_mem=122.16 GB):  71%|███████   | 41/58 [00:03<00:00, 31.74it/s]Capturing num tokens (num_tokens=176 avail_mem=122.15 GB):  71%|███████   | 41/58 [00:03<00:00, 31.74it/s]Capturing num tokens (num_tokens=160 avail_mem=122.15 GB):  71%|███████   | 41/58 [00:03<00:00, 31.74it/s]Capturing num tokens (num_tokens=144 avail_mem=122.15 GB):  71%|███████   | 41/58 [00:03<00:00, 31.74it/s]Capturing num tokens (num_tokens=128 avail_mem=122.16 GB):  71%|███████   | 41/58 [00:03<00:00, 31.74it/s]Capturing num tokens (num_tokens=128 avail_mem=122.16 GB):  78%|███████▊  | 45/58 [00:03<00:00, 33.49it/s]Capturing num tokens (num_tokens=112 avail_mem=122.15 GB):  78%|███████▊  | 45/58 [00:03<00:00, 33.49it/s]Capturing num tokens (num_tokens=96 avail_mem=122.15 GB):  78%|███████▊  | 45/58 [00:03<00:00, 33.49it/s] Capturing num tokens (num_tokens=80 avail_mem=122.14 GB):  78%|███████▊  | 45/58 [00:03<00:00, 33.49it/s]Capturing num tokens (num_tokens=64 avail_mem=122.14 GB):  78%|███████▊  | 45/58 [00:03<00:00, 33.49it/s]

    Capturing num tokens (num_tokens=64 avail_mem=122.14 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.70it/s]Capturing num tokens (num_tokens=48 avail_mem=122.14 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.70it/s]Capturing num tokens (num_tokens=32 avail_mem=122.13 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.70it/s]Capturing num tokens (num_tokens=28 avail_mem=122.13 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.70it/s]Capturing num tokens (num_tokens=24 avail_mem=122.13 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.70it/s]Capturing num tokens (num_tokens=24 avail_mem=122.13 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.67it/s]Capturing num tokens (num_tokens=20 avail_mem=122.12 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.67it/s]Capturing num tokens (num_tokens=16 avail_mem=122.12 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.67it/s]Capturing num tokens (num_tokens=12 avail_mem=122.12 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.67it/s]Capturing num tokens (num_tokens=8 avail_mem=122.11 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.67it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=122.11 GB):  98%|█████████▊| 57/58 [00:03<00:00, 36.43it/s]Capturing num tokens (num_tokens=4 avail_mem=122.11 GB):  98%|█████████▊| 57/58 [00:03<00:00, 36.43it/s]Capturing num tokens (num_tokens=4 avail_mem=122.11 GB): 100%|██████████| 58/58 [00:03<00:00, 14.80it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I start by identifying the two numbers involved: 1 and 3.<br><br>Next, I add these two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I start by identifying the two numbers involved: 1 and 3.<br><br>Next, I add these two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
