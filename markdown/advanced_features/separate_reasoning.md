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


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:03<00:03,  3.51s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:06<00:00,  3.05s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:06<00:00,  3.12s/it]


    2026-05-13 16:59:25,941 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 16:59:25] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:01,  5.29s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:01,  5.29s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:08,  2.30s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:08,  2.30s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:13,  1.34s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:13,  1.34s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:32,  1.61it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:32,  1.61it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:24,  2.17it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:24,  2.17it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.83it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.83it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.57it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.57it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.14it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.14it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.14it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.71it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.71it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.71it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.31it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.31it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.31it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.27it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.27it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.27it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.27it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.54it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.54it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.54it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.54it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.54it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.70it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.70it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.70it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.70it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.70it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.70it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.70it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.25it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.25it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.25it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.25it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.25it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.25it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.25it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.25it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.25it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 46.37it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 46.37it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 46.37it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 46.37it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 46.37it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 46.37it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 46.37it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 46.37it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 46.37it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 54.98it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 54.98it/s]

    Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 54.98it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 54.98it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 54.98it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 54.98it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 54.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=48.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=48.13 GB):   2%|▏         | 1/58 [00:00<00:27,  2.05it/s]Capturing num tokens (num_tokens=7680 avail_mem=48.09 GB):   2%|▏         | 1/58 [00:00<00:27,  2.05it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=48.09 GB):   3%|▎         | 2/58 [00:00<00:26,  2.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=48.09 GB):   3%|▎         | 2/58 [00:00<00:26,  2.12it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=48.09 GB):   5%|▌         | 3/58 [00:01<00:23,  2.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=48.09 GB):   5%|▌         | 3/58 [00:01<00:23,  2.30it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=48.09 GB):   7%|▋         | 4/58 [00:01<00:20,  2.67it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.15 GB):   7%|▋         | 4/58 [00:01<00:20,  2.67it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.15 GB):   9%|▊         | 5/58 [00:01<00:16,  3.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.15 GB):   9%|▊         | 5/58 [00:01<00:16,  3.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.15 GB):  10%|█         | 6/58 [00:02<00:14,  3.71it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.15 GB):  10%|█         | 6/58 [00:02<00:14,  3.71it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=57.15 GB):  12%|█▏        | 7/58 [00:02<00:12,  4.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.15 GB):  12%|█▏        | 7/58 [00:02<00:12,  4.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.15 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.15 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.78it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=57.15 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.15 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.15 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.15 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=57.15 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.15 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.15 GB):  21%|██        | 12/58 [00:02<00:06,  7.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.15 GB):  21%|██        | 12/58 [00:02<00:06,  7.14it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=57.15 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.15 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.14 GB):  22%|██▏       | 13/58 [00:03<00:05,  7.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.14 GB):  26%|██▌       | 15/58 [00:03<00:04,  9.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.14 GB):  26%|██▌       | 15/58 [00:03<00:04,  9.20it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=57.14 GB):  26%|██▌       | 15/58 [00:03<00:04,  9.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.14 GB):  29%|██▉       | 17/58 [00:03<00:03, 10.73it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.14 GB):  29%|██▉       | 17/58 [00:03<00:03, 10.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.13 GB):  29%|██▉       | 17/58 [00:03<00:03, 10.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.13 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.13 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=57.12 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s]Capturing num tokens (num_tokens=960 avail_mem=57.12 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s] Capturing num tokens (num_tokens=960 avail_mem=57.12 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.75it/s]Capturing num tokens (num_tokens=896 avail_mem=57.11 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.75it/s]Capturing num tokens (num_tokens=832 avail_mem=57.11 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.75it/s]Capturing num tokens (num_tokens=768 avail_mem=57.11 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.75it/s]Capturing num tokens (num_tokens=768 avail_mem=57.11 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.84it/s]Capturing num tokens (num_tokens=704 avail_mem=57.10 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.84it/s]

    Capturing num tokens (num_tokens=640 avail_mem=57.10 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.84it/s]Capturing num tokens (num_tokens=576 avail_mem=57.10 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.84it/s]Capturing num tokens (num_tokens=576 avail_mem=57.10 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.63it/s]Capturing num tokens (num_tokens=512 avail_mem=57.09 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.63it/s]Capturing num tokens (num_tokens=480 avail_mem=57.09 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.63it/s]Capturing num tokens (num_tokens=448 avail_mem=57.09 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.63it/s]Capturing num tokens (num_tokens=416 avail_mem=57.08 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.63it/s]Capturing num tokens (num_tokens=416 avail_mem=57.08 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.90it/s]Capturing num tokens (num_tokens=384 avail_mem=57.08 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.90it/s]

    Capturing num tokens (num_tokens=352 avail_mem=57.07 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.90it/s]Capturing num tokens (num_tokens=320 avail_mem=57.07 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.90it/s]Capturing num tokens (num_tokens=288 avail_mem=57.07 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.90it/s]Capturing num tokens (num_tokens=288 avail_mem=57.07 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.99it/s]Capturing num tokens (num_tokens=256 avail_mem=57.07 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.99it/s]Capturing num tokens (num_tokens=240 avail_mem=57.06 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.99it/s]Capturing num tokens (num_tokens=224 avail_mem=57.06 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.99it/s]Capturing num tokens (num_tokens=208 avail_mem=57.06 GB):  62%|██████▏   | 36/58 [00:04<00:00, 27.99it/s]Capturing num tokens (num_tokens=208 avail_mem=57.06 GB):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]Capturing num tokens (num_tokens=192 avail_mem=57.05 GB):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]

    Capturing num tokens (num_tokens=176 avail_mem=57.05 GB):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]Capturing num tokens (num_tokens=160 avail_mem=57.05 GB):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]Capturing num tokens (num_tokens=144 avail_mem=57.04 GB):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]Capturing num tokens (num_tokens=144 avail_mem=57.04 GB):  76%|███████▌  | 44/58 [00:04<00:00, 31.85it/s]Capturing num tokens (num_tokens=128 avail_mem=57.04 GB):  76%|███████▌  | 44/58 [00:04<00:00, 31.85it/s]Capturing num tokens (num_tokens=112 avail_mem=57.04 GB):  76%|███████▌  | 44/58 [00:04<00:00, 31.85it/s]Capturing num tokens (num_tokens=96 avail_mem=57.04 GB):  76%|███████▌  | 44/58 [00:04<00:00, 31.85it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  76%|███████▌  | 44/58 [00:04<00:00, 31.85it/s]Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  83%|████████▎ | 48/58 [00:04<00:00, 28.08it/s]Capturing num tokens (num_tokens=64 avail_mem=61.52 GB):  83%|████████▎ | 48/58 [00:04<00:00, 28.08it/s]Capturing num tokens (num_tokens=48 avail_mem=61.51 GB):  83%|████████▎ | 48/58 [00:04<00:00, 28.08it/s]Capturing num tokens (num_tokens=32 avail_mem=61.51 GB):  83%|████████▎ | 48/58 [00:04<00:00, 28.08it/s]Capturing num tokens (num_tokens=28 avail_mem=61.51 GB):  83%|████████▎ | 48/58 [00:04<00:00, 28.08it/s]Capturing num tokens (num_tokens=28 avail_mem=61.51 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.86it/s]Capturing num tokens (num_tokens=24 avail_mem=61.51 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.86it/s]Capturing num tokens (num_tokens=20 avail_mem=61.50 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.86it/s]Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.86it/s]

    Capturing num tokens (num_tokens=12 avail_mem=61.50 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.86it/s]Capturing num tokens (num_tokens=12 avail_mem=61.50 GB):  97%|█████████▋| 56/58 [00:04<00:00, 32.64it/s]Capturing num tokens (num_tokens=8 avail_mem=61.49 GB):  97%|█████████▋| 56/58 [00:04<00:00, 32.64it/s] Capturing num tokens (num_tokens=4 avail_mem=61.49 GB):  97%|█████████▋| 56/58 [00:04<00:00, 32.64it/s]Capturing num tokens (num_tokens=4 avail_mem=61.49 GB): 100%|██████████| 58/58 [00:04<00:00, 12.63it/s]


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



<strong style='color: #00008B;'>First, I identify the two numbers that need to be added, which are 1 and 3.<br><br>Next, I perform the addition of these two numbers to find their sum.<br><br>Finally, I conclude that the result of 1 plus 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We need to calculate the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I identify the two numbers in the addition problem: 1 and 3.<br><br>Next, I add these numbers together to find the sum.<br><br>The result of adding 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Problem:**<br>\[ 1 + 3 \]<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition of these two numbers.<br><br>Finally, I arrive at the conclusion that 1 plus 3 equals 4.</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I identify the two numbers in the problem, which are 1 and 3.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum to determine the result of 1 plus 3.<br></think><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>I recognize that the problem asks for the sum of 1 and 3.<br><br>I will add the two numbers together to find the result.<br><br>Finally, I will state the answer clearly.<br></think><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**  <br>   We have the numbers 1 and 3.<br><br>2. **Add the numbers together:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**  <br>\[<br>\boxed{4}<br>\]</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>I recognize that the problem asks for the sum of 1 and 3.<br><br>I will add the two numbers together to find the result.<br><br>Finally, I will state the answer clearly.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**  <br>   We have the numbers 1 and 3.<br><br>2. **Add the numbers together:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**  <br>\[<br>\boxed{4}<br>\]</strong>



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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.33s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.33s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.33s/it]


    2026-05-13 17:00:15,247 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 17:00:15] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:07,  2.28s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:07,  2.28s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:17,  1.42s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:17,  1.42s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:52,  1.03it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:52,  1.03it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:25,  2.01it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:25,  2.01it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:19,  2.65it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:19,  2.65it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.38it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.38it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.23it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.23it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.23it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  5.93it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  5.93it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:07,  5.93it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.47it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.47it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.47it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.15it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.15it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.15it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.09it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.09it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.09it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.09it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.35it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.35it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.35it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.35it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.35it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.76it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.76it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.76it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.76it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.76it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.76it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 19.76it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.59it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.59it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.59it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.59it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.59it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.59it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.59it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.11it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.11it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.11it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.11it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.11it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.11it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.11it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.11it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.11it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 46.33it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 46.33it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 46.33it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 46.33it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 46.33it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 46.33it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 46.33it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:08<00:00, 46.33it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:08<00:00, 46.33it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:08<00:00, 55.29it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:08<00:00, 55.29it/s]

    Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:08<00:00, 55.29it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:08<00:00, 55.29it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:08<00:00, 55.29it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:08<00:00, 55.29it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:08<00:00, 55.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=42.26 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.23 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.23 GB):   3%|▎         | 2/58 [00:01<00:33,  1.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.23 GB):   3%|▎         | 2/58 [00:01<00:33,  1.65it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.23 GB):   5%|▌         | 3/58 [00:01<00:31,  1.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.23 GB):   5%|▌         | 3/58 [00:01<00:31,  1.76it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.23 GB):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.23 GB):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.23 GB):   9%|▊         | 5/58 [00:02<00:26,  2.01it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.23 GB):   9%|▊         | 5/58 [00:02<00:26,  2.01it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=42.23 GB):  10%|█         | 6/58 [00:03<00:23,  2.18it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.22 GB):  10%|█         | 6/58 [00:03<00:23,  2.18it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.22 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.23 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.33it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=42.23 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.23 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.53it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.23 GB):  16%|█▌        | 9/58 [00:04<00:17,  2.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.23 GB):  16%|█▌        | 9/58 [00:04<00:17,  2.73it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=42.23 GB):  17%|█▋        | 10/58 [00:04<00:16,  2.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.22 GB):  17%|█▋        | 10/58 [00:04<00:16,  2.97it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.22 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.22 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.29it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=42.22 GB):  21%|██        | 12/58 [00:04<00:12,  3.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.22 GB):  21%|██        | 12/58 [00:04<00:12,  3.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.22 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.22 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.10it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=42.22 GB):  24%|██▍       | 14/58 [00:05<00:09,  4.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.22 GB):  24%|██▍       | 14/58 [00:05<00:09,  4.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.22 GB):  26%|██▌       | 15/58 [00:05<00:08,  5.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.21 GB):  26%|██▌       | 15/58 [00:05<00:08,  5.34it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=42.21 GB):  28%|██▊       | 16/58 [00:05<00:06,  6.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.21 GB):  28%|██▊       | 16/58 [00:05<00:06,  6.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.20 GB):  28%|██▊       | 16/58 [00:05<00:06,  6.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.20 GB):  31%|███       | 18/58 [00:05<00:05,  7.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.20 GB):  31%|███       | 18/58 [00:05<00:05,  7.42it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=42.20 GB):  31%|███       | 18/58 [00:05<00:05,  7.42it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.20 GB):  34%|███▍      | 20/58 [00:05<00:04,  9.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.19 GB):  34%|███▍      | 20/58 [00:05<00:04,  9.36it/s]Capturing num tokens (num_tokens=960 avail_mem=42.18 GB):  34%|███▍      | 20/58 [00:05<00:04,  9.36it/s] Capturing num tokens (num_tokens=896 avail_mem=42.18 GB):  34%|███▍      | 20/58 [00:05<00:04,  9.36it/s]Capturing num tokens (num_tokens=896 avail_mem=42.18 GB):  40%|███▉      | 23/58 [00:05<00:02, 13.25it/s]Capturing num tokens (num_tokens=832 avail_mem=42.18 GB):  40%|███▉      | 23/58 [00:05<00:02, 13.25it/s]Capturing num tokens (num_tokens=768 avail_mem=42.17 GB):  40%|███▉      | 23/58 [00:05<00:02, 13.25it/s]

    Capturing num tokens (num_tokens=704 avail_mem=42.17 GB):  40%|███▉      | 23/58 [00:05<00:02, 13.25it/s]Capturing num tokens (num_tokens=704 avail_mem=42.17 GB):  45%|████▍     | 26/58 [00:05<00:01, 16.81it/s]Capturing num tokens (num_tokens=640 avail_mem=42.16 GB):  45%|████▍     | 26/58 [00:05<00:01, 16.81it/s]Capturing num tokens (num_tokens=576 avail_mem=42.16 GB):  45%|████▍     | 26/58 [00:05<00:01, 16.81it/s]Capturing num tokens (num_tokens=512 avail_mem=42.16 GB):  45%|████▍     | 26/58 [00:05<00:01, 16.81it/s]Capturing num tokens (num_tokens=480 avail_mem=41.18 GB):  45%|████▍     | 26/58 [00:05<00:01, 16.81it/s]

    Capturing num tokens (num_tokens=480 avail_mem=41.18 GB):  52%|█████▏    | 30/58 [00:06<00:01, 17.67it/s]Capturing num tokens (num_tokens=448 avail_mem=41.18 GB):  52%|█████▏    | 30/58 [00:06<00:01, 17.67it/s]Capturing num tokens (num_tokens=416 avail_mem=41.16 GB):  52%|█████▏    | 30/58 [00:06<00:01, 17.67it/s]

    Capturing num tokens (num_tokens=416 avail_mem=41.16 GB):  55%|█████▌    | 32/58 [00:06<00:01, 14.00it/s]Capturing num tokens (num_tokens=384 avail_mem=42.09 GB):  55%|█████▌    | 32/58 [00:06<00:01, 14.00it/s]Capturing num tokens (num_tokens=352 avail_mem=42.00 GB):  55%|█████▌    | 32/58 [00:06<00:01, 14.00it/s]

    Capturing num tokens (num_tokens=352 avail_mem=42.00 GB):  59%|█████▊    | 34/58 [00:06<00:02, 11.84it/s]Capturing num tokens (num_tokens=320 avail_mem=40.61 GB):  59%|█████▊    | 34/58 [00:06<00:02, 11.84it/s]Capturing num tokens (num_tokens=288 avail_mem=40.62 GB):  59%|█████▊    | 34/58 [00:06<00:02, 11.84it/s]

    Capturing num tokens (num_tokens=288 avail_mem=40.62 GB):  62%|██████▏   | 36/58 [00:06<00:02, 10.81it/s]Capturing num tokens (num_tokens=256 avail_mem=41.43 GB):  62%|██████▏   | 36/58 [00:06<00:02, 10.81it/s]Capturing num tokens (num_tokens=240 avail_mem=40.66 GB):  62%|██████▏   | 36/58 [00:06<00:02, 10.81it/s]Capturing num tokens (num_tokens=240 avail_mem=40.66 GB):  66%|██████▌   | 38/58 [00:06<00:01, 10.65it/s]Capturing num tokens (num_tokens=224 avail_mem=40.66 GB):  66%|██████▌   | 38/58 [00:06<00:01, 10.65it/s]

    Capturing num tokens (num_tokens=208 avail_mem=40.65 GB):  66%|██████▌   | 38/58 [00:07<00:01, 10.65it/s]Capturing num tokens (num_tokens=208 avail_mem=40.65 GB):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]Capturing num tokens (num_tokens=192 avail_mem=41.42 GB):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]Capturing num tokens (num_tokens=176 avail_mem=40.70 GB):  69%|██████▉   | 40/58 [00:07<00:01, 10.31it/s]

    Capturing num tokens (num_tokens=176 avail_mem=40.70 GB):  72%|███████▏  | 42/58 [00:07<00:01,  9.87it/s]Capturing num tokens (num_tokens=160 avail_mem=40.70 GB):  72%|███████▏  | 42/58 [00:07<00:01,  9.87it/s]Capturing num tokens (num_tokens=144 avail_mem=41.40 GB):  72%|███████▏  | 42/58 [00:07<00:01,  9.87it/s]Capturing num tokens (num_tokens=144 avail_mem=41.40 GB):  76%|███████▌  | 44/58 [00:07<00:01, 10.39it/s]Capturing num tokens (num_tokens=128 avail_mem=40.75 GB):  76%|███████▌  | 44/58 [00:07<00:01, 10.39it/s]

    Capturing num tokens (num_tokens=112 avail_mem=40.75 GB):  76%|███████▌  | 44/58 [00:07<00:01, 10.39it/s]Capturing num tokens (num_tokens=112 avail_mem=40.75 GB):  79%|███████▉  | 46/58 [00:07<00:01, 10.36it/s]Capturing num tokens (num_tokens=96 avail_mem=41.40 GB):  79%|███████▉  | 46/58 [00:07<00:01, 10.36it/s] Capturing num tokens (num_tokens=80 avail_mem=40.79 GB):  79%|███████▉  | 46/58 [00:07<00:01, 10.36it/s]

    Capturing num tokens (num_tokens=80 avail_mem=40.79 GB):  83%|████████▎ | 48/58 [00:07<00:00, 10.68it/s]Capturing num tokens (num_tokens=64 avail_mem=40.79 GB):  83%|████████▎ | 48/58 [00:07<00:00, 10.68it/s]Capturing num tokens (num_tokens=48 avail_mem=41.39 GB):  83%|████████▎ | 48/58 [00:08<00:00, 10.68it/s]Capturing num tokens (num_tokens=48 avail_mem=41.39 GB):  86%|████████▌ | 50/58 [00:08<00:00, 11.07it/s]Capturing num tokens (num_tokens=32 avail_mem=40.84 GB):  86%|████████▌ | 50/58 [00:08<00:00, 11.07it/s]

    Capturing num tokens (num_tokens=28 avail_mem=40.84 GB):  86%|████████▌ | 50/58 [00:08<00:00, 11.07it/s]Capturing num tokens (num_tokens=28 avail_mem=40.84 GB):  90%|████████▉ | 52/58 [00:08<00:00, 10.98it/s]Capturing num tokens (num_tokens=24 avail_mem=41.38 GB):  90%|████████▉ | 52/58 [00:08<00:00, 10.98it/s]Capturing num tokens (num_tokens=20 avail_mem=40.88 GB):  90%|████████▉ | 52/58 [00:08<00:00, 10.98it/s]

    Capturing num tokens (num_tokens=20 avail_mem=40.88 GB):  93%|█████████▎| 54/58 [00:08<00:00, 11.26it/s]Capturing num tokens (num_tokens=16 avail_mem=40.88 GB):  93%|█████████▎| 54/58 [00:08<00:00, 11.26it/s]Capturing num tokens (num_tokens=12 avail_mem=41.37 GB):  93%|█████████▎| 54/58 [00:08<00:00, 11.26it/s]Capturing num tokens (num_tokens=12 avail_mem=41.37 GB):  97%|█████████▋| 56/58 [00:08<00:00, 11.80it/s]Capturing num tokens (num_tokens=8 avail_mem=39.87 GB):  97%|█████████▋| 56/58 [00:08<00:00, 11.80it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=26.34 GB):  97%|█████████▋| 56/58 [00:08<00:00, 11.80it/s]Capturing num tokens (num_tokens=4 avail_mem=26.34 GB): 100%|██████████| 58/58 [00:08<00:00, 12.42it/s]Capturing num tokens (num_tokens=4 avail_mem=26.34 GB): 100%|██████████| 58/58 [00:08<00:00,  6.62it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I need to identify the two numbers in the problem, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Adding 1 and 3 gives a sum of 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I need to identify the two numbers in the problem, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Adding 1 and 3 gives a sum of 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
