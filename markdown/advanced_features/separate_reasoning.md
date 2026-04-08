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


    [2026-04-08 23:15:39] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 23:15:39] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 23:15:39] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 23:15:39] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:00<00:00,  1.35it/s]

    Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.01s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.03it/s]


    2026-04-08 23:15:42,664 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 23:15:42] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:58,  3.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:58,  3.12s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:29,  1.60s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:29,  1.60s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:52,  1.04it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:52,  1.04it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:35,  1.54it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:35,  1.54it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.12it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.12it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.75it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.75it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.49it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.49it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.28it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.28it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.17it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.17it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.17it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:07,  6.35it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:07,  6.35it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:07,  6.08it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:07,  6.08it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:07,  6.00it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:07,  6.00it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:07,  6.06it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:07,  6.06it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:06,  6.30it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:06,  6.30it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:06,  6.63it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:06,  6.63it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:05,  7.11it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:05,  7.11it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:05,  7.71it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:05,  7.71it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:06<00:05,  7.71it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:04,  8.91it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:04,  8.91it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:04,  8.91it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:06<00:03, 10.58it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:06<00:03, 10.58it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:06<00:03, 10.58it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:06<00:02, 12.22it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:06<00:02, 12.22it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:06<00:02, 12.22it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:06<00:02, 13.58it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:06<00:02, 13.58it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:06<00:02, 13.58it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:06<00:02, 14.67it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:06<00:02, 14.67it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:06<00:02, 14.67it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:06<00:01, 15.88it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:06<00:01, 15.88it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:06<00:01, 15.88it/s]

    Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:06<00:01, 16.28it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:06<00:01, 16.28it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:06<00:01, 16.28it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:06<00:01, 16.83it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:06<00:01, 16.83it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:06<00:01, 16.83it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:01, 17.24it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:01, 17.24it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:01, 17.24it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:01, 17.24it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:07<00:01, 18.65it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:07<00:01, 18.65it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:07<00:01, 18.65it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:07<00:00, 18.70it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:07<00:00, 18.70it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:07<00:00, 18.70it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:07<00:00, 18.70it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 20.38it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 20.38it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 20.38it/s]

    Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 20.38it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:07<00:00, 16.37it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:07<00:00, 16.37it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:07<00:00, 16.37it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:07<00:00, 16.37it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:07<00:00, 18.73it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:07<00:00, 18.73it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 18.73it/s]

    Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 18.73it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:07<00:00, 18.73it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:07<00:00, 22.21it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:07<00:00, 22.21it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:07<00:00, 22.21it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:07<00:00, 22.21it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:07<00:00, 22.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00, 25.99it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.24it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=93.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=93.07 GB):   2%|▏         | 1/58 [00:00<00:34,  1.65it/s]Capturing num tokens (num_tokens=7680 avail_mem=93.04 GB):   2%|▏         | 1/58 [00:00<00:34,  1.65it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=93.04 GB):   3%|▎         | 2/58 [00:01<00:29,  1.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=93.04 GB):   3%|▎         | 2/58 [00:01<00:29,  1.87it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=93.04 GB):   5%|▌         | 3/58 [00:01<00:25,  2.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=93.04 GB):   5%|▌         | 3/58 [00:01<00:25,  2.17it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=93.04 GB):   7%|▋         | 4/58 [00:01<00:21,  2.49it/s]Capturing num tokens (num_tokens=6144 avail_mem=93.04 GB):   7%|▋         | 4/58 [00:01<00:21,  2.49it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=93.04 GB):   9%|▊         | 5/58 [00:02<00:18,  2.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=93.04 GB):   9%|▊         | 5/58 [00:02<00:18,  2.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=93.04 GB):  10%|█         | 6/58 [00:02<00:15,  3.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=93.04 GB):  10%|█         | 6/58 [00:02<00:15,  3.39it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=93.04 GB):  12%|█▏        | 7/58 [00:02<00:12,  3.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=93.05 GB):  12%|█▏        | 7/58 [00:02<00:12,  3.96it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=93.05 GB):  14%|█▍        | 8/58 [00:02<00:12,  3.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=92.02 GB):  14%|█▍        | 8/58 [00:02<00:12,  3.89it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=92.02 GB):  16%|█▌        | 9/58 [00:02<00:13,  3.74it/s]Capturing num tokens (num_tokens=3840 avail_mem=93.03 GB):  16%|█▌        | 9/58 [00:02<00:13,  3.74it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=93.03 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=91.87 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.43it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=91.87 GB):  19%|█▉        | 11/58 [00:03<00:13,  3.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=91.69 GB):  19%|█▉        | 11/58 [00:03<00:13,  3.43it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=91.69 GB):  21%|██        | 12/58 [00:03<00:12,  3.72it/s]Capturing num tokens (num_tokens=3072 avail_mem=92.05 GB):  21%|██        | 12/58 [00:03<00:12,  3.72it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=92.05 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.83it/s]Capturing num tokens (num_tokens=2816 avail_mem=90.96 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.83it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=90.96 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=91.70 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.01it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=91.70 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=89.62 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.21it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=89.62 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=89.17 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=89.17 GB):  29%|██▉       | 17/58 [00:04<00:08,  4.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=88.51 GB):  29%|██▉       | 17/58 [00:04<00:08,  4.90it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=88.51 GB):  31%|███       | 18/58 [00:05<00:07,  5.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=88.50 GB):  31%|███       | 18/58 [00:05<00:07,  5.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=88.50 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.66it/s]Capturing num tokens (num_tokens=1280 avail_mem=89.17 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.66it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=89.17 GB):  34%|███▍      | 20/58 [00:05<00:06,  6.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=88.56 GB):  34%|███▍      | 20/58 [00:05<00:06,  6.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=88.56 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.52it/s]Capturing num tokens (num_tokens=960 avail_mem=89.17 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.52it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=89.17 GB):  38%|███▊      | 22/58 [00:05<00:06,  5.68it/s]Capturing num tokens (num_tokens=896 avail_mem=88.66 GB):  38%|███▊      | 22/58 [00:05<00:06,  5.68it/s]Capturing num tokens (num_tokens=832 avail_mem=89.16 GB):  38%|███▊      | 22/58 [00:05<00:06,  5.68it/s]

    Capturing num tokens (num_tokens=832 avail_mem=89.16 GB):  41%|████▏     | 24/58 [00:05<00:05,  6.13it/s]Capturing num tokens (num_tokens=768 avail_mem=88.67 GB):  41%|████▏     | 24/58 [00:05<00:05,  6.13it/s]Capturing num tokens (num_tokens=768 avail_mem=88.67 GB):  43%|████▎     | 25/58 [00:06<00:04,  6.78it/s]Capturing num tokens (num_tokens=704 avail_mem=89.16 GB):  43%|████▎     | 25/58 [00:06<00:04,  6.78it/s]Capturing num tokens (num_tokens=640 avail_mem=88.72 GB):  43%|████▎     | 25/58 [00:06<00:04,  6.78it/s]

    Capturing num tokens (num_tokens=640 avail_mem=88.72 GB):  47%|████▋     | 27/58 [00:06<00:03,  7.92it/s]Capturing num tokens (num_tokens=576 avail_mem=89.19 GB):  47%|████▋     | 27/58 [00:06<00:03,  7.92it/s]Capturing num tokens (num_tokens=512 avail_mem=89.10 GB):  47%|████▋     | 27/58 [00:06<00:03,  7.92it/s]Capturing num tokens (num_tokens=512 avail_mem=89.10 GB):  50%|█████     | 29/58 [00:06<00:03,  8.87it/s]Capturing num tokens (num_tokens=480 avail_mem=88.73 GB):  50%|█████     | 29/58 [00:06<00:03,  8.87it/s]

    Capturing num tokens (num_tokens=448 avail_mem=89.15 GB):  50%|█████     | 29/58 [00:06<00:03,  8.87it/s]Capturing num tokens (num_tokens=448 avail_mem=89.15 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.60it/s]Capturing num tokens (num_tokens=416 avail_mem=88.76 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.60it/s]Capturing num tokens (num_tokens=384 avail_mem=89.14 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.60it/s]

    Capturing num tokens (num_tokens=384 avail_mem=89.14 GB):  57%|█████▋    | 33/58 [00:06<00:02, 10.13it/s]Capturing num tokens (num_tokens=352 avail_mem=88.77 GB):  57%|█████▋    | 33/58 [00:06<00:02, 10.13it/s]Capturing num tokens (num_tokens=320 avail_mem=89.13 GB):  57%|█████▋    | 33/58 [00:06<00:02, 10.13it/s]Capturing num tokens (num_tokens=320 avail_mem=89.13 GB):  60%|██████    | 35/58 [00:06<00:02, 10.87it/s]Capturing num tokens (num_tokens=288 avail_mem=88.80 GB):  60%|██████    | 35/58 [00:06<00:02, 10.87it/s]

    Capturing num tokens (num_tokens=256 avail_mem=89.12 GB):  60%|██████    | 35/58 [00:06<00:02, 10.87it/s]Capturing num tokens (num_tokens=256 avail_mem=89.12 GB):  64%|██████▍   | 37/58 [00:07<00:01, 11.57it/s]Capturing num tokens (num_tokens=240 avail_mem=88.81 GB):  64%|██████▍   | 37/58 [00:07<00:01, 11.57it/s]Capturing num tokens (num_tokens=224 avail_mem=89.11 GB):  64%|██████▍   | 37/58 [00:07<00:01, 11.57it/s]

    Capturing num tokens (num_tokens=224 avail_mem=89.11 GB):  67%|██████▋   | 39/58 [00:07<00:01, 12.59it/s]Capturing num tokens (num_tokens=208 avail_mem=89.15 GB):  67%|██████▋   | 39/58 [00:07<00:01, 12.59it/s]Capturing num tokens (num_tokens=192 avail_mem=89.11 GB):  67%|██████▋   | 39/58 [00:07<00:01, 12.59it/s]Capturing num tokens (num_tokens=192 avail_mem=89.11 GB):  71%|███████   | 41/58 [00:07<00:01, 13.64it/s]Capturing num tokens (num_tokens=176 avail_mem=89.11 GB):  71%|███████   | 41/58 [00:07<00:01, 13.64it/s]Capturing num tokens (num_tokens=160 avail_mem=89.10 GB):  71%|███████   | 41/58 [00:07<00:01, 13.64it/s]

    Capturing num tokens (num_tokens=160 avail_mem=89.10 GB):  74%|███████▍  | 43/58 [00:07<00:01, 14.88it/s]Capturing num tokens (num_tokens=144 avail_mem=89.10 GB):  74%|███████▍  | 43/58 [00:07<00:01, 14.88it/s]Capturing num tokens (num_tokens=128 avail_mem=88.91 GB):  74%|███████▍  | 43/58 [00:07<00:01, 14.88it/s]Capturing num tokens (num_tokens=128 avail_mem=88.91 GB):  78%|███████▊  | 45/58 [00:07<00:00, 15.95it/s]Capturing num tokens (num_tokens=112 avail_mem=89.10 GB):  78%|███████▊  | 45/58 [00:07<00:00, 15.95it/s]Capturing num tokens (num_tokens=96 avail_mem=89.10 GB):  78%|███████▊  | 45/58 [00:07<00:00, 15.95it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=89.10 GB):  81%|████████  | 47/58 [00:07<00:00, 16.26it/s]Capturing num tokens (num_tokens=80 avail_mem=89.11 GB):  81%|████████  | 47/58 [00:07<00:00, 16.26it/s]Capturing num tokens (num_tokens=64 avail_mem=88.97 GB):  81%|████████  | 47/58 [00:07<00:00, 16.26it/s]Capturing num tokens (num_tokens=48 avail_mem=89.08 GB):  81%|████████  | 47/58 [00:07<00:00, 16.26it/s]Capturing num tokens (num_tokens=48 avail_mem=89.08 GB):  86%|████████▌ | 50/58 [00:07<00:00, 18.35it/s]Capturing num tokens (num_tokens=32 avail_mem=89.07 GB):  86%|████████▌ | 50/58 [00:07<00:00, 18.35it/s]Capturing num tokens (num_tokens=28 avail_mem=89.07 GB):  86%|████████▌ | 50/58 [00:07<00:00, 18.35it/s]

    Capturing num tokens (num_tokens=24 avail_mem=88.97 GB):  86%|████████▌ | 50/58 [00:07<00:00, 18.35it/s]Capturing num tokens (num_tokens=24 avail_mem=88.97 GB):  91%|█████████▏| 53/58 [00:07<00:00, 19.59it/s]Capturing num tokens (num_tokens=20 avail_mem=88.97 GB):  91%|█████████▏| 53/58 [00:07<00:00, 19.59it/s]Capturing num tokens (num_tokens=16 avail_mem=88.99 GB):  91%|█████████▏| 53/58 [00:07<00:00, 19.59it/s]Capturing num tokens (num_tokens=12 avail_mem=89.04 GB):  91%|█████████▏| 53/58 [00:07<00:00, 19.59it/s]Capturing num tokens (num_tokens=12 avail_mem=89.04 GB):  97%|█████████▋| 56/58 [00:08<00:00, 20.84it/s]Capturing num tokens (num_tokens=8 avail_mem=89.03 GB):  97%|█████████▋| 56/58 [00:08<00:00, 20.84it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=89.02 GB):  97%|█████████▋| 56/58 [00:08<00:00, 20.84it/s]Capturing num tokens (num_tokens=4 avail_mem=89.02 GB): 100%|██████████| 58/58 [00:08<00:00,  7.15it/s]


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



<strong style='color: #00008B;'>I recognize that the problem is asking for the sum of 1 and 3.<br><br>I start by identifying the two numbers involved, which are 1 and 3.<br><br>Next, I add these two numbers together to find the total.<br><br>Finally, I conclude that the result of 1 plus 3 is 4.<br></strong>



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



<strong style='color: #00008B;'>To solve the problem 1 + 3, I start by identifying the two numbers involved.<br><br>Next, I add the first number, which is 1, to the second number, which is 3.<br><br>Finally, the sum of 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I arrive at the conclusion that the sum of 1 and 3 is 4.</strong>



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



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved.<br><br>Next, I perform the addition by combining these numbers.<br><br>Finally, I arrive at the sum, which is 4.<br></think><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**  <br>1. **Identify the numbers to add:**  <br>   - First number: \(1\)<br>   - Second number: \(3\)<br><br>2. **Perform the addition:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**  <br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I'll add the two numbers together.<br><br>Finally, I calculate that 1 plus 3 equals 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I'll add the two numbers together.<br><br>Finally, I calculate that 1 plus 3 equals 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:00<00:00,  1.35it/s]

    Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.01s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.04it/s]


    2026-04-08 23:16:32,710 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 23:16:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:55,  3.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:55,  3.07s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:23,  1.49s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:23,  1.49s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:49,  1.12it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:49,  1.12it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:23,  2.24it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:23,  2.24it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.89it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.89it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.63it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.63it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.43it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.43it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.31it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.31it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.31it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  7.01it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  7.01it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  7.01it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.52it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.52it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.52it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04, 10.11it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04, 10.11it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04, 10.11it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 12.05it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 12.05it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 12.05it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 12.05it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.34it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.34it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.34it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.34it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.34it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.45it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.45it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.45it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.45it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.45it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.45it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 26.97it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 26.97it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 26.97it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 26.97it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 26.97it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 26.97it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 26.97it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 34.41it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 34.41it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 34.41it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 34.41it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 34.41it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 34.41it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 34.41it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 38.75it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 38.75it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 38.75it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 38.75it/s]

    Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 38.75it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 38.75it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 30.77it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 30.77it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 30.77it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 30.77it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 30.77it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:06<00:00, 29.59it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:06<00:00, 29.59it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:06<00:00, 29.59it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:06<00:00, 29.59it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:06<00:00, 29.59it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 30.19it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 30.19it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 30.19it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:06<00:00, 30.19it/s] 

    Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:06<00:00, 30.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00, 30.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=85.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=85.22 GB):   2%|▏         | 1/58 [00:00<00:55,  1.03it/s]Capturing num tokens (num_tokens=7680 avail_mem=85.19 GB):   2%|▏         | 1/58 [00:00<00:55,  1.03it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=85.19 GB):   3%|▎         | 2/58 [00:01<00:50,  1.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=85.19 GB):   3%|▎         | 2/58 [00:01<00:50,  1.12it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=85.19 GB):   5%|▌         | 3/58 [00:02<00:45,  1.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=85.19 GB):   5%|▌         | 3/58 [00:02<00:45,  1.20it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=85.19 GB):   7%|▋         | 4/58 [00:03<00:41,  1.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=85.19 GB):   7%|▋         | 4/58 [00:03<00:41,  1.29it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=85.19 GB):   9%|▊         | 5/58 [00:03<00:38,  1.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=85.20 GB):   9%|▊         | 5/58 [00:03<00:38,  1.37it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=85.20 GB):  10%|█         | 6/58 [00:04<00:34,  1.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=85.20 GB):  10%|█         | 6/58 [00:04<00:34,  1.49it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=85.20 GB):  12%|█▏        | 7/58 [00:04<00:31,  1.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=85.20 GB):  12%|█▏        | 7/58 [00:04<00:31,  1.62it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=85.20 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=85.21 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.81it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=85.21 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=85.21 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.00it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=85.21 GB):  17%|█▋        | 10/58 [00:06<00:21,  2.21it/s]Capturing num tokens (num_tokens=3584 avail_mem=85.21 GB):  17%|█▋        | 10/58 [00:06<00:21,  2.21it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=85.21 GB):  19%|█▉        | 11/58 [00:06<00:18,  2.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=85.20 GB):  19%|█▉        | 11/58 [00:06<00:18,  2.48it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=85.20 GB):  21%|██        | 12/58 [00:06<00:16,  2.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=85.20 GB):  21%|██        | 12/58 [00:06<00:16,  2.79it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=85.20 GB):  22%|██▏       | 13/58 [00:06<00:14,  3.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=85.20 GB):  22%|██▏       | 13/58 [00:06<00:14,  3.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=85.20 GB):  24%|██▍       | 14/58 [00:07<00:11,  3.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=85.20 GB):  24%|██▍       | 14/58 [00:07<00:11,  3.69it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=85.20 GB):  26%|██▌       | 15/58 [00:07<00:09,  4.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=85.20 GB):  26%|██▌       | 15/58 [00:07<00:09,  4.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=85.20 GB):  28%|██▊       | 16/58 [00:07<00:08,  5.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=85.20 GB):  28%|██▊       | 16/58 [00:07<00:08,  5.19it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=85.20 GB):  29%|██▉       | 17/58 [00:07<00:07,  5.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=84.01 GB):  29%|██▉       | 17/58 [00:07<00:07,  5.41it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=84.01 GB):  31%|███       | 18/58 [00:07<00:07,  5.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=84.00 GB):  31%|███       | 18/58 [00:07<00:07,  5.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=84.00 GB):  33%|███▎      | 19/58 [00:07<00:07,  5.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=85.16 GB):  33%|███▎      | 19/58 [00:07<00:07,  5.51it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=85.16 GB):  34%|███▍      | 20/58 [00:07<00:06,  5.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=84.18 GB):  34%|███▍      | 20/58 [00:07<00:06,  5.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=84.18 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.21it/s]Capturing num tokens (num_tokens=960 avail_mem=84.17 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.21it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=84.17 GB):  38%|███▊      | 22/58 [00:08<00:07,  4.86it/s]Capturing num tokens (num_tokens=896 avail_mem=84.12 GB):  38%|███▊      | 22/58 [00:08<00:07,  4.86it/s]

    Capturing num tokens (num_tokens=896 avail_mem=84.12 GB):  40%|███▉      | 23/58 [00:08<00:07,  4.79it/s]Capturing num tokens (num_tokens=832 avail_mem=83.20 GB):  40%|███▉      | 23/58 [00:08<00:07,  4.79it/s]

    Capturing num tokens (num_tokens=832 avail_mem=83.20 GB):  41%|████▏     | 24/58 [00:08<00:07,  4.67it/s]Capturing num tokens (num_tokens=768 avail_mem=83.19 GB):  41%|████▏     | 24/58 [00:08<00:07,  4.67it/s]Capturing num tokens (num_tokens=768 avail_mem=83.19 GB):  43%|████▎     | 25/58 [00:09<00:06,  4.95it/s]Capturing num tokens (num_tokens=704 avail_mem=85.11 GB):  43%|████▎     | 25/58 [00:09<00:06,  4.95it/s]

    Capturing num tokens (num_tokens=704 avail_mem=85.11 GB):  45%|████▍     | 26/58 [00:09<00:06,  5.25it/s]Capturing num tokens (num_tokens=640 avail_mem=84.26 GB):  45%|████▍     | 26/58 [00:09<00:06,  5.25it/s]

    Capturing num tokens (num_tokens=640 avail_mem=84.26 GB):  47%|████▋     | 27/58 [00:09<00:06,  5.12it/s]Capturing num tokens (num_tokens=576 avail_mem=84.30 GB):  47%|████▋     | 27/58 [00:09<00:06,  5.12it/s]Capturing num tokens (num_tokens=576 avail_mem=84.30 GB):  48%|████▊     | 28/58 [00:09<00:05,  5.25it/s]Capturing num tokens (num_tokens=512 avail_mem=83.50 GB):  48%|████▊     | 28/58 [00:09<00:05,  5.25it/s]

    Capturing num tokens (num_tokens=512 avail_mem=83.50 GB):  50%|█████     | 29/58 [00:09<00:05,  5.28it/s]Capturing num tokens (num_tokens=480 avail_mem=84.32 GB):  50%|█████     | 29/58 [00:09<00:05,  5.28it/s]Capturing num tokens (num_tokens=480 avail_mem=84.32 GB):  52%|█████▏    | 30/58 [00:09<00:05,  5.58it/s]Capturing num tokens (num_tokens=448 avail_mem=85.11 GB):  52%|█████▏    | 30/58 [00:09<00:05,  5.58it/s]

    Capturing num tokens (num_tokens=448 avail_mem=85.11 GB):  53%|█████▎    | 31/58 [00:10<00:04,  5.84it/s]Capturing num tokens (num_tokens=416 avail_mem=83.62 GB):  53%|█████▎    | 31/58 [00:10<00:04,  5.84it/s]Capturing num tokens (num_tokens=416 avail_mem=83.62 GB):  55%|█████▌    | 32/58 [00:10<00:04,  5.56it/s]Capturing num tokens (num_tokens=384 avail_mem=83.61 GB):  55%|█████▌    | 32/58 [00:10<00:04,  5.56it/s]

    Capturing num tokens (num_tokens=384 avail_mem=83.61 GB):  57%|█████▋    | 33/58 [00:10<00:04,  5.88it/s]Capturing num tokens (num_tokens=352 avail_mem=84.45 GB):  57%|█████▋    | 33/58 [00:10<00:04,  5.88it/s]Capturing num tokens (num_tokens=352 avail_mem=84.45 GB):  59%|█████▊    | 34/58 [00:10<00:04,  5.79it/s]Capturing num tokens (num_tokens=320 avail_mem=83.73 GB):  59%|█████▊    | 34/58 [00:10<00:04,  5.79it/s]

    Capturing num tokens (num_tokens=320 avail_mem=83.73 GB):  60%|██████    | 35/58 [00:10<00:03,  5.89it/s]Capturing num tokens (num_tokens=288 avail_mem=84.39 GB):  60%|██████    | 35/58 [00:10<00:03,  5.89it/s]Capturing num tokens (num_tokens=288 avail_mem=84.39 GB):  62%|██████▏   | 36/58 [00:10<00:03,  5.84it/s]Capturing num tokens (num_tokens=256 avail_mem=83.79 GB):  62%|██████▏   | 36/58 [00:10<00:03,  5.84it/s]

    Capturing num tokens (num_tokens=256 avail_mem=83.79 GB):  64%|██████▍   | 37/58 [00:11<00:03,  6.11it/s]Capturing num tokens (num_tokens=240 avail_mem=85.10 GB):  64%|██████▍   | 37/58 [00:11<00:03,  6.11it/s]Capturing num tokens (num_tokens=240 avail_mem=85.10 GB):  66%|██████▌   | 38/58 [00:11<00:03,  6.49it/s]Capturing num tokens (num_tokens=224 avail_mem=83.92 GB):  66%|██████▌   | 38/58 [00:11<00:03,  6.49it/s]

    Capturing num tokens (num_tokens=224 avail_mem=83.92 GB):  67%|██████▋   | 39/58 [00:11<00:03,  6.24it/s]Capturing num tokens (num_tokens=208 avail_mem=84.44 GB):  67%|██████▋   | 39/58 [00:11<00:03,  6.24it/s]Capturing num tokens (num_tokens=208 avail_mem=84.44 GB):  69%|██████▉   | 40/58 [00:11<00:02,  6.70it/s]Capturing num tokens (num_tokens=192 avail_mem=84.60 GB):  69%|██████▉   | 40/58 [00:11<00:02,  6.70it/s]

    Capturing num tokens (num_tokens=192 avail_mem=84.60 GB):  71%|███████   | 41/58 [00:11<00:02,  6.92it/s]Capturing num tokens (num_tokens=176 avail_mem=84.49 GB):  71%|███████   | 41/58 [00:11<00:02,  6.92it/s]Capturing num tokens (num_tokens=176 avail_mem=84.49 GB):  72%|███████▏  | 42/58 [00:11<00:02,  7.17it/s]Capturing num tokens (num_tokens=160 avail_mem=84.02 GB):  72%|███████▏  | 42/58 [00:11<00:02,  7.17it/s]

    Capturing num tokens (num_tokens=160 avail_mem=84.02 GB):  74%|███████▍  | 43/58 [00:11<00:02,  6.64it/s]Capturing num tokens (num_tokens=144 avail_mem=85.09 GB):  74%|███████▍  | 43/58 [00:11<00:02,  6.64it/s]Capturing num tokens (num_tokens=128 avail_mem=84.12 GB):  74%|███████▍  | 43/58 [00:12<00:02,  6.64it/s]

    Capturing num tokens (num_tokens=128 avail_mem=84.12 GB):  78%|███████▊  | 45/58 [00:12<00:02,  6.43it/s]Capturing num tokens (num_tokens=112 avail_mem=84.63 GB):  78%|███████▊  | 45/58 [00:12<00:02,  6.43it/s]Capturing num tokens (num_tokens=112 avail_mem=84.63 GB):  79%|███████▉  | 46/58 [00:12<00:01,  7.05it/s]Capturing num tokens (num_tokens=96 avail_mem=84.20 GB):  79%|███████▉  | 46/58 [00:12<00:01,  7.05it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=84.20 GB):  81%|████████  | 47/58 [00:12<00:01,  6.78it/s]Capturing num tokens (num_tokens=80 avail_mem=84.59 GB):  81%|████████  | 47/58 [00:12<00:01,  6.78it/s]Capturing num tokens (num_tokens=80 avail_mem=84.59 GB):  83%|████████▎ | 48/58 [00:12<00:01,  7.32it/s]Capturing num tokens (num_tokens=64 avail_mem=84.73 GB):  83%|████████▎ | 48/58 [00:12<00:01,  7.32it/s]

    Capturing num tokens (num_tokens=64 avail_mem=84.73 GB):  84%|████████▍ | 49/58 [00:12<00:01,  7.81it/s]Capturing num tokens (num_tokens=48 avail_mem=84.65 GB):  84%|████████▍ | 49/58 [00:12<00:01,  7.81it/s]Capturing num tokens (num_tokens=48 avail_mem=84.65 GB):  86%|████████▌ | 50/58 [00:12<00:01,  7.35it/s]Capturing num tokens (num_tokens=32 avail_mem=84.31 GB):  86%|████████▌ | 50/58 [00:12<00:01,  7.35it/s]

    Capturing num tokens (num_tokens=28 avail_mem=85.08 GB):  86%|████████▌ | 50/58 [00:12<00:01,  7.35it/s]Capturing num tokens (num_tokens=28 avail_mem=85.08 GB):  90%|████████▉ | 52/58 [00:13<00:00,  8.16it/s]Capturing num tokens (num_tokens=24 avail_mem=84.67 GB):  90%|████████▉ | 52/58 [00:13<00:00,  8.16it/s]

    Capturing num tokens (num_tokens=24 avail_mem=84.67 GB):  91%|█████████▏| 53/58 [00:13<00:00,  8.52it/s]Capturing num tokens (num_tokens=20 avail_mem=84.81 GB):  91%|█████████▏| 53/58 [00:13<00:00,  8.52it/s]Capturing num tokens (num_tokens=16 avail_mem=84.69 GB):  91%|█████████▏| 53/58 [00:13<00:00,  8.52it/s]

    Capturing num tokens (num_tokens=16 avail_mem=84.69 GB):  95%|█████████▍| 55/58 [00:13<00:00,  8.76it/s]Capturing num tokens (num_tokens=12 avail_mem=85.07 GB):  95%|█████████▍| 55/58 [00:13<00:00,  8.76it/s]Capturing num tokens (num_tokens=8 avail_mem=84.50 GB):  95%|█████████▍| 55/58 [00:13<00:00,  8.76it/s] Capturing num tokens (num_tokens=8 avail_mem=84.50 GB):  98%|█████████▊| 57/58 [00:13<00:00,  9.29it/s]Capturing num tokens (num_tokens=4 avail_mem=84.72 GB):  98%|█████████▊| 57/58 [00:13<00:00,  9.29it/s]

    Capturing num tokens (num_tokens=4 avail_mem=84.72 GB): 100%|██████████| 58/58 [00:13<00:00,  4.23it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I arrive at the answer, which is 4.<br></think><br><br>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I arrive at the answer, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
