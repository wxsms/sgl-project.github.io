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


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.36s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.33s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.33s/it]


    2026-04-27 17:03:59,715 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 17:03:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:57,  5.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:57,  5.21s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:07,  2.27s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:07,  2.27s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.15it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.62it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.83it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.83it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.57it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.57it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.42it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.42it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.42it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.14it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.14it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.14it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.60it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.60it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.60it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.26it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.26it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.26it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.24it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.24it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.24it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.24it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.58it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.58it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.58it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.58it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.58it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 20.10it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 20.10it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 20.10it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 20.10it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.10it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.10it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.10it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 29.05it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 29.05it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 29.05it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 29.05it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 29.05it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 29.05it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 29.05it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.70it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.70it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.70it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.70it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.70it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.70it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.70it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.70it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.70it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 47.16it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 47.16it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 47.16it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 47.16it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 47.16it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 47.16it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 47.16it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 47.16it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 47.16it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 55.77it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 55.77it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 55.77it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 55.77it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 55.77it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 55.77it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 55.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=27.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=27.44 GB):   2%|▏         | 1/58 [00:00<00:16,  3.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=27.41 GB):   2%|▏         | 1/58 [00:00<00:16,  3.47it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=27.41 GB):   3%|▎         | 2/58 [00:00<00:15,  3.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=27.40 GB):   3%|▎         | 2/58 [00:00<00:15,  3.64it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=27.40 GB):   5%|▌         | 3/58 [00:00<00:14,  3.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=27.40 GB):   5%|▌         | 3/58 [00:00<00:14,  3.88it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=27.40 GB):   7%|▋         | 4/58 [00:01<00:12,  4.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=27.40 GB):   7%|▋         | 4/58 [00:01<00:12,  4.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=27.40 GB):   9%|▊         | 5/58 [00:01<00:11,  4.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=27.40 GB):   9%|▊         | 5/58 [00:01<00:11,  4.44it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=27.40 GB):  10%|█         | 6/58 [00:01<00:10,  4.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=27.40 GB):  10%|█         | 6/58 [00:01<00:10,  4.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=27.40 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=27.40 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.17it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=27.40 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.46 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.46it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.46 GB):  16%|█▌        | 9/58 [00:02<00:11,  4.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.46 GB):  16%|█▌        | 9/58 [00:02<00:11,  4.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.46 GB):  17%|█▋        | 10/58 [00:02<00:10,  4.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=23.99 GB):  17%|█▋        | 10/58 [00:02<00:10,  4.60it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=23.99 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.26it/s]Capturing num tokens (num_tokens=3328 avail_mem=21.20 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.26it/s]Capturing num tokens (num_tokens=3328 avail_mem=21.20 GB):  21%|██        | 12/58 [00:02<00:07,  6.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=21.20 GB):  21%|██        | 12/58 [00:02<00:07,  6.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=21.20 GB):  21%|██        | 12/58 [00:02<00:07,  6.12it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=21.20 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=21.20 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=21.20 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=21.20 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=21.19 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.31it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=21.19 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=21.19 GB):  31%|███       | 18/58 [00:02<00:03, 10.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=21.19 GB):  31%|███       | 18/58 [00:02<00:03, 10.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=21.19 GB):  31%|███       | 18/58 [00:02<00:03, 10.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=21.19 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.60it/s]Capturing num tokens (num_tokens=1024 avail_mem=21.17 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.60it/s]

    Capturing num tokens (num_tokens=960 avail_mem=21.17 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.60it/s] Capturing num tokens (num_tokens=896 avail_mem=21.17 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.60it/s]Capturing num tokens (num_tokens=896 avail_mem=21.17 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.07it/s]Capturing num tokens (num_tokens=832 avail_mem=21.16 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.07it/s]Capturing num tokens (num_tokens=768 avail_mem=21.16 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.07it/s]Capturing num tokens (num_tokens=704 avail_mem=21.15 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.07it/s]Capturing num tokens (num_tokens=704 avail_mem=21.15 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Capturing num tokens (num_tokens=640 avail_mem=21.15 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]

    Capturing num tokens (num_tokens=576 avail_mem=21.15 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Capturing num tokens (num_tokens=512 avail_mem=21.14 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Capturing num tokens (num_tokens=480 avail_mem=21.14 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Capturing num tokens (num_tokens=480 avail_mem=21.14 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.31it/s]Capturing num tokens (num_tokens=448 avail_mem=21.14 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.31it/s]Capturing num tokens (num_tokens=416 avail_mem=21.14 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.31it/s]Capturing num tokens (num_tokens=384 avail_mem=21.13 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.31it/s]Capturing num tokens (num_tokens=352 avail_mem=21.13 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.31it/s]

    Capturing num tokens (num_tokens=352 avail_mem=21.13 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.56it/s]Capturing num tokens (num_tokens=320 avail_mem=21.12 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.56it/s]Capturing num tokens (num_tokens=288 avail_mem=21.13 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.56it/s]Capturing num tokens (num_tokens=256 avail_mem=21.12 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.56it/s]Capturing num tokens (num_tokens=240 avail_mem=21.12 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.56it/s]Capturing num tokens (num_tokens=240 avail_mem=21.12 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.03it/s]Capturing num tokens (num_tokens=224 avail_mem=21.12 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.03it/s]Capturing num tokens (num_tokens=208 avail_mem=21.11 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.03it/s]Capturing num tokens (num_tokens=192 avail_mem=21.11 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.03it/s]Capturing num tokens (num_tokens=176 avail_mem=21.10 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.03it/s]Capturing num tokens (num_tokens=160 avail_mem=21.10 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.03it/s]

    Capturing num tokens (num_tokens=160 avail_mem=21.10 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.32it/s]Capturing num tokens (num_tokens=144 avail_mem=21.10 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.32it/s]Capturing num tokens (num_tokens=128 avail_mem=21.10 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.32it/s]Capturing num tokens (num_tokens=112 avail_mem=21.10 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.32it/s]Capturing num tokens (num_tokens=96 avail_mem=21.09 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.32it/s] Capturing num tokens (num_tokens=80 avail_mem=21.08 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.32it/s]Capturing num tokens (num_tokens=80 avail_mem=21.08 GB):  83%|████████▎ | 48/58 [00:03<00:00, 34.84it/s]Capturing num tokens (num_tokens=64 avail_mem=21.08 GB):  83%|████████▎ | 48/58 [00:03<00:00, 34.84it/s]Capturing num tokens (num_tokens=48 avail_mem=21.08 GB):  83%|████████▎ | 48/58 [00:03<00:00, 34.84it/s]Capturing num tokens (num_tokens=32 avail_mem=21.07 GB):  83%|████████▎ | 48/58 [00:03<00:00, 34.84it/s]

    Capturing num tokens (num_tokens=28 avail_mem=21.07 GB):  83%|████████▎ | 48/58 [00:03<00:00, 34.84it/s]Capturing num tokens (num_tokens=28 avail_mem=21.07 GB):  90%|████████▉ | 52/58 [00:03<00:00, 35.30it/s]Capturing num tokens (num_tokens=24 avail_mem=21.07 GB):  90%|████████▉ | 52/58 [00:03<00:00, 35.30it/s]Capturing num tokens (num_tokens=20 avail_mem=21.07 GB):  90%|████████▉ | 52/58 [00:03<00:00, 35.30it/s]Capturing num tokens (num_tokens=16 avail_mem=21.06 GB):  90%|████████▉ | 52/58 [00:03<00:00, 35.30it/s]Capturing num tokens (num_tokens=12 avail_mem=21.06 GB):  90%|████████▉ | 52/58 [00:04<00:00, 35.30it/s]Capturing num tokens (num_tokens=12 avail_mem=21.06 GB):  97%|█████████▋| 56/58 [00:04<00:00, 36.16it/s]Capturing num tokens (num_tokens=8 avail_mem=21.06 GB):  97%|█████████▋| 56/58 [00:04<00:00, 36.16it/s] Capturing num tokens (num_tokens=4 avail_mem=21.05 GB):  97%|█████████▋| 56/58 [00:04<00:00, 36.16it/s]Capturing num tokens (num_tokens=4 avail_mem=21.05 GB): 100%|██████████| 58/58 [00:04<00:00, 14.15it/s]


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



<strong style='color: #00008B;'>First, I identify the two numbers to be added: 1 and 3.<br><br>Next, I perform the addition of these two numbers.<br><br>Finally, I arrive at the sum, which is 4.<br></strong>



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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition operation by combining the two numbers.<br><br>Finally, I calculate the result to find that 1 plus 3 equals 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I need to identify the two numbers in the addition problem, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Finally, the sum of 1 and 3 is 4.</strong>



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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I arrive at the conclusion that 1 plus 3 equals 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I'll start by identifying the two numbers involved in the addition: 1 and 3.<br><br>Next, I'll add these two numbers together to find the total.<br><br>After performing the addition, I'll determine that the sum of 1 and 3 is 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>Sure, let's solve the problem step by step.<br><br>**Problem:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I'll start by identifying the two numbers involved in the addition: 1 and 3.<br><br>Next, I'll add these two numbers together to find the total.<br><br>After performing the addition, I'll determine that the sum of 1 and 3 is 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Sure, let's solve the problem step by step.<br><br>**Problem:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.39s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]


    2026-04-27 17:04:42,676 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 17:04:42] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:03,  2.21s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:03,  2.21s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:11,  1.29s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:11,  1.29s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:31,  1.66it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:23,  2.22it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:23,  2.22it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.89it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.89it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.65it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.65it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.21it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.21it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.21it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.40it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.40it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.40it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.34it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.34it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.34it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.34it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.64it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.64it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.64it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.64it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.64it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 20.17it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 20.17it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 20.17it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 20.17it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.17it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.17it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.17it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 29.18it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 29.18it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 29.18it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 29.18it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 29.18it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 29.18it/s]

    Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 29.18it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.67it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.67it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.67it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.67it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.67it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.67it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.67it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.67it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.67it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 46.95it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 46.95it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 46.95it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 46.95it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 46.95it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 46.95it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 46.95it/s]

    Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 46.95it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 46.95it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 55.84it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 55.84it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 55.84it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 55.84it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 55.84it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 55.84it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 55.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=61.67 GB):   2%|▏         | 1/58 [00:00<00:17,  3.30it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.64 GB):   2%|▏         | 1/58 [00:00<00:17,  3.30it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=61.64 GB):   3%|▎         | 2/58 [00:00<00:16,  3.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.62 GB):   3%|▎         | 2/58 [00:00<00:16,  3.38it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=61.62 GB):   5%|▌         | 3/58 [00:00<00:15,  3.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.97 GB):   5%|▌         | 3/58 [00:00<00:15,  3.57it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=60.97 GB):   7%|▋         | 4/58 [00:01<00:13,  3.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.97 GB):   7%|▋         | 4/58 [00:01<00:13,  3.92it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=60.97 GB):   9%|▊         | 5/58 [00:01<00:12,  4.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.97 GB):   9%|▊         | 5/58 [00:01<00:12,  4.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.97 GB):  10%|█         | 6/58 [00:01<00:11,  4.64it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.96 GB):  10%|█         | 6/58 [00:01<00:11,  4.64it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=60.96 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.97 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.97 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.58it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.97 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.58it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=60.97 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.97 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.97 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.96 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.62it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=60.96 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.96 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.96 GB):  21%|██        | 12/58 [00:02<00:05,  7.85it/s]Capturing num tokens (num_tokens=3072 avail_mem=60.96 GB):  21%|██        | 12/58 [00:02<00:05,  7.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.96 GB):  21%|██        | 12/58 [00:02<00:05,  7.85it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=60.96 GB):  24%|██▍       | 14/58 [00:02<00:04,  9.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.96 GB):  24%|██▍       | 14/58 [00:02<00:04,  9.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.95 GB):  24%|██▍       | 14/58 [00:02<00:04,  9.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.95 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.95 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.40it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=60.95 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.95 GB):  31%|███       | 18/58 [00:02<00:03, 12.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.89 GB):  31%|███       | 18/58 [00:02<00:03, 12.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=51.33 GB):  31%|███       | 18/58 [00:02<00:03, 12.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=51.33 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=45.91 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.68it/s]

    Capturing num tokens (num_tokens=960 avail_mem=45.91 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.68it/s] Capturing num tokens (num_tokens=896 avail_mem=45.91 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.68it/s]Capturing num tokens (num_tokens=896 avail_mem=45.91 GB):  40%|███▉      | 23/58 [00:02<00:02, 17.27it/s]Capturing num tokens (num_tokens=832 avail_mem=45.90 GB):  40%|███▉      | 23/58 [00:02<00:02, 17.27it/s]Capturing num tokens (num_tokens=768 avail_mem=45.90 GB):  40%|███▉      | 23/58 [00:02<00:02, 17.27it/s]Capturing num tokens (num_tokens=704 avail_mem=45.89 GB):  40%|███▉      | 23/58 [00:02<00:02, 17.27it/s]Capturing num tokens (num_tokens=704 avail_mem=45.89 GB):  45%|████▍     | 26/58 [00:02<00:01, 20.45it/s]Capturing num tokens (num_tokens=640 avail_mem=45.89 GB):  45%|████▍     | 26/58 [00:02<00:01, 20.45it/s]

    Capturing num tokens (num_tokens=576 avail_mem=45.89 GB):  45%|████▍     | 26/58 [00:03<00:01, 20.45it/s]Capturing num tokens (num_tokens=512 avail_mem=45.88 GB):  45%|████▍     | 26/58 [00:03<00:01, 20.45it/s]Capturing num tokens (num_tokens=480 avail_mem=45.88 GB):  45%|████▍     | 26/58 [00:03<00:01, 20.45it/s]Capturing num tokens (num_tokens=480 avail_mem=45.88 GB):  52%|█████▏    | 30/58 [00:03<00:01, 24.48it/s]Capturing num tokens (num_tokens=448 avail_mem=45.88 GB):  52%|█████▏    | 30/58 [00:03<00:01, 24.48it/s]Capturing num tokens (num_tokens=416 avail_mem=45.87 GB):  52%|█████▏    | 30/58 [00:03<00:01, 24.48it/s]Capturing num tokens (num_tokens=384 avail_mem=45.87 GB):  52%|█████▏    | 30/58 [00:03<00:01, 24.48it/s]Capturing num tokens (num_tokens=352 avail_mem=45.86 GB):  52%|█████▏    | 30/58 [00:03<00:01, 24.48it/s]

    Capturing num tokens (num_tokens=352 avail_mem=45.86 GB):  59%|█████▊    | 34/58 [00:03<00:00, 27.54it/s]Capturing num tokens (num_tokens=320 avail_mem=45.86 GB):  59%|█████▊    | 34/58 [00:03<00:00, 27.54it/s]Capturing num tokens (num_tokens=288 avail_mem=45.87 GB):  59%|█████▊    | 34/58 [00:03<00:00, 27.54it/s]Capturing num tokens (num_tokens=256 avail_mem=45.86 GB):  59%|█████▊    | 34/58 [00:03<00:00, 27.54it/s]Capturing num tokens (num_tokens=240 avail_mem=45.86 GB):  59%|█████▊    | 34/58 [00:03<00:00, 27.54it/s]Capturing num tokens (num_tokens=240 avail_mem=45.86 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.85it/s]Capturing num tokens (num_tokens=224 avail_mem=45.85 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.85it/s]Capturing num tokens (num_tokens=208 avail_mem=45.85 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.85it/s]Capturing num tokens (num_tokens=192 avail_mem=45.85 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.85it/s]Capturing num tokens (num_tokens=176 avail_mem=45.84 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.85it/s]Capturing num tokens (num_tokens=160 avail_mem=45.84 GB):  66%|██████▌   | 38/58 [00:03<00:00, 30.85it/s]

    Capturing num tokens (num_tokens=160 avail_mem=45.84 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.97it/s]Capturing num tokens (num_tokens=144 avail_mem=45.83 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.97it/s]Capturing num tokens (num_tokens=128 avail_mem=45.84 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.97it/s]Capturing num tokens (num_tokens=112 avail_mem=45.83 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.97it/s]Capturing num tokens (num_tokens=96 avail_mem=45.83 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.97it/s] Capturing num tokens (num_tokens=80 avail_mem=45.82 GB):  74%|███████▍  | 43/58 [00:03<00:00, 33.97it/s]Capturing num tokens (num_tokens=80 avail_mem=45.82 GB):  83%|████████▎ | 48/58 [00:03<00:00, 36.31it/s]Capturing num tokens (num_tokens=64 avail_mem=45.82 GB):  83%|████████▎ | 48/58 [00:03<00:00, 36.31it/s]Capturing num tokens (num_tokens=48 avail_mem=45.82 GB):  83%|████████▎ | 48/58 [00:03<00:00, 36.31it/s]Capturing num tokens (num_tokens=32 avail_mem=45.81 GB):  83%|████████▎ | 48/58 [00:03<00:00, 36.31it/s]Capturing num tokens (num_tokens=28 avail_mem=45.81 GB):  83%|████████▎ | 48/58 [00:03<00:00, 36.31it/s]

    Capturing num tokens (num_tokens=24 avail_mem=45.81 GB):  83%|████████▎ | 48/58 [00:03<00:00, 36.31it/s]Capturing num tokens (num_tokens=24 avail_mem=45.81 GB):  91%|█████████▏| 53/58 [00:03<00:00, 38.20it/s]Capturing num tokens (num_tokens=20 avail_mem=45.81 GB):  91%|█████████▏| 53/58 [00:03<00:00, 38.20it/s]Capturing num tokens (num_tokens=16 avail_mem=45.80 GB):  91%|█████████▏| 53/58 [00:03<00:00, 38.20it/s]Capturing num tokens (num_tokens=12 avail_mem=45.80 GB):  91%|█████████▏| 53/58 [00:03<00:00, 38.20it/s]Capturing num tokens (num_tokens=8 avail_mem=45.79 GB):  91%|█████████▏| 53/58 [00:03<00:00, 38.20it/s] Capturing num tokens (num_tokens=4 avail_mem=45.79 GB):  91%|█████████▏| 53/58 [00:03<00:00, 38.20it/s]Capturing num tokens (num_tokens=4 avail_mem=45.79 GB): 100%|██████████| 58/58 [00:03<00:00, 39.42it/s]Capturing num tokens (num_tokens=4 avail_mem=45.79 GB): 100%|██████████| 58/58 [00:03<00:00, 15.25it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition by combining the two numbers.<br><br>Finally, I arrive at the result, which is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition by combining the two numbers.<br><br>Finally, I arrive at the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
