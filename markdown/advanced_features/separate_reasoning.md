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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.38s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]


    2026-04-28 20:40:44,613 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 20:40:44] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:53,  5.15s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:53,  5.15s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:05,  2.24s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:05,  2.24s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:11,  1.31s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:11,  1.31s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.21it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.21it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.88it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.88it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.63it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.63it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.48it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.48it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.48it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.19it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.19it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.19it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.76it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.76it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.76it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.36it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.36it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.36it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.29it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.29it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.29it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.29it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 13.46it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 13.46it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 13.46it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:07<00:02, 13.67it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:07<00:02, 13.67it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:07<00:02, 13.67it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:02, 14.16it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:02, 14.16it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:02, 14.16it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:07<00:02, 15.17it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:07<00:02, 15.17it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:07<00:02, 15.17it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:07<00:02, 15.17it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:07<00:02, 15.17it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:07<00:02, 15.17it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:07<00:01, 23.42it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:07<00:01, 23.42it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:07<00:01, 23.42it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:07<00:01, 23.42it/s]

    Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:07<00:01, 23.42it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:07<00:01, 23.42it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:07<00:01, 23.42it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:07<00:00, 32.18it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:07<00:00, 32.18it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:07<00:00, 32.18it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:07<00:00, 32.18it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:07<00:00, 32.18it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:07<00:00, 32.18it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:07<00:00, 32.18it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:07<00:00, 32.18it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:07<00:00, 32.18it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:07<00:00, 44.19it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:07<00:00, 44.19it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:07<00:00, 44.19it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:07<00:00, 44.19it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:07<00:00, 44.19it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:07<00:00, 44.19it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:07<00:00, 44.19it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:07<00:00, 44.19it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:07<00:00, 44.19it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:07<00:00, 53.32it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:07<00:00, 53.32it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:07<00:00, 53.32it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:07<00:00, 53.32it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:07<00:00, 53.32it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:07<00:00, 53.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=21.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=21.24 GB):   2%|▏         | 1/58 [00:00<00:16,  3.45it/s]Capturing num tokens (num_tokens=7680 avail_mem=21.21 GB):   2%|▏         | 1/58 [00:00<00:16,  3.45it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=21.21 GB):   3%|▎         | 2/58 [00:00<00:17,  3.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=21.20 GB):   3%|▎         | 2/58 [00:00<00:17,  3.17it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=21.20 GB):   5%|▌         | 3/58 [00:00<00:15,  3.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=21.20 GB):   5%|▌         | 3/58 [00:00<00:15,  3.48it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=21.20 GB):   7%|▋         | 4/58 [00:01<00:14,  3.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=21.20 GB):   7%|▋         | 4/58 [00:01<00:14,  3.81it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=21.20 GB):   9%|▊         | 5/58 [00:01<00:14,  3.57it/s]Capturing num tokens (num_tokens=5632 avail_mem=21.20 GB):   9%|▊         | 5/58 [00:01<00:14,  3.57it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=21.20 GB):  10%|█         | 6/58 [00:01<00:16,  3.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=21.20 GB):  10%|█         | 6/58 [00:01<00:16,  3.14it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=21.20 GB):  12%|█▏        | 7/58 [00:02<00:17,  2.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=21.20 GB):  12%|█▏        | 7/58 [00:02<00:17,  2.96it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=21.20 GB):  14%|█▍        | 8/58 [00:02<00:16,  2.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=21.20 GB):  14%|█▍        | 8/58 [00:02<00:16,  2.97it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=21.20 GB):  16%|█▌        | 9/58 [00:02<00:16,  3.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=21.20 GB):  16%|█▌        | 9/58 [00:02<00:16,  3.06it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=21.20 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=21.20 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.15it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=21.20 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=21.20 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.29it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=21.20 GB):  21%|██        | 12/58 [00:03<00:13,  3.47it/s]Capturing num tokens (num_tokens=3072 avail_mem=21.20 GB):  21%|██        | 12/58 [00:03<00:13,  3.47it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=21.20 GB):  22%|██▏       | 13/58 [00:03<00:12,  3.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=21.20 GB):  22%|██▏       | 13/58 [00:03<00:12,  3.64it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=21.20 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=21.19 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.88it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=21.19 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=21.19 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=21.19 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=21.19 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.41it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=21.19 GB):  29%|██▉       | 17/58 [00:04<00:08,  4.80it/s]Capturing num tokens (num_tokens=1792 avail_mem=21.18 GB):  29%|██▉       | 17/58 [00:04<00:08,  4.80it/s]Capturing num tokens (num_tokens=1792 avail_mem=21.18 GB):  31%|███       | 18/58 [00:04<00:07,  5.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=21.18 GB):  31%|███       | 18/58 [00:04<00:07,  5.13it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=21.18 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=21.18 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=21.18 GB):  34%|███▍      | 20/58 [00:05<00:06,  5.94it/s]Capturing num tokens (num_tokens=1024 avail_mem=21.13 GB):  34%|███▍      | 20/58 [00:05<00:06,  5.94it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=21.13 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.75it/s]Capturing num tokens (num_tokens=960 avail_mem=21.13 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.75it/s] Capturing num tokens (num_tokens=960 avail_mem=21.13 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.30it/s]Capturing num tokens (num_tokens=896 avail_mem=21.13 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.30it/s]

    Capturing num tokens (num_tokens=896 avail_mem=21.13 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.43it/s]Capturing num tokens (num_tokens=832 avail_mem=21.12 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.43it/s]Capturing num tokens (num_tokens=832 avail_mem=21.12 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.43it/s]Capturing num tokens (num_tokens=768 avail_mem=21.12 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.43it/s]

    Capturing num tokens (num_tokens=704 avail_mem=21.11 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.43it/s]Capturing num tokens (num_tokens=704 avail_mem=21.11 GB):  45%|████▍     | 26/58 [00:05<00:03,  9.24it/s]Capturing num tokens (num_tokens=640 avail_mem=21.11 GB):  45%|████▍     | 26/58 [00:05<00:03,  9.24it/s]Capturing num tokens (num_tokens=576 avail_mem=21.11 GB):  45%|████▍     | 26/58 [00:05<00:03,  9.24it/s]Capturing num tokens (num_tokens=576 avail_mem=21.11 GB):  48%|████▊     | 28/58 [00:05<00:02, 10.96it/s]Capturing num tokens (num_tokens=512 avail_mem=21.10 GB):  48%|████▊     | 28/58 [00:05<00:02, 10.96it/s]

    Capturing num tokens (num_tokens=480 avail_mem=21.10 GB):  48%|████▊     | 28/58 [00:05<00:02, 10.96it/s]Capturing num tokens (num_tokens=448 avail_mem=21.10 GB):  48%|████▊     | 28/58 [00:05<00:02, 10.96it/s]Capturing num tokens (num_tokens=448 avail_mem=21.10 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.01it/s]Capturing num tokens (num_tokens=416 avail_mem=21.09 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.01it/s]Capturing num tokens (num_tokens=384 avail_mem=21.09 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.01it/s]Capturing num tokens (num_tokens=352 avail_mem=21.08 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.01it/s]

    Capturing num tokens (num_tokens=352 avail_mem=21.08 GB):  59%|█████▊    | 34/58 [00:06<00:01, 16.52it/s]Capturing num tokens (num_tokens=320 avail_mem=21.07 GB):  59%|█████▊    | 34/58 [00:06<00:01, 16.52it/s]Capturing num tokens (num_tokens=288 avail_mem=21.08 GB):  59%|█████▊    | 34/58 [00:06<00:01, 16.52it/s]Capturing num tokens (num_tokens=256 avail_mem=21.07 GB):  59%|█████▊    | 34/58 [00:06<00:01, 16.52it/s]Capturing num tokens (num_tokens=256 avail_mem=21.07 GB):  64%|██████▍   | 37/58 [00:06<00:01, 18.06it/s]Capturing num tokens (num_tokens=240 avail_mem=21.07 GB):  64%|██████▍   | 37/58 [00:06<00:01, 18.06it/s]Capturing num tokens (num_tokens=224 avail_mem=21.07 GB):  64%|██████▍   | 37/58 [00:06<00:01, 18.06it/s]Capturing num tokens (num_tokens=208 avail_mem=21.06 GB):  64%|██████▍   | 37/58 [00:06<00:01, 18.06it/s]

    Capturing num tokens (num_tokens=192 avail_mem=21.06 GB):  64%|██████▍   | 37/58 [00:06<00:01, 18.06it/s]Capturing num tokens (num_tokens=176 avail_mem=21.06 GB):  64%|██████▍   | 37/58 [00:06<00:01, 18.06it/s]Capturing num tokens (num_tokens=176 avail_mem=21.06 GB):  72%|███████▏  | 42/58 [00:06<00:00, 24.21it/s]Capturing num tokens (num_tokens=160 avail_mem=21.06 GB):  72%|███████▏  | 42/58 [00:06<00:00, 24.21it/s]Capturing num tokens (num_tokens=144 avail_mem=21.05 GB):  72%|███████▏  | 42/58 [00:06<00:00, 24.21it/s]Capturing num tokens (num_tokens=128 avail_mem=21.05 GB):  72%|███████▏  | 42/58 [00:06<00:00, 24.21it/s]Capturing num tokens (num_tokens=112 avail_mem=21.05 GB):  72%|███████▏  | 42/58 [00:06<00:00, 24.21it/s]Capturing num tokens (num_tokens=112 avail_mem=21.05 GB):  79%|███████▉  | 46/58 [00:06<00:00, 27.95it/s]Capturing num tokens (num_tokens=96 avail_mem=21.04 GB):  79%|███████▉  | 46/58 [00:06<00:00, 27.95it/s] Capturing num tokens (num_tokens=80 avail_mem=21.04 GB):  79%|███████▉  | 46/58 [00:06<00:00, 27.95it/s]Capturing num tokens (num_tokens=64 avail_mem=21.04 GB):  79%|███████▉  | 46/58 [00:06<00:00, 27.95it/s]

    Capturing num tokens (num_tokens=48 avail_mem=21.03 GB):  79%|███████▉  | 46/58 [00:06<00:00, 27.95it/s]Capturing num tokens (num_tokens=32 avail_mem=21.03 GB):  79%|███████▉  | 46/58 [00:06<00:00, 27.95it/s]Capturing num tokens (num_tokens=32 avail_mem=21.03 GB):  88%|████████▊ | 51/58 [00:06<00:00, 31.95it/s]Capturing num tokens (num_tokens=28 avail_mem=21.03 GB):  88%|████████▊ | 51/58 [00:06<00:00, 31.95it/s]Capturing num tokens (num_tokens=24 avail_mem=21.02 GB):  88%|████████▊ | 51/58 [00:06<00:00, 31.95it/s]Capturing num tokens (num_tokens=20 avail_mem=21.02 GB):  88%|████████▊ | 51/58 [00:06<00:00, 31.95it/s]Capturing num tokens (num_tokens=16 avail_mem=21.02 GB):  88%|████████▊ | 51/58 [00:06<00:00, 31.95it/s]Capturing num tokens (num_tokens=16 avail_mem=21.02 GB):  95%|█████████▍| 55/58 [00:06<00:00, 31.30it/s]Capturing num tokens (num_tokens=12 avail_mem=20.04 GB):  95%|█████████▍| 55/58 [00:06<00:00, 31.30it/s]

    Capturing num tokens (num_tokens=8 avail_mem=20.04 GB):  95%|█████████▍| 55/58 [00:06<00:00, 31.30it/s] Capturing num tokens (num_tokens=4 avail_mem=20.04 GB):  95%|█████████▍| 55/58 [00:06<00:00, 31.30it/s]Capturing num tokens (num_tokens=4 avail_mem=20.04 GB): 100%|██████████| 58/58 [00:07<00:00,  8.22it/s]


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I conclude that the result of adding 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I will add the two numbers together.<br><br>After performing the addition, I find that 1 plus 3 equals 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>To find the sum of 1 and 3, I start by identifying the two numbers involved.<br><br>Next, I add these numbers together to get the total.<br><br>Finally, I conclude that the result of 1 plus 3 is 4.<br></think><br><br>To solve the addition problem \(1 + 3\), follow these simple steps:<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Present the final answer:**<br>   \[<br>   \boxed{4}<br>   \]<br><br>Therefore, the sum of \(1\) and \(3\) is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.36s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]


    2026-04-28 20:41:31,127 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 20:41:31] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:56,  5.20s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:56,  5.20s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:06,  2.26s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:06,  2.26s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.15it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.63it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.83it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.83it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.53it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.53it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.31it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.31it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.31it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  5.98it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  5.98it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  5.98it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:06,  7.49it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:06,  7.49it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:06,  7.49it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.10it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.10it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.10it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.86it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.86it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.86it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:07<00:03, 12.76it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:07<00:03, 12.76it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:07<00:03, 12.76it/s]

    Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:07<00:03, 12.76it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:07<00:02, 16.42it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:07<00:02, 16.42it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:07<00:02, 16.42it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:07<00:02, 16.42it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:07<00:02, 16.42it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:07<00:01, 22.03it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:07<00:01, 22.03it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:07<00:01, 22.03it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:07<00:01, 22.03it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:07<00:01, 22.03it/s]

    Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:07<00:01, 22.03it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:07<00:01, 22.03it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:07<00:00, 29.85it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:07<00:00, 29.85it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:07<00:00, 29.85it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:07<00:00, 29.85it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:07<00:00, 29.85it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:07<00:00, 29.85it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:07<00:00, 29.85it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:07<00:00, 36.80it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:07<00:00, 36.80it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:07<00:00, 36.80it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:07<00:00, 36.80it/s]

    Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:07<00:00, 36.80it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:07<00:00, 36.80it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:07<00:00, 36.80it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:07<00:00, 36.80it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:07<00:00, 44.46it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:07<00:00, 44.46it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:07<00:00, 44.46it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:07<00:00, 44.46it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:07<00:00, 44.46it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:07<00:00, 44.46it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:07<00:00, 45.94it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:07<00:00, 45.94it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 45.94it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 45.94it/s]

    Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:07<00:00, 45.94it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:07<00:00, 45.94it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:07<00:00, 45.94it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:07<00:00, 49.15it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:07<00:00, 49.15it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:07<00:00, 49.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=45.99 GB):   2%|▏         | 1/58 [00:00<00:17,  3.33it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.95 GB):   2%|▏         | 1/58 [00:00<00:17,  3.33it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=45.95 GB):   3%|▎         | 2/58 [00:00<00:15,  3.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.95 GB):   3%|▎         | 2/58 [00:00<00:15,  3.52it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=45.95 GB):   5%|▌         | 3/58 [00:00<00:14,  3.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=45.95 GB):   5%|▌         | 3/58 [00:00<00:14,  3.75it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=45.95 GB):   7%|▋         | 4/58 [00:01<00:13,  4.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.95 GB):   7%|▋         | 4/58 [00:01<00:13,  4.06it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=45.95 GB):   9%|▊         | 5/58 [00:01<00:12,  4.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.95 GB):   9%|▊         | 5/58 [00:01<00:12,  4.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.95 GB):  10%|█         | 6/58 [00:01<00:11,  4.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.95 GB):  10%|█         | 6/58 [00:01<00:11,  4.72it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=45.95 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.95 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.95 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.16 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.45it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.16 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.16 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.16 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.16 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.41it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.16 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.16 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.16 GB):  21%|██        | 12/58 [00:02<00:06,  7.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.16 GB):  21%|██        | 12/58 [00:02<00:06,  7.64it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.16 GB):  21%|██        | 12/58 [00:02<00:06,  7.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.16 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.15 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.15 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.74it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.15 GB):  31%|███       | 18/58 [00:02<00:03, 11.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.14 GB):  31%|███       | 18/58 [00:02<00:03, 11.53it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.14 GB):  31%|███       | 18/58 [00:02<00:03, 11.53it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=43.13 GB):  31%|███       | 18/58 [00:02<00:03, 11.53it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.13 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.49it/s]Capturing num tokens (num_tokens=960 avail_mem=43.13 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.49it/s] Capturing num tokens (num_tokens=896 avail_mem=43.12 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.49it/s]Capturing num tokens (num_tokens=832 avail_mem=43.12 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.49it/s]Capturing num tokens (num_tokens=832 avail_mem=43.12 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.57it/s]Capturing num tokens (num_tokens=768 avail_mem=43.11 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.57it/s]Capturing num tokens (num_tokens=704 avail_mem=43.11 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.57it/s]

    Capturing num tokens (num_tokens=640 avail_mem=43.11 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.57it/s]Capturing num tokens (num_tokens=640 avail_mem=43.11 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Capturing num tokens (num_tokens=576 avail_mem=43.10 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Capturing num tokens (num_tokens=512 avail_mem=43.10 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Capturing num tokens (num_tokens=480 avail_mem=43.10 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Capturing num tokens (num_tokens=448 avail_mem=43.09 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.39it/s]Capturing num tokens (num_tokens=448 avail_mem=43.09 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.98it/s]Capturing num tokens (num_tokens=416 avail_mem=43.09 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.98it/s]Capturing num tokens (num_tokens=384 avail_mem=43.09 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.98it/s]

    Capturing num tokens (num_tokens=352 avail_mem=43.08 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.98it/s]Capturing num tokens (num_tokens=320 avail_mem=43.08 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.98it/s]Capturing num tokens (num_tokens=320 avail_mem=43.08 GB):  60%|██████    | 35/58 [00:03<00:00, 27.23it/s]Capturing num tokens (num_tokens=288 avail_mem=43.08 GB):  60%|██████    | 35/58 [00:03<00:00, 27.23it/s]Capturing num tokens (num_tokens=256 avail_mem=43.08 GB):  60%|██████    | 35/58 [00:03<00:00, 27.23it/s]Capturing num tokens (num_tokens=240 avail_mem=43.07 GB):  60%|██████    | 35/58 [00:03<00:00, 27.23it/s]Capturing num tokens (num_tokens=224 avail_mem=43.07 GB):  60%|██████    | 35/58 [00:03<00:00, 27.23it/s]Capturing num tokens (num_tokens=224 avail_mem=43.07 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.33it/s]Capturing num tokens (num_tokens=208 avail_mem=43.07 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.33it/s]Capturing num tokens (num_tokens=192 avail_mem=43.06 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.33it/s]

    Capturing num tokens (num_tokens=176 avail_mem=43.06 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.33it/s]Capturing num tokens (num_tokens=160 avail_mem=43.06 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.33it/s]Capturing num tokens (num_tokens=160 avail_mem=43.06 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.54it/s]Capturing num tokens (num_tokens=144 avail_mem=43.05 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.54it/s]Capturing num tokens (num_tokens=128 avail_mem=43.05 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.54it/s]Capturing num tokens (num_tokens=112 avail_mem=43.05 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.54it/s]Capturing num tokens (num_tokens=96 avail_mem=43.05 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.54it/s] Capturing num tokens (num_tokens=96 avail_mem=43.05 GB):  81%|████████  | 47/58 [00:03<00:00, 34.02it/s]Capturing num tokens (num_tokens=80 avail_mem=43.04 GB):  81%|████████  | 47/58 [00:03<00:00, 34.02it/s]Capturing num tokens (num_tokens=64 avail_mem=43.04 GB):  81%|████████  | 47/58 [00:03<00:00, 34.02it/s]

    Capturing num tokens (num_tokens=48 avail_mem=43.03 GB):  81%|████████  | 47/58 [00:03<00:00, 34.02it/s]Capturing num tokens (num_tokens=32 avail_mem=43.03 GB):  81%|████████  | 47/58 [00:03<00:00, 34.02it/s]Capturing num tokens (num_tokens=28 avail_mem=43.03 GB):  81%|████████  | 47/58 [00:03<00:00, 34.02it/s]Capturing num tokens (num_tokens=28 avail_mem=43.03 GB):  90%|████████▉ | 52/58 [00:03<00:00, 36.46it/s]Capturing num tokens (num_tokens=24 avail_mem=43.03 GB):  90%|████████▉ | 52/58 [00:03<00:00, 36.46it/s]Capturing num tokens (num_tokens=20 avail_mem=43.02 GB):  90%|████████▉ | 52/58 [00:03<00:00, 36.46it/s]Capturing num tokens (num_tokens=16 avail_mem=43.02 GB):  90%|████████▉ | 52/58 [00:03<00:00, 36.46it/s]Capturing num tokens (num_tokens=12 avail_mem=43.01 GB):  90%|████████▉ | 52/58 [00:03<00:00, 36.46it/s]Capturing num tokens (num_tokens=12 avail_mem=43.01 GB):  97%|█████████▋| 56/58 [00:03<00:00, 37.39it/s]Capturing num tokens (num_tokens=8 avail_mem=43.01 GB):  97%|█████████▋| 56/58 [00:03<00:00, 37.39it/s] Capturing num tokens (num_tokens=4 avail_mem=43.01 GB):  97%|█████████▋| 56/58 [00:03<00:00, 37.39it/s]

    Capturing num tokens (num_tokens=4 avail_mem=43.01 GB): 100%|██████████| 58/58 [00:03<00:00, 15.01it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>To find the sum of 1 and 3, follow these simple steps:<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\(\boxed{4}\)</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>To find the sum of 1 and 3, follow these simple steps:<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\(\boxed{4}\)</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
