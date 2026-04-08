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


    [2026-04-08 17:28:13] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 17:28:13] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 17:28:13] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 17:28:13] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.07s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.17s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.15s/it]


    2026-04-08 17:28:17,217 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 17:28:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:03,  3.23s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:03,  3.23s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:27,  1.56s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:27,  1.56s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:25,  2.09it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:25,  2.09it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:19,  2.71it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:19,  2.71it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.44it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.44it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.22it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.11it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.11it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.11it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.79it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.79it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.79it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:05,  8.31it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:05,  8.31it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.31it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.89it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.89it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.89it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.80it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.80it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.80it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.80it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.07it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.07it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.07it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.07it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.07it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 19.93it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 19.93it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 19.93it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 19.93it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 19.93it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 19.93it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 26.13it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 26.13it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 26.13it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 26.13it/s]

    Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 26.13it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 26.13it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 26.13it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 33.59it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 38.87it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 38.87it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 38.87it/s]

    Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 38.87it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 38.87it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 38.87it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 41.66it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 41.66it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:06<00:00, 41.66it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:06<00:00, 41.66it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:06<00:00, 41.66it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:06<00:00, 41.66it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:06<00:00, 41.66it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:06<00:00, 45.49it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:06<00:00, 45.49it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:06<00:00, 45.49it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:06<00:00, 45.49it/s]

    Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:06<00:00, 45.49it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:06<00:00, 45.49it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:06<00:00, 45.49it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=102.52 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=102.52 GB):   2%|▏         | 1/58 [00:00<00:16,  3.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=102.49 GB):   2%|▏         | 1/58 [00:00<00:16,  3.47it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=102.49 GB):   3%|▎         | 2/58 [00:00<00:15,  3.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=102.49 GB):   3%|▎         | 2/58 [00:00<00:15,  3.60it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=102.49 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=101.97 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=101.97 GB):   7%|▋         | 4/58 [00:01<00:13,  4.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=100.14 GB):   7%|▋         | 4/58 [00:01<00:13,  4.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=100.14 GB):   9%|▊         | 5/58 [00:01<00:12,  4.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=100.15 GB):   9%|▊         | 5/58 [00:01<00:12,  4.26it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=100.15 GB):  10%|█         | 6/58 [00:01<00:12,  4.31it/s]Capturing num tokens (num_tokens=5120 avail_mem=100.15 GB):  10%|█         | 6/58 [00:01<00:12,  4.31it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=100.15 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=99.30 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.00it/s] Capturing num tokens (num_tokens=4608 avail_mem=99.30 GB):  14%|█▍        | 8/58 [00:01<00:11,  4.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=99.31 GB):  14%|█▍        | 8/58 [00:01<00:11,  4.40it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=99.31 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=99.31 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=99.31 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=99.30 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.62it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=99.30 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=99.30 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=99.30 GB):  21%|██        | 12/58 [00:02<00:06,  6.75it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=99.30 GB):  21%|██        | 12/58 [00:02<00:06,  6.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=99.30 GB):  22%|██▏       | 13/58 [00:02<00:07,  6.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=99.30 GB):  22%|██▏       | 13/58 [00:02<00:07,  6.15it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=99.30 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=99.27 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=99.27 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=99.27 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=99.27 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.23it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=99.27 GB):  28%|██▊       | 16/58 [00:03<00:05,  8.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=99.27 GB):  31%|███       | 18/58 [00:03<00:03, 10.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=99.26 GB):  31%|███       | 18/58 [00:03<00:03, 10.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=99.27 GB):  31%|███       | 18/58 [00:03<00:03, 10.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=99.27 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.38it/s]Capturing num tokens (num_tokens=1024 avail_mem=99.27 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.38it/s]

    Capturing num tokens (num_tokens=960 avail_mem=99.26 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.38it/s] Capturing num tokens (num_tokens=960 avail_mem=99.26 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.74it/s]Capturing num tokens (num_tokens=896 avail_mem=99.26 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.74it/s]Capturing num tokens (num_tokens=832 avail_mem=99.26 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.74it/s]Capturing num tokens (num_tokens=832 avail_mem=99.26 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.22it/s]Capturing num tokens (num_tokens=768 avail_mem=99.25 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.22it/s]

    Capturing num tokens (num_tokens=704 avail_mem=99.25 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.22it/s]Capturing num tokens (num_tokens=640 avail_mem=99.24 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.22it/s]Capturing num tokens (num_tokens=640 avail_mem=99.24 GB):  47%|████▋     | 27/58 [00:03<00:01, 18.99it/s]Capturing num tokens (num_tokens=576 avail_mem=99.24 GB):  47%|████▋     | 27/58 [00:03<00:01, 18.99it/s]Capturing num tokens (num_tokens=512 avail_mem=99.23 GB):  47%|████▋     | 27/58 [00:03<00:01, 18.99it/s]Capturing num tokens (num_tokens=480 avail_mem=99.23 GB):  47%|████▋     | 27/58 [00:03<00:01, 18.99it/s]Capturing num tokens (num_tokens=448 avail_mem=99.23 GB):  47%|████▋     | 27/58 [00:03<00:01, 18.99it/s]Capturing num tokens (num_tokens=448 avail_mem=99.23 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.30it/s]Capturing num tokens (num_tokens=416 avail_mem=99.23 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.30it/s]

    Capturing num tokens (num_tokens=384 avail_mem=99.22 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.30it/s]Capturing num tokens (num_tokens=352 avail_mem=99.22 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.30it/s]Capturing num tokens (num_tokens=320 avail_mem=99.21 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.30it/s]Capturing num tokens (num_tokens=320 avail_mem=99.21 GB):  60%|██████    | 35/58 [00:03<00:00, 26.72it/s]Capturing num tokens (num_tokens=288 avail_mem=99.21 GB):  60%|██████    | 35/58 [00:03<00:00, 26.72it/s]Capturing num tokens (num_tokens=256 avail_mem=99.20 GB):  60%|██████    | 35/58 [00:03<00:00, 26.72it/s]Capturing num tokens (num_tokens=240 avail_mem=99.20 GB):  60%|██████    | 35/58 [00:03<00:00, 26.72it/s]Capturing num tokens (num_tokens=224 avail_mem=99.20 GB):  60%|██████    | 35/58 [00:03<00:00, 26.72it/s]Capturing num tokens (num_tokens=224 avail_mem=99.20 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]Capturing num tokens (num_tokens=208 avail_mem=99.19 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]

    Capturing num tokens (num_tokens=192 avail_mem=99.19 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]Capturing num tokens (num_tokens=176 avail_mem=99.19 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]Capturing num tokens (num_tokens=160 avail_mem=99.19 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.48it/s]Capturing num tokens (num_tokens=160 avail_mem=99.19 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.63it/s]Capturing num tokens (num_tokens=144 avail_mem=99.18 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.63it/s]Capturing num tokens (num_tokens=128 avail_mem=99.19 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.63it/s]Capturing num tokens (num_tokens=112 avail_mem=99.19 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.63it/s]

    Capturing num tokens (num_tokens=96 avail_mem=99.18 GB):  74%|███████▍  | 43/58 [00:04<00:00, 31.63it/s] Capturing num tokens (num_tokens=96 avail_mem=99.18 GB):  81%|████████  | 47/58 [00:04<00:00, 29.59it/s]Capturing num tokens (num_tokens=80 avail_mem=99.18 GB):  81%|████████  | 47/58 [00:04<00:00, 29.59it/s]Capturing num tokens (num_tokens=64 avail_mem=99.17 GB):  81%|████████  | 47/58 [00:04<00:00, 29.59it/s]Capturing num tokens (num_tokens=48 avail_mem=99.17 GB):  81%|████████  | 47/58 [00:04<00:00, 29.59it/s]Capturing num tokens (num_tokens=32 avail_mem=99.17 GB):  81%|████████  | 47/58 [00:04<00:00, 29.59it/s]Capturing num tokens (num_tokens=32 avail_mem=99.17 GB):  88%|████████▊ | 51/58 [00:04<00:00, 31.58it/s]Capturing num tokens (num_tokens=28 avail_mem=99.17 GB):  88%|████████▊ | 51/58 [00:04<00:00, 31.58it/s]Capturing num tokens (num_tokens=24 avail_mem=99.16 GB):  88%|████████▊ | 51/58 [00:04<00:00, 31.58it/s]Capturing num tokens (num_tokens=20 avail_mem=99.16 GB):  88%|████████▊ | 51/58 [00:04<00:00, 31.58it/s]

    Capturing num tokens (num_tokens=16 avail_mem=99.15 GB):  88%|████████▊ | 51/58 [00:04<00:00, 31.58it/s]Capturing num tokens (num_tokens=16 avail_mem=99.15 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.96it/s]Capturing num tokens (num_tokens=12 avail_mem=99.15 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.96it/s]Capturing num tokens (num_tokens=8 avail_mem=99.15 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.96it/s] Capturing num tokens (num_tokens=4 avail_mem=99.14 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.96it/s]Capturing num tokens (num_tokens=4 avail_mem=99.14 GB): 100%|██████████| 58/58 [00:04<00:00, 13.20it/s]


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>I start by identifying the two numbers involved, which are 1 and 3.<br><br>Next, I perform the addition operation by combining these two numbers together.<br><br>Finally, I calculate the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of the numbers 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I recognize that I need to add the numbers 1 and 3.<br><br>I'll start by identifying the two numbers involved in the addition.<br><br>Next, I'll perform the addition operation by combining the values of 1 and 3.<br><br>Finally, I'll calculate the sum to find the result of 1 plus 3.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = \boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Certainly! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add 1 and 3 together.<br><br>The result of this addition is 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>To find the sum of 1 and 3, follow these easy steps:<br><br>1. **Start with the first number:**<br>   <br>   \[<br>   1<br>   \]<br><br>2. **Add the second number to it:**<br>   <br>   \[<br>   1 + 3<br>   \]<br><br>3. **Calculate the sum:**<br>   <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add 1 and 3 together.<br><br>The result of this addition is 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>To find the sum of 1 and 3, follow these easy steps:<br><br>1. **Start with the first number:**<br>   <br>   \[<br>   1<br>   \]<br><br>2. **Add the second number to it:**<br>   <br>   \[<br>   1 + 3<br>   \]<br><br>3. **Calculate the sum:**<br>   <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.03s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.12s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.11s/it]


    2026-04-08 17:29:03,760 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 17:29:03] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:05,  3.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:05,  3.25s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:27,  1.57s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:27,  1.57s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:34,  1.57it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:34,  1.57it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.16it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.16it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.53it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.53it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.31it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.31it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.19it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.19it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.19it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.87it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.87it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.87it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:05,  8.37it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:05,  8.37it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.37it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.98it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.98it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.98it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.89it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.89it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.89it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.89it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.15it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.15it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.15it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.15it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.15it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 26.75it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 26.75it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 26.75it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 26.75it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 26.75it/s]

    Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 26.75it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 26.75it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 34.11it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 34.11it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 34.11it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 34.11it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 34.11it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 34.11it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 34.11it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]

    Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 42.42it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 42.42it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 42.42it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 42.42it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:06<00:00, 42.42it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:06<00:00, 42.42it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:06<00:00, 42.42it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:06<00:00, 46.56it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:06<00:00, 46.56it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:06<00:00, 46.56it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 46.56it/s]

    Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 46.56it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 46.56it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.48it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=122.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=122.24 GB):   2%|▏         | 1/58 [00:00<00:16,  3.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=122.21 GB):   2%|▏         | 1/58 [00:00<00:16,  3.48it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=122.21 GB):   3%|▎         | 2/58 [00:00<00:19,  2.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=122.21 GB):   3%|▎         | 2/58 [00:00<00:19,  2.88it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=122.21 GB):   5%|▌         | 3/58 [00:00<00:16,  3.33it/s]Capturing num tokens (num_tokens=6656 avail_mem=122.22 GB):   5%|▌         | 3/58 [00:00<00:16,  3.33it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=122.22 GB):   7%|▋         | 4/58 [00:01<00:14,  3.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=122.22 GB):   7%|▋         | 4/58 [00:01<00:14,  3.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=122.22 GB):   9%|▊         | 5/58 [00:01<00:12,  4.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=122.22 GB):   9%|▊         | 5/58 [00:01<00:12,  4.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=122.22 GB):  10%|█         | 6/58 [00:01<00:11,  4.52it/s]Capturing num tokens (num_tokens=5120 avail_mem=122.22 GB):  10%|█         | 6/58 [00:01<00:11,  4.52it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=122.22 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=122.23 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=122.23 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=122.23 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.46it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=122.23 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=122.24 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=122.24 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=122.24 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.47it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=122.24 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=122.24 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=122.24 GB):  21%|██        | 12/58 [00:02<00:06,  7.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=122.23 GB):  21%|██        | 12/58 [00:02<00:06,  7.64it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=122.24 GB):  21%|██        | 12/58 [00:02<00:06,  7.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=122.24 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.82it/s]Capturing num tokens (num_tokens=2560 avail_mem=122.24 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=122.24 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.82it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=122.24 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=122.23 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=122.24 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=122.24 GB):  31%|███       | 18/58 [00:02<00:03, 11.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=122.23 GB):  31%|███       | 18/58 [00:02<00:03, 11.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=122.23 GB):  31%|███       | 18/58 [00:02<00:03, 11.72it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=122.23 GB):  31%|███       | 18/58 [00:02<00:03, 11.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=122.23 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s]Capturing num tokens (num_tokens=960 avail_mem=122.23 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s] Capturing num tokens (num_tokens=896 avail_mem=122.22 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s]Capturing num tokens (num_tokens=832 avail_mem=122.22 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s]Capturing num tokens (num_tokens=832 avail_mem=122.22 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.73it/s]Capturing num tokens (num_tokens=768 avail_mem=122.22 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.73it/s]Capturing num tokens (num_tokens=704 avail_mem=122.21 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.73it/s]

    Capturing num tokens (num_tokens=640 avail_mem=122.21 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.73it/s]Capturing num tokens (num_tokens=640 avail_mem=122.21 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.70it/s]Capturing num tokens (num_tokens=576 avail_mem=122.21 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.70it/s]Capturing num tokens (num_tokens=512 avail_mem=122.20 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.70it/s]Capturing num tokens (num_tokens=480 avail_mem=122.20 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.70it/s]Capturing num tokens (num_tokens=448 avail_mem=122.20 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.70it/s]Capturing num tokens (num_tokens=448 avail_mem=122.20 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=416 avail_mem=122.19 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=384 avail_mem=122.19 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]

    Capturing num tokens (num_tokens=352 avail_mem=122.18 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=320 avail_mem=122.18 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=320 avail_mem=122.18 GB):  60%|██████    | 35/58 [00:03<00:00, 26.30it/s]Capturing num tokens (num_tokens=288 avail_mem=122.18 GB):  60%|██████    | 35/58 [00:03<00:00, 26.30it/s]Capturing num tokens (num_tokens=256 avail_mem=122.17 GB):  60%|██████    | 35/58 [00:03<00:00, 26.30it/s]Capturing num tokens (num_tokens=240 avail_mem=122.17 GB):  60%|██████    | 35/58 [00:03<00:00, 26.30it/s]Capturing num tokens (num_tokens=240 avail_mem=122.17 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.79it/s]Capturing num tokens (num_tokens=224 avail_mem=122.17 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.79it/s]

    Capturing num tokens (num_tokens=208 avail_mem=122.16 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.79it/s]Capturing num tokens (num_tokens=192 avail_mem=122.16 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.79it/s]

    Capturing num tokens (num_tokens=192 avail_mem=122.16 GB):  71%|███████   | 41/58 [00:03<00:00, 19.22it/s]Capturing num tokens (num_tokens=176 avail_mem=122.15 GB):  71%|███████   | 41/58 [00:03<00:00, 19.22it/s]Capturing num tokens (num_tokens=160 avail_mem=122.15 GB):  71%|███████   | 41/58 [00:03<00:00, 19.22it/s]Capturing num tokens (num_tokens=144 avail_mem=122.15 GB):  71%|███████   | 41/58 [00:03<00:00, 19.22it/s]Capturing num tokens (num_tokens=128 avail_mem=122.16 GB):  71%|███████   | 41/58 [00:03<00:00, 19.22it/s]Capturing num tokens (num_tokens=128 avail_mem=122.16 GB):  78%|███████▊  | 45/58 [00:03<00:00, 23.01it/s]Capturing num tokens (num_tokens=112 avail_mem=122.15 GB):  78%|███████▊  | 45/58 [00:03<00:00, 23.01it/s]Capturing num tokens (num_tokens=96 avail_mem=122.15 GB):  78%|███████▊  | 45/58 [00:03<00:00, 23.01it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=122.14 GB):  78%|███████▊  | 45/58 [00:04<00:00, 23.01it/s]Capturing num tokens (num_tokens=80 avail_mem=122.14 GB):  83%|████████▎ | 48/58 [00:04<00:00, 19.73it/s]Capturing num tokens (num_tokens=64 avail_mem=122.14 GB):  83%|████████▎ | 48/58 [00:04<00:00, 19.73it/s]Capturing num tokens (num_tokens=48 avail_mem=122.14 GB):  83%|████████▎ | 48/58 [00:04<00:00, 19.73it/s]Capturing num tokens (num_tokens=32 avail_mem=122.13 GB):  83%|████████▎ | 48/58 [00:04<00:00, 19.73it/s]Capturing num tokens (num_tokens=28 avail_mem=122.13 GB):  83%|████████▎ | 48/58 [00:04<00:00, 19.73it/s]Capturing num tokens (num_tokens=28 avail_mem=122.13 GB):  90%|████████▉ | 52/58 [00:04<00:00, 23.18it/s]Capturing num tokens (num_tokens=24 avail_mem=122.13 GB):  90%|████████▉ | 52/58 [00:04<00:00, 23.18it/s]Capturing num tokens (num_tokens=20 avail_mem=122.12 GB):  90%|████████▉ | 52/58 [00:04<00:00, 23.18it/s]Capturing num tokens (num_tokens=16 avail_mem=122.12 GB):  90%|████████▉ | 52/58 [00:04<00:00, 23.18it/s]

    Capturing num tokens (num_tokens=12 avail_mem=122.12 GB):  90%|████████▉ | 52/58 [00:04<00:00, 23.18it/s]Capturing num tokens (num_tokens=12 avail_mem=122.12 GB):  97%|█████████▋| 56/58 [00:04<00:00, 26.18it/s]Capturing num tokens (num_tokens=8 avail_mem=122.11 GB):  97%|█████████▋| 56/58 [00:04<00:00, 26.18it/s] Capturing num tokens (num_tokens=4 avail_mem=122.11 GB):  97%|█████████▋| 56/58 [00:04<00:00, 26.18it/s]Capturing num tokens (num_tokens=4 avail_mem=122.11 GB): 100%|██████████| 58/58 [00:04<00:00, 13.33it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
