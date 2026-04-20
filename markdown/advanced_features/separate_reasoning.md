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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(
    [2026-04-20 21:46:15] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-20 21:46:16] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-20 21:46:17] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-20 21:46:18] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-20 21:46:24] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-20 21:46:25] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [2026-04-20 21:46:25] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.52s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.53s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.53s/it]


    2026-04-20 21:46:33,689 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-20 21:46:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:50,  3.00s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:50,  3.00s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:22,  1.48s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:22,  1.48s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:49,  1.12it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:49,  1.12it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.87it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.87it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.59it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.59it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.39it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.39it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.26it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.26it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.26it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.92it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.92it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.92it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.42it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.42it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.42it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  9.92it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  9.92it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.92it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.81it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.81it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.81it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.81it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.05it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.05it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.05it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.05it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.05it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.50it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.50it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.50it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.50it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.50it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.50it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:01, 20.50it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.29it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 46.95it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 46.95it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 46.95it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 46.95it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 46.95it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 46.95it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 46.95it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 48.77it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 48.77it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 48.77it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 48.77it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 48.77it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 48.77it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 48.77it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 48.77it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 48.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=61.11 GB):   2%|▏         | 1/58 [00:00<00:17,  3.33it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.92 GB):   2%|▏         | 1/58 [00:00<00:17,  3.33it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=60.92 GB):   3%|▎         | 2/58 [00:00<00:15,  3.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.92 GB):   3%|▎         | 2/58 [00:00<00:15,  3.52it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=60.92 GB):   5%|▌         | 3/58 [00:00<00:14,  3.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.92 GB):   5%|▌         | 3/58 [00:00<00:14,  3.75it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=60.92 GB):   7%|▋         | 4/58 [00:01<00:13,  4.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.93 GB):   7%|▋         | 4/58 [00:01<00:13,  4.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=60.93 GB):   9%|▊         | 5/58 [00:01<00:12,  4.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.93 GB):   9%|▊         | 5/58 [00:01<00:12,  4.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.93 GB):  10%|█         | 6/58 [00:01<00:11,  4.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.93 GB):  10%|█         | 6/58 [00:01<00:11,  4.69it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=60.93 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.94 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.94 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.94 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.54it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=60.94 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.94 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.94 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.94 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.51it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=60.94 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.94 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.94 GB):  21%|██        | 12/58 [00:02<00:06,  7.55it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.88 GB):  21%|██        | 12/58 [00:02<00:06,  7.55it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=59.88 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.92 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.92 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.92 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.92 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.40it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=45.92 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=45.92 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.89it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.92 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.92 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.92 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.92 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.58it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=45.92 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.58it/s]Capturing num tokens (num_tokens=960 avail_mem=45.91 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.58it/s] Capturing num tokens (num_tokens=960 avail_mem=45.91 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.54it/s]Capturing num tokens (num_tokens=896 avail_mem=45.91 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.54it/s]Capturing num tokens (num_tokens=832 avail_mem=45.91 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.54it/s]Capturing num tokens (num_tokens=768 avail_mem=45.90 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.54it/s]

    Capturing num tokens (num_tokens=768 avail_mem=45.90 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.33it/s]Capturing num tokens (num_tokens=704 avail_mem=45.90 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.33it/s]Capturing num tokens (num_tokens=640 avail_mem=45.90 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.33it/s]Capturing num tokens (num_tokens=576 avail_mem=45.89 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.33it/s]Capturing num tokens (num_tokens=576 avail_mem=45.89 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.95it/s]Capturing num tokens (num_tokens=512 avail_mem=45.89 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.95it/s]Capturing num tokens (num_tokens=480 avail_mem=45.89 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.95it/s]Capturing num tokens (num_tokens=448 avail_mem=45.88 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.95it/s]

    Capturing num tokens (num_tokens=416 avail_mem=45.88 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.95it/s]Capturing num tokens (num_tokens=416 avail_mem=45.88 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]Capturing num tokens (num_tokens=384 avail_mem=45.88 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]Capturing num tokens (num_tokens=352 avail_mem=45.87 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]Capturing num tokens (num_tokens=320 avail_mem=45.87 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]Capturing num tokens (num_tokens=288 avail_mem=45.86 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]Capturing num tokens (num_tokens=288 avail_mem=45.86 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.95it/s]Capturing num tokens (num_tokens=256 avail_mem=45.86 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.95it/s]Capturing num tokens (num_tokens=240 avail_mem=45.85 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.95it/s]

    Capturing num tokens (num_tokens=224 avail_mem=45.85 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.95it/s]Capturing num tokens (num_tokens=208 avail_mem=45.85 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.95it/s]Capturing num tokens (num_tokens=208 avail_mem=45.85 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.72it/s]Capturing num tokens (num_tokens=192 avail_mem=45.84 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.72it/s]Capturing num tokens (num_tokens=176 avail_mem=45.84 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.72it/s]Capturing num tokens (num_tokens=160 avail_mem=45.84 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.72it/s]Capturing num tokens (num_tokens=144 avail_mem=45.83 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.72it/s]Capturing num tokens (num_tokens=144 avail_mem=45.83 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.90it/s]Capturing num tokens (num_tokens=128 avail_mem=45.84 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.90it/s]Capturing num tokens (num_tokens=112 avail_mem=45.84 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.90it/s]

    Capturing num tokens (num_tokens=96 avail_mem=45.83 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.90it/s] Capturing num tokens (num_tokens=80 avail_mem=45.83 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.90it/s]Capturing num tokens (num_tokens=80 avail_mem=45.83 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.69it/s]Capturing num tokens (num_tokens=64 avail_mem=45.83 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.69it/s]Capturing num tokens (num_tokens=48 avail_mem=45.83 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.69it/s]Capturing num tokens (num_tokens=32 avail_mem=45.82 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.69it/s]Capturing num tokens (num_tokens=28 avail_mem=45.82 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.69it/s]Capturing num tokens (num_tokens=28 avail_mem=45.82 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.77it/s]Capturing num tokens (num_tokens=24 avail_mem=45.82 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.77it/s]Capturing num tokens (num_tokens=20 avail_mem=45.81 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.77it/s]

    Capturing num tokens (num_tokens=16 avail_mem=45.81 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.77it/s]Capturing num tokens (num_tokens=12 avail_mem=45.80 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.77it/s]Capturing num tokens (num_tokens=12 avail_mem=45.80 GB):  97%|█████████▋| 56/58 [00:03<00:00, 35.27it/s]Capturing num tokens (num_tokens=8 avail_mem=45.80 GB):  97%|█████████▋| 56/58 [00:03<00:00, 35.27it/s] Capturing num tokens (num_tokens=4 avail_mem=45.80 GB):  97%|█████████▋| 56/58 [00:03<00:00, 35.27it/s]Capturing num tokens (num_tokens=4 avail_mem=45.80 GB): 100%|██████████| 58/58 [00:03<00:00, 14.73it/s]


    [2026-04-20 21:46:45] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I'll add the two numbers together to find the total.<br><br>Finally, I'll calculate the result, which is 4.<br></strong>



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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition by adding the two numbers together.<br><br>Finally, I arrive at the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Certainly! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Start with the number 1.**<br>2. **Add 3 to it.**<br>   <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I identify the two numbers that need to be added, which are 1 and 3.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum to determine that 1 plus 3 equals 4.</strong>



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



<strong style='color: #00008B;'>First, I identify the two numbers in the addition problem: 1 and 3.<br><br>Next, I add the units place of both numbers: 1 plus 3 equals 4.<br><br>Since there are no digits in the tens place, the final result is simply 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I recognize that I need to add the numbers 1 and 3.<br><br>Next, I start with the number 1 and add 3 to it.<br><br>Finally, I calculate the sum to get the result of 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that I need to add the numbers 1 and 3.<br><br>Next, I start with the number 1 and add 3 to it.<br><br>Finally, I calculate the sum to get the result of 4.<br></strong>



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

    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-20 21:47:07] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.51s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.51s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.51s/it]


    2026-04-20 21:47:17,250 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-20 21:47:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:56,  3.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:56,  3.10s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:25,  1.53s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:25,  1.53s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:50,  1.08it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:50,  1.08it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:34,  1.59it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:34,  1.59it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.17it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.17it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.53it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.53it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.30it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.30it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.16it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.16it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.16it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.80it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.80it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.80it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.28it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.28it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.28it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.82it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.82it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.82it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.65it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.65it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.65it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.65it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 14.75it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 14.75it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.09it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.09it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.09it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.09it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.09it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.09it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 27.11it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 27.11it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 27.11it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 27.11it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 27.11it/s]

    Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 27.11it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 27.11it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 34.74it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 34.74it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 34.74it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 34.74it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 34.74it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 34.74it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 34.74it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:00, 34.74it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 43.47it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 43.47it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 43.47it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 43.47it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 43.47it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 43.47it/s] 

    Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 43.47it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 43.47it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 50.39it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 50.39it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 50.39it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 50.39it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 50.39it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 50.39it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 50.39it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 50.39it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 50.39it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 50.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.87 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=45.87 GB):   2%|▏         | 1/58 [00:00<00:17,  3.32it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.83 GB):   2%|▏         | 1/58 [00:00<00:17,  3.32it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=45.83 GB):   3%|▎         | 2/58 [00:00<00:17,  3.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.83 GB):   3%|▎         | 2/58 [00:00<00:17,  3.21it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=45.83 GB):   5%|▌         | 3/58 [00:00<00:15,  3.52it/s]Capturing num tokens (num_tokens=6656 avail_mem=45.84 GB):   5%|▌         | 3/58 [00:00<00:15,  3.52it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=45.84 GB):   7%|▋         | 4/58 [00:01<00:13,  3.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.84 GB):   7%|▋         | 4/58 [00:01<00:13,  3.89it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=45.84 GB):   9%|▊         | 5/58 [00:01<00:12,  4.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.85 GB):   9%|▊         | 5/58 [00:01<00:12,  4.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.85 GB):  10%|█         | 6/58 [00:01<00:11,  4.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.85 GB):  10%|█         | 6/58 [00:01<00:11,  4.61it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=45.85 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.76it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.85 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.76it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=45.85 GB):  14%|█▍        | 8/58 [00:02<00:12,  3.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.79 GB):  14%|█▍        | 8/58 [00:02<00:12,  3.91it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=44.79 GB):  16%|█▌        | 9/58 [00:02<00:14,  3.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.29 GB):  16%|█▌        | 9/58 [00:02<00:14,  3.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.29 GB):  17%|█▋        | 10/58 [00:02<00:12,  3.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.50 GB):  17%|█▋        | 10/58 [00:02<00:12,  3.97it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=38.50 GB):  19%|█▉        | 11/58 [00:02<00:09,  4.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.50 GB):  19%|█▉        | 11/58 [00:02<00:09,  4.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.50 GB):  21%|██        | 12/58 [00:02<00:08,  5.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.50 GB):  21%|██        | 12/58 [00:02<00:08,  5.48it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=38.50 GB):  22%|██▏       | 13/58 [00:03<00:10,  4.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.50 GB):  22%|██▏       | 13/58 [00:03<00:10,  4.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.50 GB):  22%|██▏       | 13/58 [00:03<00:10,  4.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.50 GB):  26%|██▌       | 15/58 [00:03<00:06,  6.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.50 GB):  26%|██▌       | 15/58 [00:03<00:06,  6.21it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=38.50 GB):  26%|██▌       | 15/58 [00:03<00:06,  6.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.50 GB):  29%|██▉       | 17/58 [00:03<00:05,  7.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.50 GB):  29%|██▉       | 17/58 [00:03<00:05,  7.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.49 GB):  29%|██▉       | 17/58 [00:03<00:05,  7.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.49 GB):  33%|███▎      | 19/58 [00:03<00:03,  9.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.50 GB):  33%|███▎      | 19/58 [00:03<00:03,  9.90it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=38.50 GB):  33%|███▎      | 19/58 [00:03<00:03,  9.90it/s]Capturing num tokens (num_tokens=960 avail_mem=38.49 GB):  33%|███▎      | 19/58 [00:03<00:03,  9.90it/s] Capturing num tokens (num_tokens=960 avail_mem=38.49 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.18it/s]Capturing num tokens (num_tokens=896 avail_mem=38.49 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.18it/s]Capturing num tokens (num_tokens=832 avail_mem=38.49 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.18it/s]Capturing num tokens (num_tokens=768 avail_mem=38.48 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.18it/s]

    Capturing num tokens (num_tokens=768 avail_mem=38.48 GB):  43%|████▎     | 25/58 [00:03<00:02, 16.08it/s]Capturing num tokens (num_tokens=704 avail_mem=38.48 GB):  43%|████▎     | 25/58 [00:03<00:02, 16.08it/s]Capturing num tokens (num_tokens=640 avail_mem=38.47 GB):  43%|████▎     | 25/58 [00:03<00:02, 16.08it/s]Capturing num tokens (num_tokens=576 avail_mem=38.47 GB):  43%|████▎     | 25/58 [00:03<00:02, 16.08it/s]Capturing num tokens (num_tokens=576 avail_mem=38.47 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.30it/s]Capturing num tokens (num_tokens=512 avail_mem=38.46 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.30it/s]Capturing num tokens (num_tokens=480 avail_mem=38.46 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.30it/s]Capturing num tokens (num_tokens=448 avail_mem=38.46 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.30it/s]

    Capturing num tokens (num_tokens=416 avail_mem=38.46 GB):  48%|████▊     | 28/58 [00:04<00:01, 18.30it/s]Capturing num tokens (num_tokens=416 avail_mem=38.46 GB):  55%|█████▌    | 32/58 [00:04<00:01, 21.78it/s]Capturing num tokens (num_tokens=384 avail_mem=38.45 GB):  55%|█████▌    | 32/58 [00:04<00:01, 21.78it/s]Capturing num tokens (num_tokens=352 avail_mem=38.45 GB):  55%|█████▌    | 32/58 [00:04<00:01, 21.78it/s]Capturing num tokens (num_tokens=320 avail_mem=38.44 GB):  55%|█████▌    | 32/58 [00:04<00:01, 21.78it/s]Capturing num tokens (num_tokens=288 avail_mem=38.44 GB):  55%|█████▌    | 32/58 [00:04<00:01, 21.78it/s]Capturing num tokens (num_tokens=288 avail_mem=38.44 GB):  62%|██████▏   | 36/58 [00:04<00:00, 24.88it/s]Capturing num tokens (num_tokens=256 avail_mem=38.43 GB):  62%|██████▏   | 36/58 [00:04<00:00, 24.88it/s]Capturing num tokens (num_tokens=240 avail_mem=38.43 GB):  62%|██████▏   | 36/58 [00:04<00:00, 24.88it/s]

    Capturing num tokens (num_tokens=224 avail_mem=38.43 GB):  62%|██████▏   | 36/58 [00:04<00:00, 24.88it/s]Capturing num tokens (num_tokens=208 avail_mem=38.42 GB):  62%|██████▏   | 36/58 [00:04<00:00, 24.88it/s]Capturing num tokens (num_tokens=208 avail_mem=38.42 GB):  69%|██████▉   | 40/58 [00:04<00:00, 27.62it/s]Capturing num tokens (num_tokens=192 avail_mem=38.42 GB):  69%|██████▉   | 40/58 [00:04<00:00, 27.62it/s]Capturing num tokens (num_tokens=176 avail_mem=38.42 GB):  69%|██████▉   | 40/58 [00:04<00:00, 27.62it/s]Capturing num tokens (num_tokens=160 avail_mem=38.41 GB):  69%|██████▉   | 40/58 [00:04<00:00, 27.62it/s]Capturing num tokens (num_tokens=144 avail_mem=38.41 GB):  69%|██████▉   | 40/58 [00:04<00:00, 27.62it/s]Capturing num tokens (num_tokens=144 avail_mem=38.41 GB):  76%|███████▌  | 44/58 [00:04<00:00, 29.55it/s]Capturing num tokens (num_tokens=128 avail_mem=38.42 GB):  76%|███████▌  | 44/58 [00:04<00:00, 29.55it/s]

    Capturing num tokens (num_tokens=112 avail_mem=38.42 GB):  76%|███████▌  | 44/58 [00:04<00:00, 29.55it/s]Capturing num tokens (num_tokens=96 avail_mem=38.41 GB):  76%|███████▌  | 44/58 [00:04<00:00, 29.55it/s] Capturing num tokens (num_tokens=80 avail_mem=38.41 GB):  76%|███████▌  | 44/58 [00:04<00:00, 29.55it/s]Capturing num tokens (num_tokens=80 avail_mem=38.41 GB):  83%|████████▎ | 48/58 [00:04<00:00, 30.52it/s]Capturing num tokens (num_tokens=64 avail_mem=38.40 GB):  83%|████████▎ | 48/58 [00:04<00:00, 30.52it/s]Capturing num tokens (num_tokens=48 avail_mem=38.40 GB):  83%|████████▎ | 48/58 [00:04<00:00, 30.52it/s]Capturing num tokens (num_tokens=32 avail_mem=38.40 GB):  83%|████████▎ | 48/58 [00:04<00:00, 30.52it/s]Capturing num tokens (num_tokens=28 avail_mem=38.40 GB):  83%|████████▎ | 48/58 [00:04<00:00, 30.52it/s]

    Capturing num tokens (num_tokens=28 avail_mem=38.40 GB):  90%|████████▉ | 52/58 [00:04<00:00, 31.86it/s]Capturing num tokens (num_tokens=24 avail_mem=38.39 GB):  90%|████████▉ | 52/58 [00:04<00:00, 31.86it/s]Capturing num tokens (num_tokens=20 avail_mem=38.39 GB):  90%|████████▉ | 52/58 [00:04<00:00, 31.86it/s]Capturing num tokens (num_tokens=16 avail_mem=38.38 GB):  90%|████████▉ | 52/58 [00:04<00:00, 31.86it/s]Capturing num tokens (num_tokens=12 avail_mem=38.38 GB):  90%|████████▉ | 52/58 [00:04<00:00, 31.86it/s]Capturing num tokens (num_tokens=12 avail_mem=38.38 GB):  97%|█████████▋| 56/58 [00:04<00:00, 30.58it/s]Capturing num tokens (num_tokens=8 avail_mem=38.38 GB):  97%|█████████▋| 56/58 [00:04<00:00, 30.58it/s] Capturing num tokens (num_tokens=4 avail_mem=38.37 GB):  97%|█████████▋| 56/58 [00:04<00:00, 30.58it/s]

    Capturing num tokens (num_tokens=4 avail_mem=38.37 GB): 100%|██████████| 58/58 [00:04<00:00, 11.95it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition by combining the two numbers.<br><br>Finally, I arrive at the result, which is 4.<br></think><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\(\boxed{4}\)</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition by combining the two numbers.<br><br>Finally, I arrive at the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Sure! Let's solve the addition problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\(\boxed{4}\)</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
