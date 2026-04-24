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
    [2026-04-24 02:44:13] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 02:44:14] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-24 02:44:15] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-24 02:44:17] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 02:44:24] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-24 02:44:25] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [2026-04-24 02:44:25] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:03<00:03,  3.06s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:05<00:00,  2.55s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:05<00:00,  2.63s/it]


    2026-04-24 02:44:37,009 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 02:44:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:58,  3.13s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:58,  3.13s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:25,  1.53s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:25,  1.53s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:50,  1.09it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:50,  1.09it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.59it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.59it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.19it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.19it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.82it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.82it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.55it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.55it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.33it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.33it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.21it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.21it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.21it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.87it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.87it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.87it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.38it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.38it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.38it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.96it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.96it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.96it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.87it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.87it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.87it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.87it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.14it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.14it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.14it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.14it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.14it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.66it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.66it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.66it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.66it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.66it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.66it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:01, 20.66it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:00, 29.41it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:00, 29.41it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:00, 29.41it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:00, 29.41it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.41it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.41it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:00, 29.41it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 46.59it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 46.59it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 46.59it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 46.59it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 46.59it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 46.59it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 46.59it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 46.59it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 52.83it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 52.83it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 52.83it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 52.83it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 52.83it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 52.83it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 52.83it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 52.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.82it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.61 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=42.61 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.58 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.58 GB):   3%|▎         | 2/58 [00:00<00:15,  3.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.58 GB):   3%|▎         | 2/58 [00:00<00:15,  3.57it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.58 GB):   5%|▌         | 3/58 [00:00<00:14,  3.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.58 GB):   5%|▌         | 3/58 [00:00<00:14,  3.81it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.58 GB):   7%|▋         | 4/58 [00:01<00:13,  4.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.58 GB):   7%|▋         | 4/58 [00:01<00:13,  4.09it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.58 GB):   9%|▊         | 5/58 [00:01<00:12,  4.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.59 GB):   9%|▊         | 5/58 [00:01<00:12,  4.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.59 GB):  10%|█         | 6/58 [00:01<00:10,  4.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.59 GB):  10%|█         | 6/58 [00:01<00:10,  4.74it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.59 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.59 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.59 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.60 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.61it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.60 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.60 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.60 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.60 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.59it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.60 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.60 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.60 GB):  21%|██        | 12/58 [00:02<00:05,  7.72it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.60 GB):  21%|██        | 12/58 [00:02<00:05,  7.72it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=42.60 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.60 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.60 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.60 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.60 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.67it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=42.59 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.59 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.59 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.59 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.59 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.53it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.59 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.53it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=42.59 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.53it/s]Capturing num tokens (num_tokens=960 avail_mem=42.58 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.53it/s] Capturing num tokens (num_tokens=960 avail_mem=42.58 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.61it/s]Capturing num tokens (num_tokens=896 avail_mem=42.58 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.61it/s]Capturing num tokens (num_tokens=832 avail_mem=42.58 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.61it/s]Capturing num tokens (num_tokens=768 avail_mem=42.54 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.61it/s]Capturing num tokens (num_tokens=768 avail_mem=42.54 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.39it/s]Capturing num tokens (num_tokens=704 avail_mem=42.54 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.39it/s]

    Capturing num tokens (num_tokens=640 avail_mem=42.54 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.39it/s]Capturing num tokens (num_tokens=576 avail_mem=42.53 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.39it/s]Capturing num tokens (num_tokens=576 avail_mem=42.53 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.87it/s]Capturing num tokens (num_tokens=512 avail_mem=42.53 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.87it/s]Capturing num tokens (num_tokens=480 avail_mem=42.53 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.87it/s]Capturing num tokens (num_tokens=448 avail_mem=42.52 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.87it/s]Capturing num tokens (num_tokens=416 avail_mem=42.52 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.87it/s]Capturing num tokens (num_tokens=416 avail_mem=42.52 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.21it/s]Capturing num tokens (num_tokens=384 avail_mem=42.52 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.21it/s]

    Capturing num tokens (num_tokens=352 avail_mem=42.51 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.21it/s]Capturing num tokens (num_tokens=320 avail_mem=42.51 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.21it/s]Capturing num tokens (num_tokens=288 avail_mem=42.50 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.21it/s]Capturing num tokens (num_tokens=288 avail_mem=42.50 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.73it/s]Capturing num tokens (num_tokens=256 avail_mem=42.50 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.73it/s]Capturing num tokens (num_tokens=240 avail_mem=42.49 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.73it/s]Capturing num tokens (num_tokens=224 avail_mem=42.49 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.73it/s]

    Capturing num tokens (num_tokens=224 avail_mem=42.49 GB):  67%|██████▋   | 39/58 [00:03<00:00, 26.22it/s]Capturing num tokens (num_tokens=208 avail_mem=42.49 GB):  67%|██████▋   | 39/58 [00:03<00:00, 26.22it/s]Capturing num tokens (num_tokens=192 avail_mem=42.48 GB):  67%|██████▋   | 39/58 [00:03<00:00, 26.22it/s]Capturing num tokens (num_tokens=176 avail_mem=42.48 GB):  67%|██████▋   | 39/58 [00:03<00:00, 26.22it/s]Capturing num tokens (num_tokens=160 avail_mem=42.48 GB):  67%|██████▋   | 39/58 [00:03<00:00, 26.22it/s]Capturing num tokens (num_tokens=160 avail_mem=42.48 GB):  74%|███████▍  | 43/58 [00:03<00:00, 29.08it/s]Capturing num tokens (num_tokens=144 avail_mem=42.47 GB):  74%|███████▍  | 43/58 [00:03<00:00, 29.08it/s]Capturing num tokens (num_tokens=128 avail_mem=42.48 GB):  74%|███████▍  | 43/58 [00:03<00:00, 29.08it/s]Capturing num tokens (num_tokens=112 avail_mem=42.48 GB):  74%|███████▍  | 43/58 [00:03<00:00, 29.08it/s]Capturing num tokens (num_tokens=96 avail_mem=42.48 GB):  74%|███████▍  | 43/58 [00:03<00:00, 29.08it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=42.48 GB):  81%|████████  | 47/58 [00:03<00:00, 31.66it/s]Capturing num tokens (num_tokens=80 avail_mem=42.47 GB):  81%|████████  | 47/58 [00:03<00:00, 31.66it/s]Capturing num tokens (num_tokens=64 avail_mem=42.47 GB):  81%|████████  | 47/58 [00:03<00:00, 31.66it/s]Capturing num tokens (num_tokens=48 avail_mem=42.47 GB):  81%|████████  | 47/58 [00:03<00:00, 31.66it/s]Capturing num tokens (num_tokens=32 avail_mem=42.46 GB):  81%|████████  | 47/58 [00:03<00:00, 31.66it/s]Capturing num tokens (num_tokens=32 avail_mem=42.46 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.84it/s]Capturing num tokens (num_tokens=28 avail_mem=42.46 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.84it/s]Capturing num tokens (num_tokens=24 avail_mem=42.45 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.84it/s]Capturing num tokens (num_tokens=20 avail_mem=42.45 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.84it/s]Capturing num tokens (num_tokens=16 avail_mem=42.45 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.84it/s]

    Capturing num tokens (num_tokens=16 avail_mem=42.45 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.47it/s]Capturing num tokens (num_tokens=12 avail_mem=42.44 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.47it/s]Capturing num tokens (num_tokens=8 avail_mem=42.44 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.47it/s] Capturing num tokens (num_tokens=4 avail_mem=42.44 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.47it/s]Capturing num tokens (num_tokens=4 avail_mem=42.44 GB): 100%|██████████| 58/58 [00:03<00:00, 14.75it/s]


    [2026-04-24 02:44:48] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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



<strong style='color: #00008B;'>I recognize that the problem is asking for the sum of 1 and 3.<br><br>I start by identifying the two numbers involved in the addition: 1 and 3.<br><br>Next, I add these two numbers together to find the total.<br><br>Finally, I conclude that the sum of 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**  <br>   We have two numbers, 1 and 3.<br><br>2. **Add the numbers:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**  <br>   \[<br>   \boxed{4}<br>   \]<br><br>So, \(1 + 3 = \boxed{4}\).</strong>


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



<strong style='color: #00008B;'>I need to calculate the sum of 1 and 3.<br><br>First, I identify the two numbers involved in the addition: 1 and 3.<br><br>Next, I add these two numbers together.<br><br>Finally, I determine that the total is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**  <br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>To solve the problem \(1 + 3\), I start by identifying the two numbers involved, which are 1 and 3.<br><br>Next, I add these two numbers together.<br><br>Finally, the sum of 1 and 3 is 4.</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>To solve the problem \(1 + 3\), follow these simple steps:<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Present the final answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I arrive at the conclusion that the result of 1 plus 3 is 4.<br></think><br><br>**Solution:**<br><br>We need to find the sum of the numbers 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>I need to calculate the sum of 1 and 3.<br><br>First, I identify the two numbers involved in the addition, which are 1 and 3.<br><br>Next, I add these two numbers together.<br><br>Finally, the result of adding 1 and 3 is 4.<br></think><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**  <br>   - First number: \(1\)<br>   - Second number: \(3\)<br><br>2. **Add the numbers together:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**  <br>\(\boxed{4}\)</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>I need to calculate the sum of 1 and 3.<br><br>First, I identify the two numbers involved in the addition, which are 1 and 3.<br><br>Next, I add these two numbers together.<br><br>Finally, the result of adding 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Sure! Let's solve the addition problem step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**  <br>   - First number: \(1\)<br>   - Second number: \(3\)<br><br>2. **Add the numbers together:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**  <br>\(\boxed{4}\)</strong>



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
    [2026-04-24 02:45:17] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.06s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.02it/s]Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.00it/s]


    2026-04-24 02:45:26,318 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 02:45:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:59,  3.15s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:59,  3.15s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:41,  1.82s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:41,  1.82s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:53,  1.01it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:53,  1.01it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:31,  1.63it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:31,  1.63it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:25,  1.97it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:25,  1.97it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:21,  2.32it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:21,  2.32it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:18,  2.72it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:18,  2.72it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:15,  3.12it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:15,  3.12it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:13,  3.52it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:13,  3.52it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:11,  3.94it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:11,  3.94it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:10,  4.34it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:10,  4.34it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:09,  4.77it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:09,  4.77it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:08,  5.26it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:08,  5.26it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:07,  5.86it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:07,  5.86it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:06,  6.52it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:06,  6.52it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:08<00:06,  6.52it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:08<00:04,  8.82it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:08<00:04,  8.82it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:08<00:04,  8.82it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:08<00:04,  8.82it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:08<00:04,  8.82it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:08<00:02, 15.12it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:08<00:02, 15.12it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:08<00:02, 15.12it/s]

    Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:08<00:02, 15.12it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:08<00:02, 15.12it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:08<00:02, 15.12it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:08<00:01, 22.74it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:08<00:01, 22.74it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:08<00:01, 22.74it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:08<00:01, 22.74it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:08<00:01, 22.74it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:08<00:01, 22.74it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:08<00:01, 22.74it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:08<00:00, 31.92it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:08<00:00, 31.92it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:08<00:00, 31.92it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:08<00:00, 31.92it/s]

    Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:08<00:00, 31.92it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:08<00:00, 31.92it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:08<00:00, 31.92it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:08<00:00, 31.92it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:08<00:00, 41.81it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:08<00:00, 41.81it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:08<00:00, 41.81it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:08<00:00, 41.81it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:08<00:00, 41.81it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:08<00:00, 41.81it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:08<00:00, 41.81it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:08<00:00, 41.81it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:08<00:00, 41.81it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 50.25it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 50.25it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 50.25it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 50.25it/s]

    Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 50.25it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:08<00:00, 50.25it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:08<00:00, 50.25it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:08<00:00, 50.25it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:08<00:00, 50.25it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:08<00:00, 50.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.27 GB):   2%|▏         | 1/58 [00:00<00:27,  2.08it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.24 GB):   2%|▏         | 1/58 [00:00<00:27,  2.08it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.24 GB):   3%|▎         | 2/58 [00:01<00:29,  1.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.24 GB):   3%|▎         | 2/58 [00:01<00:29,  1.87it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.24 GB):   5%|▌         | 3/58 [00:01<00:29,  1.89it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.24 GB):   5%|▌         | 3/58 [00:01<00:29,  1.89it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.24 GB):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.25 GB):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.25 GB):   9%|▊         | 5/58 [00:02<00:25,  2.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.25 GB):   9%|▊         | 5/58 [00:02<00:25,  2.05it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.25 GB):  10%|█         | 6/58 [00:02<00:23,  2.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.25 GB):  10%|█         | 6/58 [00:02<00:23,  2.19it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.25 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.26 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.31it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.26 GB):  14%|█▍        | 8/58 [00:03<00:20,  2.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.26 GB):  14%|█▍        | 8/58 [00:03<00:20,  2.49it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.26 GB):  16%|█▌        | 9/58 [00:03<00:18,  2.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.26 GB):  16%|█▌        | 9/58 [00:03<00:18,  2.61it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.26 GB):  17%|█▋        | 10/58 [00:04<00:17,  2.79it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.26 GB):  17%|█▋        | 10/58 [00:04<00:17,  2.79it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.26 GB):  19%|█▉        | 11/58 [00:04<00:15,  2.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.26 GB):  19%|█▉        | 11/58 [00:04<00:15,  2.99it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=43.26 GB):  21%|██        | 12/58 [00:04<00:14,  3.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.26 GB):  21%|██        | 12/58 [00:04<00:14,  3.21it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=43.26 GB):  22%|██▏       | 13/58 [00:05<00:13,  3.43it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.26 GB):  22%|██▏       | 13/58 [00:05<00:13,  3.43it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.26 GB):  24%|██▍       | 14/58 [00:05<00:11,  3.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.26 GB):  24%|██▍       | 14/58 [00:05<00:11,  3.70it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=43.26 GB):  26%|██▌       | 15/58 [00:05<00:11,  3.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.26 GB):  26%|██▌       | 15/58 [00:05<00:11,  3.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.26 GB):  28%|██▊       | 16/58 [00:05<00:10,  4.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.26 GB):  28%|██▊       | 16/58 [00:05<00:10,  4.11it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=43.26 GB):  29%|██▉       | 17/58 [00:05<00:09,  4.42it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.26 GB):  29%|██▉       | 17/58 [00:05<00:09,  4.42it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=43.26 GB):  31%|███       | 18/58 [00:06<00:08,  4.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.26 GB):  31%|███       | 18/58 [00:06<00:08,  4.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.26 GB):  33%|███▎      | 19/58 [00:06<00:07,  4.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.26 GB):  33%|███▎      | 19/58 [00:06<00:07,  4.96it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=43.26 GB):  34%|███▍      | 20/58 [00:06<00:07,  5.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.26 GB):  34%|███▍      | 20/58 [00:06<00:07,  5.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.26 GB):  36%|███▌      | 21/58 [00:06<00:06,  5.87it/s]Capturing num tokens (num_tokens=960 avail_mem=43.25 GB):  36%|███▌      | 21/58 [00:06<00:06,  5.87it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=43.25 GB):  38%|███▊      | 22/58 [00:06<00:05,  6.17it/s]Capturing num tokens (num_tokens=896 avail_mem=43.25 GB):  38%|███▊      | 22/58 [00:06<00:05,  6.17it/s]Capturing num tokens (num_tokens=896 avail_mem=43.25 GB):  40%|███▉      | 23/58 [00:06<00:05,  6.71it/s]Capturing num tokens (num_tokens=832 avail_mem=43.25 GB):  40%|███▉      | 23/58 [00:06<00:05,  6.71it/s]

    Capturing num tokens (num_tokens=832 avail_mem=43.25 GB):  41%|████▏     | 24/58 [00:06<00:04,  7.12it/s]Capturing num tokens (num_tokens=768 avail_mem=43.24 GB):  41%|████▏     | 24/58 [00:06<00:04,  7.12it/s]Capturing num tokens (num_tokens=768 avail_mem=43.24 GB):  43%|████▎     | 25/58 [00:07<00:04,  7.56it/s]Capturing num tokens (num_tokens=704 avail_mem=43.24 GB):  43%|████▎     | 25/58 [00:07<00:04,  7.56it/s]

    Capturing num tokens (num_tokens=704 avail_mem=43.24 GB):  45%|████▍     | 26/58 [00:07<00:04,  7.90it/s]Capturing num tokens (num_tokens=640 avail_mem=43.24 GB):  45%|████▍     | 26/58 [00:07<00:04,  7.90it/s]Capturing num tokens (num_tokens=640 avail_mem=43.24 GB):  47%|████▋     | 27/58 [00:07<00:03,  8.20it/s]Capturing num tokens (num_tokens=576 avail_mem=43.23 GB):  47%|████▋     | 27/58 [00:07<00:03,  8.20it/s]

    Capturing num tokens (num_tokens=576 avail_mem=43.23 GB):  48%|████▊     | 28/58 [00:07<00:03,  8.40it/s]Capturing num tokens (num_tokens=512 avail_mem=43.23 GB):  48%|████▊     | 28/58 [00:07<00:03,  8.40it/s]Capturing num tokens (num_tokens=512 avail_mem=43.23 GB):  50%|█████     | 29/58 [00:07<00:03,  8.82it/s]Capturing num tokens (num_tokens=480 avail_mem=43.23 GB):  50%|█████     | 29/58 [00:07<00:03,  8.82it/s]Capturing num tokens (num_tokens=448 avail_mem=43.22 GB):  50%|█████     | 29/58 [00:07<00:03,  8.82it/s]

    Capturing num tokens (num_tokens=448 avail_mem=43.22 GB):  53%|█████▎    | 31/58 [00:07<00:02,  9.08it/s]Capturing num tokens (num_tokens=416 avail_mem=43.22 GB):  53%|█████▎    | 31/58 [00:07<00:02,  9.08it/s]Capturing num tokens (num_tokens=384 avail_mem=43.22 GB):  53%|█████▎    | 31/58 [00:07<00:02,  9.08it/s]Capturing num tokens (num_tokens=384 avail_mem=43.22 GB):  57%|█████▋    | 33/58 [00:07<00:02,  9.48it/s]Capturing num tokens (num_tokens=352 avail_mem=43.21 GB):  57%|█████▋    | 33/58 [00:07<00:02,  9.48it/s]

    Capturing num tokens (num_tokens=320 avail_mem=43.21 GB):  57%|█████▋    | 33/58 [00:07<00:02,  9.48it/s]Capturing num tokens (num_tokens=320 avail_mem=43.21 GB):  60%|██████    | 35/58 [00:08<00:02,  9.85it/s]Capturing num tokens (num_tokens=288 avail_mem=43.20 GB):  60%|██████    | 35/58 [00:08<00:02,  9.85it/s]Capturing num tokens (num_tokens=256 avail_mem=43.20 GB):  60%|██████    | 35/58 [00:08<00:02,  9.85it/s]

    Capturing num tokens (num_tokens=256 avail_mem=43.20 GB):  64%|██████▍   | 37/58 [00:08<00:02, 10.17it/s]Capturing num tokens (num_tokens=240 avail_mem=43.19 GB):  64%|██████▍   | 37/58 [00:08<00:02, 10.17it/s]Capturing num tokens (num_tokens=224 avail_mem=43.19 GB):  64%|██████▍   | 37/58 [00:08<00:02, 10.17it/s]Capturing num tokens (num_tokens=224 avail_mem=43.19 GB):  67%|██████▋   | 39/58 [00:08<00:01, 10.65it/s]Capturing num tokens (num_tokens=208 avail_mem=43.19 GB):  67%|██████▋   | 39/58 [00:08<00:01, 10.65it/s]

    Capturing num tokens (num_tokens=192 avail_mem=43.18 GB):  67%|██████▋   | 39/58 [00:08<00:01, 10.65it/s]Capturing num tokens (num_tokens=192 avail_mem=43.18 GB):  71%|███████   | 41/58 [00:08<00:01, 11.89it/s]Capturing num tokens (num_tokens=176 avail_mem=43.18 GB):  71%|███████   | 41/58 [00:08<00:01, 11.89it/s]Capturing num tokens (num_tokens=160 avail_mem=43.18 GB):  71%|███████   | 41/58 [00:08<00:01, 11.89it/s]

    Capturing num tokens (num_tokens=160 avail_mem=43.18 GB):  74%|███████▍  | 43/58 [00:08<00:01, 12.75it/s]Capturing num tokens (num_tokens=144 avail_mem=43.17 GB):  74%|███████▍  | 43/58 [00:08<00:01, 12.75it/s]Capturing num tokens (num_tokens=128 avail_mem=43.18 GB):  74%|███████▍  | 43/58 [00:08<00:01, 12.75it/s]Capturing num tokens (num_tokens=128 avail_mem=43.18 GB):  78%|███████▊  | 45/58 [00:08<00:01, 12.79it/s]Capturing num tokens (num_tokens=112 avail_mem=43.18 GB):  78%|███████▊  | 45/58 [00:08<00:01, 12.79it/s]

    Capturing num tokens (num_tokens=96 avail_mem=43.17 GB):  78%|███████▊  | 45/58 [00:08<00:01, 12.79it/s] Capturing num tokens (num_tokens=96 avail_mem=43.17 GB):  81%|████████  | 47/58 [00:09<00:00, 12.57it/s]Capturing num tokens (num_tokens=80 avail_mem=43.17 GB):  81%|████████  | 47/58 [00:09<00:00, 12.57it/s]Capturing num tokens (num_tokens=64 avail_mem=43.17 GB):  81%|████████  | 47/58 [00:09<00:00, 12.57it/s]

    Capturing num tokens (num_tokens=64 avail_mem=43.17 GB):  84%|████████▍ | 49/58 [00:09<00:00, 12.43it/s]Capturing num tokens (num_tokens=48 avail_mem=43.16 GB):  84%|████████▍ | 49/58 [00:09<00:00, 12.43it/s]Capturing num tokens (num_tokens=32 avail_mem=43.16 GB):  84%|████████▍ | 49/58 [00:09<00:00, 12.43it/s]Capturing num tokens (num_tokens=32 avail_mem=43.16 GB):  88%|████████▊ | 51/58 [00:09<00:00, 12.40it/s]Capturing num tokens (num_tokens=28 avail_mem=43.16 GB):  88%|████████▊ | 51/58 [00:09<00:00, 12.40it/s]

    Capturing num tokens (num_tokens=24 avail_mem=43.15 GB):  88%|████████▊ | 51/58 [00:09<00:00, 12.40it/s]Capturing num tokens (num_tokens=24 avail_mem=43.15 GB):  91%|█████████▏| 53/58 [00:09<00:00, 12.18it/s]Capturing num tokens (num_tokens=20 avail_mem=43.15 GB):  91%|█████████▏| 53/58 [00:09<00:00, 12.18it/s]Capturing num tokens (num_tokens=16 avail_mem=43.15 GB):  91%|█████████▏| 53/58 [00:09<00:00, 12.18it/s]

    Capturing num tokens (num_tokens=16 avail_mem=43.15 GB):  95%|█████████▍| 55/58 [00:09<00:00, 12.01it/s]Capturing num tokens (num_tokens=12 avail_mem=43.14 GB):  95%|█████████▍| 55/58 [00:09<00:00, 12.01it/s]Capturing num tokens (num_tokens=8 avail_mem=43.14 GB):  95%|█████████▍| 55/58 [00:09<00:00, 12.01it/s] Capturing num tokens (num_tokens=8 avail_mem=43.14 GB):  98%|█████████▊| 57/58 [00:09<00:00, 12.15it/s]Capturing num tokens (num_tokens=4 avail_mem=43.14 GB):  98%|█████████▊| 57/58 [00:09<00:00, 12.15it/s]

    Capturing num tokens (num_tokens=4 avail_mem=43.14 GB): 100%|██████████| 58/58 [00:09<00:00,  5.85it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum to determine the result of 1 plus 3.<br></think><br><br>Sure, let's solve the addition problem step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**  <br>To find the sum of 1 and 3, follow these steps:<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\(\boxed{4}\)</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum to determine the result of 1 plus 3.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Sure, let's solve the addition problem step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**  <br>To find the sum of 1 and 3, follow these steps:<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\(\boxed{4}\)</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
