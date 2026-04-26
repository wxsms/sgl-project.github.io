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
    [2026-04-26 17:14:26] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-26 17:14:27] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-26 17:14:28] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-26 17:14:30] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-26 17:14:36] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-26 17:14:37] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False
    [2026-04-26 17:14:37] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.11s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.19s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.18s/it]


    2026-04-26 17:14:44,611 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 17:14:44] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:06,  5.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:06,  5.38s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:13,  2.38s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:13,  2.38s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:21,  1.48s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:21,  1.48s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:52,  1.04it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:52,  1.04it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:25,  2.01it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:25,  2.01it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:19,  2.64it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:19,  2.64it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.38it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.38it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.22it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.22it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:07<00:11,  4.22it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:07,  5.92it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:07,  5.92it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:07,  5.92it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:05,  7.52it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:05,  7.52it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:05,  7.52it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.16it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.16it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.16it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.13it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.13it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.13it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.13it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.57it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.57it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.57it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.57it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.57it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]

    Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.06it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:07<00:00, 38.46it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:07<00:00, 38.46it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:07<00:00, 38.46it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:07<00:00, 38.46it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:07<00:00, 38.46it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:08<00:00, 38.46it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:08<00:00, 38.46it/s]

    Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:08<00:00, 38.46it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:08<00:00, 46.15it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:08<00:00, 46.15it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:08<00:00, 46.15it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:08<00:00, 46.15it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:08<00:00, 46.15it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:08<00:00, 46.15it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:08<00:00, 46.15it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:08<00:00, 46.15it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:08<00:00, 51.82it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:08<00:00, 51.82it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:08<00:00, 51.82it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:08<00:00, 51.82it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:08<00:00, 51.82it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:08<00:00, 51.82it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:08<00:00, 51.82it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:08<00:00, 51.82it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.04it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=91.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=91.20 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=89.29 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=89.29 GB):   3%|▎         | 2/58 [00:01<00:35,  1.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=89.29 GB):   3%|▎         | 2/58 [00:01<00:35,  1.60it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=89.29 GB):   5%|▌         | 3/58 [00:01<00:32,  1.71it/s]Capturing num tokens (num_tokens=6656 avail_mem=89.29 GB):   5%|▌         | 3/58 [00:01<00:32,  1.71it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=89.29 GB):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=89.29 GB):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=89.29 GB):   9%|▊         | 5/58 [00:02<00:26,  1.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=89.29 GB):   9%|▊         | 5/58 [00:02<00:26,  1.98it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=89.29 GB):  10%|█         | 6/58 [00:03<00:24,  2.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=89.28 GB):  10%|█         | 6/58 [00:03<00:24,  2.15it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=89.28 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=89.29 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.32it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=89.29 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=89.28 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.54it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=89.28 GB):  16%|█▌        | 9/58 [00:04<00:18,  2.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=89.28 GB):  16%|█▌        | 9/58 [00:04<00:18,  2.62it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=89.28 GB):  17%|█▋        | 10/58 [00:04<00:17,  2.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=89.28 GB):  17%|█▋        | 10/58 [00:04<00:17,  2.78it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=89.28 GB):  19%|█▉        | 11/58 [00:04<00:15,  3.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=89.28 GB):  19%|█▉        | 11/58 [00:04<00:15,  3.03it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=89.28 GB):  21%|██        | 12/58 [00:04<00:13,  3.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=89.28 GB):  21%|██        | 12/58 [00:04<00:13,  3.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=89.28 GB):  22%|██▏       | 13/58 [00:05<00:11,  3.99it/s]Capturing num tokens (num_tokens=2816 avail_mem=89.27 GB):  22%|██▏       | 13/58 [00:05<00:11,  3.99it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=89.27 GB):  24%|██▍       | 14/58 [00:05<00:09,  4.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=89.27 GB):  24%|██▍       | 14/58 [00:05<00:09,  4.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=89.27 GB):  26%|██▌       | 15/58 [00:05<00:08,  4.98it/s]Capturing num tokens (num_tokens=2304 avail_mem=98.34 GB):  26%|██▌       | 15/58 [00:05<00:08,  4.98it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=98.34 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.84it/s]Capturing num tokens (num_tokens=2048 avail_mem=98.34 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=98.34 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=98.34 GB):  31%|███       | 18/58 [00:05<00:04,  8.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=98.33 GB):  31%|███       | 18/58 [00:05<00:04,  8.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=98.33 GB):  31%|███       | 18/58 [00:05<00:04,  8.27it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=98.32 GB):  31%|███       | 18/58 [00:05<00:04,  8.27it/s]Capturing num tokens (num_tokens=1024 avail_mem=98.32 GB):  36%|███▌      | 21/58 [00:05<00:03, 11.93it/s]Capturing num tokens (num_tokens=960 avail_mem=98.31 GB):  36%|███▌      | 21/58 [00:05<00:03, 11.93it/s] Capturing num tokens (num_tokens=896 avail_mem=98.31 GB):  36%|███▌      | 21/58 [00:05<00:03, 11.93it/s]Capturing num tokens (num_tokens=832 avail_mem=98.31 GB):  36%|███▌      | 21/58 [00:05<00:03, 11.93it/s]Capturing num tokens (num_tokens=832 avail_mem=98.31 GB):  41%|████▏     | 24/58 [00:05<00:02, 15.42it/s]Capturing num tokens (num_tokens=768 avail_mem=98.30 GB):  41%|████▏     | 24/58 [00:05<00:02, 15.42it/s]

    Capturing num tokens (num_tokens=704 avail_mem=97.33 GB):  41%|████▏     | 24/58 [00:05<00:02, 15.42it/s]Capturing num tokens (num_tokens=704 avail_mem=97.33 GB):  45%|████▍     | 26/58 [00:06<00:02, 13.18it/s]Capturing num tokens (num_tokens=640 avail_mem=97.33 GB):  45%|████▍     | 26/58 [00:06<00:02, 13.18it/s]

    Capturing num tokens (num_tokens=576 avail_mem=97.32 GB):  45%|████▍     | 26/58 [00:06<00:02, 13.18it/s]Capturing num tokens (num_tokens=576 avail_mem=97.32 GB):  48%|████▊     | 28/58 [00:06<00:02, 11.91it/s]Capturing num tokens (num_tokens=512 avail_mem=98.26 GB):  48%|████▊     | 28/58 [00:06<00:02, 11.91it/s]Capturing num tokens (num_tokens=480 avail_mem=98.25 GB):  48%|████▊     | 28/58 [00:06<00:02, 11.91it/s]

    Capturing num tokens (num_tokens=480 avail_mem=98.25 GB):  52%|█████▏    | 30/58 [00:06<00:02, 11.70it/s]Capturing num tokens (num_tokens=448 avail_mem=97.43 GB):  52%|█████▏    | 30/58 [00:06<00:02, 11.70it/s]Capturing num tokens (num_tokens=416 avail_mem=97.43 GB):  52%|█████▏    | 30/58 [00:06<00:02, 11.70it/s]

    Capturing num tokens (num_tokens=416 avail_mem=97.43 GB):  55%|█████▌    | 32/58 [00:06<00:02, 11.07it/s]Capturing num tokens (num_tokens=384 avail_mem=98.25 GB):  55%|█████▌    | 32/58 [00:06<00:02, 11.07it/s]Capturing num tokens (num_tokens=352 avail_mem=98.24 GB):  55%|█████▌    | 32/58 [00:06<00:02, 11.07it/s]Capturing num tokens (num_tokens=352 avail_mem=98.24 GB):  59%|█████▊    | 34/58 [00:06<00:02, 11.62it/s]Capturing num tokens (num_tokens=320 avail_mem=97.47 GB):  59%|█████▊    | 34/58 [00:06<00:02, 11.62it/s]

    Capturing num tokens (num_tokens=288 avail_mem=97.47 GB):  59%|█████▊    | 34/58 [00:06<00:02, 11.62it/s]Capturing num tokens (num_tokens=288 avail_mem=97.47 GB):  62%|██████▏   | 36/58 [00:07<00:02, 10.95it/s]Capturing num tokens (num_tokens=256 avail_mem=97.47 GB):  62%|██████▏   | 36/58 [00:07<00:02, 10.95it/s]Capturing num tokens (num_tokens=240 avail_mem=98.23 GB):  62%|██████▏   | 36/58 [00:07<00:02, 10.95it/s]

    Capturing num tokens (num_tokens=240 avail_mem=98.23 GB):  66%|██████▌   | 38/58 [00:07<00:01, 11.19it/s]Capturing num tokens (num_tokens=224 avail_mem=97.52 GB):  66%|██████▌   | 38/58 [00:07<00:01, 11.19it/s]Capturing num tokens (num_tokens=208 avail_mem=97.51 GB):  66%|██████▌   | 38/58 [00:07<00:01, 11.19it/s]

    Capturing num tokens (num_tokens=208 avail_mem=97.51 GB):  69%|██████▉   | 40/58 [00:07<00:01, 10.74it/s]Capturing num tokens (num_tokens=192 avail_mem=97.51 GB):  69%|██████▉   | 40/58 [00:07<00:01, 10.74it/s]Capturing num tokens (num_tokens=176 avail_mem=98.22 GB):  69%|██████▉   | 40/58 [00:07<00:01, 10.74it/s]Capturing num tokens (num_tokens=176 avail_mem=98.22 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.27it/s]

    Capturing num tokens (num_tokens=160 avail_mem=97.56 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.27it/s]Capturing num tokens (num_tokens=144 avail_mem=98.21 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.27it/s]Capturing num tokens (num_tokens=144 avail_mem=98.21 GB):  76%|███████▌  | 44/58 [00:07<00:01,  9.04it/s]Capturing num tokens (num_tokens=128 avail_mem=97.61 GB):  76%|███████▌  | 44/58 [00:07<00:01,  9.04it/s]

    Capturing num tokens (num_tokens=112 avail_mem=97.60 GB):  76%|███████▌  | 44/58 [00:07<00:01,  9.04it/s]Capturing num tokens (num_tokens=112 avail_mem=97.60 GB):  79%|███████▉  | 46/58 [00:08<00:01,  9.71it/s]Capturing num tokens (num_tokens=96 avail_mem=98.20 GB):  79%|███████▉  | 46/58 [00:08<00:01,  9.71it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=102.14 GB):  79%|███████▉  | 46/58 [00:08<00:01,  9.71it/s]Capturing num tokens (num_tokens=80 avail_mem=102.14 GB):  83%|████████▎ | 48/58 [00:08<00:01,  9.36it/s]Capturing num tokens (num_tokens=64 avail_mem=102.68 GB):  83%|████████▎ | 48/58 [00:08<00:01,  9.36it/s]Capturing num tokens (num_tokens=48 avail_mem=102.19 GB):  83%|████████▎ | 48/58 [00:08<00:01,  9.36it/s]

    Capturing num tokens (num_tokens=48 avail_mem=102.19 GB):  86%|████████▌ | 50/58 [00:08<00:00,  9.96it/s]Capturing num tokens (num_tokens=32 avail_mem=102.68 GB):  86%|████████▌ | 50/58 [00:08<00:00,  9.96it/s]Capturing num tokens (num_tokens=28 avail_mem=102.24 GB):  86%|████████▌ | 50/58 [00:08<00:00,  9.96it/s]Capturing num tokens (num_tokens=28 avail_mem=102.24 GB):  90%|████████▉ | 52/58 [00:08<00:00, 10.76it/s]Capturing num tokens (num_tokens=24 avail_mem=102.23 GB):  90%|████████▉ | 52/58 [00:08<00:00, 10.76it/s]

    Capturing num tokens (num_tokens=20 avail_mem=102.67 GB):  90%|████████▉ | 52/58 [00:08<00:00, 10.76it/s]Capturing num tokens (num_tokens=20 avail_mem=102.67 GB):  93%|█████████▎| 54/58 [00:08<00:00, 11.57it/s]Capturing num tokens (num_tokens=16 avail_mem=102.25 GB):  93%|█████████▎| 54/58 [00:08<00:00, 11.57it/s]Capturing num tokens (num_tokens=12 avail_mem=102.66 GB):  93%|█████████▎| 54/58 [00:08<00:00, 11.57it/s]

    Capturing num tokens (num_tokens=12 avail_mem=102.66 GB):  97%|█████████▋| 56/58 [00:08<00:00, 12.22it/s]Capturing num tokens (num_tokens=8 avail_mem=102.27 GB):  97%|█████████▋| 56/58 [00:08<00:00, 12.22it/s] Capturing num tokens (num_tokens=4 avail_mem=102.27 GB):  97%|█████████▋| 56/58 [00:08<00:00, 12.22it/s]Capturing num tokens (num_tokens=4 avail_mem=102.27 GB): 100%|██████████| 58/58 [00:09<00:00, 12.68it/s]Capturing num tokens (num_tokens=4 avail_mem=102.27 GB): 100%|██████████| 58/58 [00:09<00:00,  6.44it/s]


    [2026-04-26 17:15:04] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I'll add the two numbers together.<br><br>After performing the addition, I find that the result is 4.<br><br>Therefore, the answer to 1 + 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We need to calculate the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I start by identifying the numbers involved, which are 1 and 3.<br><br>Next, I add these two numbers together to find the total.<br><br>Finally, I conclude that the sum of 1 and 3 is 4.<br></strong>



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



<strong style='color: #00008B;'>First, I identify the two numbers that need to be added: 1 and 3.<br><br>Next, I perform the addition of these two numbers.<br><br>Finally, I arrive at the sum, which is 4.</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \text{ and } 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I identify the two numbers involved in the addition, which are 1 and 3.<br><br>Next, I add these two numbers together.<br><br>The sum of 1 and 3 is 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>Certainly! Let's solve the addition problem step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**  <br>To find the sum of \(1\) and \(3\), follow these simple steps:<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\[<br>\boxed{4}<br>\]</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I identify the two numbers involved in the addition, which are 1 and 3.<br><br>Next, I add these two numbers together.<br><br>The sum of 1 and 3 is 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Certainly! Let's solve the addition problem step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**  <br>To find the sum of \(1\) and \(3\), follow these simple steps:<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\[<br>\boxed{4}<br>\]</strong>



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


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-26 17:15:29] `torch_dtype` is deprecated! Use `dtype` instead!
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.11s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.20s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.19s/it]


    2026-04-26 17:15:38,182 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 17:15:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:08,  5.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:08,  5.41s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:17,  2.45s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:17,  2.45s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:22,  1.49s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:22,  1.49s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:55,  1.03s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:55,  1.03s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:40,  1.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:40,  1.31it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:31,  1.67it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:24,  2.11it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:24,  2.11it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:19,  2.59it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:19,  2.59it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:16,  3.04it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:16,  3.04it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:14,  3.27it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:14,  3.27it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:13,  3.51it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:13,  3.51it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:11,  4.17it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:11,  4.17it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:09,  4.94it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:09,  4.94it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:07,  5.78it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:07,  5.78it/s]

    Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:08<00:07,  5.78it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:05,  7.45it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:05,  7.45it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:08<00:05,  7.45it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:08<00:04,  9.27it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:08<00:04,  9.27it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:08<00:04,  9.27it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:08<00:03, 11.39it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:08<00:03, 11.39it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:08<00:03, 11.39it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:09<00:03, 11.39it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:09<00:02, 14.97it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:09<00:02, 14.97it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:09<00:02, 14.97it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:09<00:02, 14.97it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:09<00:02, 14.97it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:09<00:01, 19.85it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:09<00:01, 19.85it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:09<00:01, 19.85it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:09<00:01, 19.85it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:09<00:01, 19.85it/s]

    Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:09<00:01, 19.85it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:09<00:00, 26.18it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:09<00:00, 26.18it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:09<00:00, 26.18it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:09<00:00, 26.18it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:09<00:00, 26.18it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:09<00:00, 26.18it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:09<00:00, 31.74it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:09<00:00, 31.74it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:09<00:00, 31.74it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:09<00:00, 31.74it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:09<00:00, 31.74it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:09<00:00, 31.74it/s]

    Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:09<00:00, 31.74it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:09<00:00, 38.41it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:09<00:00, 38.41it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:09<00:00, 38.41it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:09<00:00, 38.41it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:09<00:00, 38.41it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:09<00:00, 38.41it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:09<00:00, 38.41it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:09<00:00, 42.33it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:09<00:00, 42.33it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:09<00:00, 42.33it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:09<00:00, 42.33it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:09<00:00, 42.33it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:09<00:00, 42.33it/s]

    Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:09<00:00, 42.33it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:09<00:00, 42.33it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:09<00:00, 48.30it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:09<00:00, 48.30it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:09<00:00, 48.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  5.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=85.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=85.19 GB):   2%|▏         | 1/58 [00:00<00:33,  1.69it/s]Capturing num tokens (num_tokens=7680 avail_mem=85.16 GB):   2%|▏         | 1/58 [00:00<00:33,  1.69it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=85.16 GB):   3%|▎         | 2/58 [00:01<00:32,  1.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=85.15 GB):   3%|▎         | 2/58 [00:01<00:32,  1.75it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=85.15 GB):   5%|▌         | 3/58 [00:01<00:29,  1.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=85.15 GB):   5%|▌         | 3/58 [00:01<00:29,  1.84it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=85.15 GB):   7%|▋         | 4/58 [00:02<00:27,  1.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=85.15 GB):   7%|▋         | 4/58 [00:02<00:27,  1.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=85.15 GB):   9%|▊         | 5/58 [00:02<00:25,  2.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=85.15 GB):   9%|▊         | 5/58 [00:02<00:25,  2.06it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=85.15 GB):  10%|█         | 6/58 [00:02<00:23,  2.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=85.15 GB):  10%|█         | 6/58 [00:02<00:23,  2.22it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=85.15 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=85.15 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.39it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=85.15 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=85.15 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.61it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=85.15 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=85.15 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.82it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=85.15 GB):  17%|█▋        | 10/58 [00:04<00:16,  3.00it/s]Capturing num tokens (num_tokens=3584 avail_mem=85.15 GB):  17%|█▋        | 10/58 [00:04<00:16,  3.00it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=85.15 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=85.15 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.27it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=85.15 GB):  21%|██        | 12/58 [00:04<00:12,  3.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=85.14 GB):  21%|██        | 12/58 [00:04<00:12,  3.64it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=85.14 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=85.11 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=85.11 GB):  24%|██▍       | 14/58 [00:05<00:10,  4.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=85.11 GB):  24%|██▍       | 14/58 [00:05<00:10,  4.28it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=85.11 GB):  26%|██▌       | 15/58 [00:05<00:08,  4.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=85.10 GB):  26%|██▌       | 15/58 [00:05<00:08,  4.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=85.10 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=85.10 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.69it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=85.10 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=85.10 GB):  31%|███       | 18/58 [00:05<00:05,  7.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=85.09 GB):  31%|███       | 18/58 [00:05<00:05,  7.42it/s]Capturing num tokens (num_tokens=1280 avail_mem=85.09 GB):  31%|███       | 18/58 [00:05<00:05,  7.42it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=85.09 GB):  34%|███▍      | 20/58 [00:05<00:04,  9.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=85.08 GB):  34%|███▍      | 20/58 [00:05<00:04,  9.05it/s]Capturing num tokens (num_tokens=960 avail_mem=85.07 GB):  34%|███▍      | 20/58 [00:05<00:04,  9.05it/s] Capturing num tokens (num_tokens=896 avail_mem=85.07 GB):  34%|███▍      | 20/58 [00:05<00:04,  9.05it/s]Capturing num tokens (num_tokens=896 avail_mem=85.07 GB):  40%|███▉      | 23/58 [00:05<00:02, 12.91it/s]Capturing num tokens (num_tokens=832 avail_mem=85.07 GB):  40%|███▉      | 23/58 [00:05<00:02, 12.91it/s]Capturing num tokens (num_tokens=768 avail_mem=85.06 GB):  40%|███▉      | 23/58 [00:05<00:02, 12.91it/s]Capturing num tokens (num_tokens=704 avail_mem=85.06 GB):  40%|███▉      | 23/58 [00:05<00:02, 12.91it/s]

    Capturing num tokens (num_tokens=704 avail_mem=85.06 GB):  45%|████▍     | 26/58 [00:05<00:01, 16.49it/s]Capturing num tokens (num_tokens=640 avail_mem=85.06 GB):  45%|████▍     | 26/58 [00:05<00:01, 16.49it/s]Capturing num tokens (num_tokens=576 avail_mem=85.05 GB):  45%|████▍     | 26/58 [00:05<00:01, 16.49it/s]Capturing num tokens (num_tokens=512 avail_mem=85.05 GB):  45%|████▍     | 26/58 [00:05<00:01, 16.49it/s]Capturing num tokens (num_tokens=480 avail_mem=85.04 GB):  45%|████▍     | 26/58 [00:05<00:01, 16.49it/s]Capturing num tokens (num_tokens=480 avail_mem=85.04 GB):  52%|█████▏    | 30/58 [00:05<00:01, 21.01it/s]Capturing num tokens (num_tokens=448 avail_mem=85.04 GB):  52%|█████▏    | 30/58 [00:05<00:01, 21.01it/s]Capturing num tokens (num_tokens=416 avail_mem=85.04 GB):  52%|█████▏    | 30/58 [00:05<00:01, 21.01it/s]Capturing num tokens (num_tokens=384 avail_mem=85.03 GB):  52%|█████▏    | 30/58 [00:05<00:01, 21.01it/s]

    Capturing num tokens (num_tokens=352 avail_mem=85.03 GB):  52%|█████▏    | 30/58 [00:06<00:01, 21.01it/s]Capturing num tokens (num_tokens=352 avail_mem=85.03 GB):  59%|█████▊    | 34/58 [00:06<00:00, 24.79it/s]Capturing num tokens (num_tokens=320 avail_mem=85.02 GB):  59%|█████▊    | 34/58 [00:06<00:00, 24.79it/s]Capturing num tokens (num_tokens=288 avail_mem=85.03 GB):  59%|█████▊    | 34/58 [00:06<00:00, 24.79it/s]Capturing num tokens (num_tokens=256 avail_mem=84.05 GB):  59%|█████▊    | 34/58 [00:06<00:00, 24.79it/s]Capturing num tokens (num_tokens=256 avail_mem=84.05 GB):  64%|██████▍   | 37/58 [00:06<00:00, 22.16it/s]Capturing num tokens (num_tokens=240 avail_mem=84.05 GB):  64%|██████▍   | 37/58 [00:06<00:00, 22.16it/s]

    Capturing num tokens (num_tokens=224 avail_mem=84.05 GB):  64%|██████▍   | 37/58 [00:06<00:00, 22.16it/s]Capturing num tokens (num_tokens=208 avail_mem=85.64 GB):  64%|██████▍   | 37/58 [00:06<00:00, 22.16it/s]Capturing num tokens (num_tokens=208 avail_mem=85.64 GB):  69%|██████▉   | 40/58 [00:06<00:01, 17.62it/s]Capturing num tokens (num_tokens=192 avail_mem=84.98 GB):  69%|██████▉   | 40/58 [00:06<00:01, 17.62it/s]

    Capturing num tokens (num_tokens=176 avail_mem=84.92 GB):  69%|██████▉   | 40/58 [00:06<00:01, 17.62it/s]Capturing num tokens (num_tokens=160 avail_mem=84.15 GB):  69%|██████▉   | 40/58 [00:06<00:01, 17.62it/s]

    Capturing num tokens (num_tokens=160 avail_mem=84.15 GB):  74%|███████▍  | 43/58 [00:06<00:01, 14.97it/s]Capturing num tokens (num_tokens=144 avail_mem=84.15 GB):  74%|███████▍  | 43/58 [00:06<00:01, 14.97it/s]Capturing num tokens (num_tokens=128 avail_mem=84.97 GB):  74%|███████▍  | 43/58 [00:06<00:01, 14.97it/s]Capturing num tokens (num_tokens=128 avail_mem=84.97 GB):  78%|███████▊  | 45/58 [00:06<00:00, 14.50it/s]Capturing num tokens (num_tokens=112 avail_mem=84.97 GB):  78%|███████▊  | 45/58 [00:06<00:00, 14.50it/s]

    Capturing num tokens (num_tokens=96 avail_mem=84.20 GB):  78%|███████▊  | 45/58 [00:06<00:00, 14.50it/s] Capturing num tokens (num_tokens=96 avail_mem=84.20 GB):  81%|████████  | 47/58 [00:07<00:00, 13.90it/s]Capturing num tokens (num_tokens=80 avail_mem=84.19 GB):  81%|████████  | 47/58 [00:07<00:00, 13.90it/s]Capturing num tokens (num_tokens=64 avail_mem=84.19 GB):  81%|████████  | 47/58 [00:07<00:00, 13.90it/s]

    Capturing num tokens (num_tokens=64 avail_mem=84.19 GB):  84%|████████▍ | 49/58 [00:07<00:00, 13.04it/s]Capturing num tokens (num_tokens=48 avail_mem=84.95 GB):  84%|████████▍ | 49/58 [00:07<00:00, 13.04it/s]Capturing num tokens (num_tokens=32 avail_mem=84.23 GB):  84%|████████▍ | 49/58 [00:07<00:00, 13.04it/s]Capturing num tokens (num_tokens=32 avail_mem=84.23 GB):  88%|████████▊ | 51/58 [00:07<00:00, 12.51it/s]Capturing num tokens (num_tokens=28 avail_mem=84.24 GB):  88%|████████▊ | 51/58 [00:07<00:00, 12.51it/s]

    Capturing num tokens (num_tokens=24 avail_mem=84.23 GB):  88%|████████▊ | 51/58 [00:07<00:00, 12.51it/s]Capturing num tokens (num_tokens=24 avail_mem=84.23 GB):  91%|█████████▏| 53/58 [00:07<00:00, 11.79it/s]Capturing num tokens (num_tokens=20 avail_mem=84.94 GB):  91%|█████████▏| 53/58 [00:07<00:00, 11.79it/s]Capturing num tokens (num_tokens=16 avail_mem=84.28 GB):  91%|█████████▏| 53/58 [00:07<00:00, 11.79it/s]

    Capturing num tokens (num_tokens=16 avail_mem=84.28 GB):  95%|█████████▍| 55/58 [00:07<00:00, 11.38it/s]Capturing num tokens (num_tokens=12 avail_mem=84.27 GB):  95%|█████████▍| 55/58 [00:07<00:00, 11.38it/s]Capturing num tokens (num_tokens=8 avail_mem=84.93 GB):  95%|█████████▍| 55/58 [00:07<00:00, 11.38it/s] Capturing num tokens (num_tokens=8 avail_mem=84.93 GB):  98%|█████████▊| 57/58 [00:07<00:00, 11.55it/s]Capturing num tokens (num_tokens=4 avail_mem=84.32 GB):  98%|█████████▊| 57/58 [00:07<00:00, 11.55it/s]

    Capturing num tokens (num_tokens=4 avail_mem=84.32 GB): 100%|██████████| 58/58 [00:08<00:00,  7.18it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of the numbers 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of the numbers 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
