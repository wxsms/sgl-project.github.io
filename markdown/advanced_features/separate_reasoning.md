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
    [2026-04-23 04:27:48] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-23 04:27:50] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-23 04:27:52] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-23 04:27:53] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-23 04:27:59] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-23 04:28:00] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [2026-04-23 04:28:00] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.53s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.53s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.53s/it]


    2026-04-23 04:28:09,300 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 04:28:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:53,  3.05s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:53,  3.05s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:23,  1.50s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:23,  1.50s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.59it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.59it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.88it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.88it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:24,  2.13it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:24,  2.13it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:21,  2.42it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:21,  2.42it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:18,  2.70it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:18,  2.70it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:15,  3.10it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:15,  3.10it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:14,  3.43it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:14,  3.43it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:09,  4.85it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:09,  4.85it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:07,  5.86it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:07,  5.86it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:06,  6.45it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:06,  6.45it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:06<00:05,  7.09it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:06<00:05,  7.09it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:06<00:05,  7.09it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:04,  8.46it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:04,  8.46it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:04,  8.46it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:07<00:03, 10.38it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:07<00:03, 10.38it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:07<00:03, 10.38it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:02, 11.80it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:02, 11.80it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:02, 11.80it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:02, 11.80it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:07<00:02, 14.12it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:07<00:02, 14.12it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:07<00:02, 14.12it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:07<00:02, 14.12it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:01, 16.83it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:01, 16.83it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:01, 16.83it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:07<00:01, 17.52it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:07<00:01, 17.52it/s]

    Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:07<00:01, 17.52it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:07<00:01, 17.52it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:01, 19.96it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:01, 19.96it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:01, 19.96it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:01, 19.96it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:07<00:00, 22.12it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:07<00:00, 22.12it/s]

    Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:07<00:00, 22.12it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:07<00:00, 22.12it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:07<00:00, 23.68it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:07<00:00, 23.68it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:08<00:00, 23.68it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:08<00:00, 23.68it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:08<00:00, 24.66it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:08<00:00, 24.66it/s]

    Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:08<00:00, 24.66it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:08<00:00, 24.66it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:08<00:00, 26.06it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:08<00:00, 26.06it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:08<00:00, 26.06it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:08<00:00, 26.06it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:08<00:00, 25.21it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:08<00:00, 25.21it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:08<00:00, 25.21it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:08<00:00, 25.21it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:08<00:00, 26.19it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:08<00:00, 26.19it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:08<00:00, 26.19it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:08<00:00, 26.19it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:08<00:00, 26.19it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:08<00:00, 29.89it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:08<00:00, 29.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=23.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=23.09 GB):   2%|▏         | 1/58 [00:00<00:25,  2.25it/s]Capturing num tokens (num_tokens=7680 avail_mem=23.05 GB):   2%|▏         | 1/58 [00:00<00:25,  2.25it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=23.05 GB):   3%|▎         | 2/58 [00:00<00:21,  2.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=23.05 GB):   3%|▎         | 2/58 [00:00<00:21,  2.60it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=23.05 GB):   5%|▌         | 3/58 [00:01<00:18,  3.03it/s]Capturing num tokens (num_tokens=6656 avail_mem=22.02 GB):   5%|▌         | 3/58 [00:01<00:18,  3.03it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=22.02 GB):   7%|▋         | 4/58 [00:01<00:20,  2.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=23.69 GB):   7%|▋         | 4/58 [00:01<00:20,  2.59it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=23.69 GB):   9%|▊         | 5/58 [00:01<00:20,  2.57it/s]Capturing num tokens (num_tokens=5632 avail_mem=22.21 GB):   9%|▊         | 5/58 [00:01<00:20,  2.57it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=22.21 GB):  10%|█         | 6/58 [00:02<00:19,  2.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=23.04 GB):  10%|█         | 6/58 [00:02<00:19,  2.60it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=23.04 GB):  12%|█▏        | 7/58 [00:02<00:18,  2.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=22.28 GB):  12%|█▏        | 7/58 [00:02<00:18,  2.73it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=22.28 GB):  14%|█▍        | 8/58 [00:02<00:17,  2.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=23.05 GB):  14%|█▍        | 8/58 [00:02<00:17,  2.82it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=23.05 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=22.34 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.95it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=22.34 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=22.40 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.19it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=22.40 GB):  19%|█▉        | 11/58 [00:03<00:15,  2.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=22.45 GB):  19%|█▉        | 11/58 [00:03<00:15,  2.99it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=22.45 GB):  21%|██        | 12/58 [00:04<00:14,  3.26it/s]Capturing num tokens (num_tokens=3072 avail_mem=23.06 GB):  21%|██        | 12/58 [00:04<00:14,  3.26it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=23.06 GB):  22%|██▏       | 13/58 [00:04<00:13,  3.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=23.03 GB):  22%|██▏       | 13/58 [00:04<00:13,  3.32it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=23.03 GB):  24%|██▍       | 14/58 [00:04<00:12,  3.44it/s]Capturing num tokens (num_tokens=2560 avail_mem=22.54 GB):  24%|██▍       | 14/58 [00:04<00:12,  3.44it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=22.54 GB):  26%|██▌       | 15/58 [00:04<00:12,  3.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=22.59 GB):  26%|██▌       | 15/58 [00:04<00:12,  3.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=22.59 GB):  28%|██▊       | 16/58 [00:05<00:10,  4.05it/s]Capturing num tokens (num_tokens=2048 avail_mem=22.93 GB):  28%|██▊       | 16/58 [00:05<00:10,  4.05it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=22.93 GB):  29%|██▉       | 17/58 [00:05<00:09,  4.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=23.04 GB):  29%|██▉       | 17/58 [00:05<00:09,  4.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=23.04 GB):  31%|███       | 18/58 [00:05<00:08,  4.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=22.65 GB):  31%|███       | 18/58 [00:05<00:08,  4.90it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=22.65 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=22.67 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=22.67 GB):  34%|███▍      | 20/58 [00:05<00:06,  5.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=23.03 GB):  34%|███▍      | 20/58 [00:05<00:06,  5.95it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=23.03 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.60it/s]Capturing num tokens (num_tokens=960 avail_mem=22.70 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.60it/s] Capturing num tokens (num_tokens=896 avail_mem=22.97 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.60it/s]

    Capturing num tokens (num_tokens=896 avail_mem=22.97 GB):  40%|███▉      | 23/58 [00:06<00:04,  7.61it/s]Capturing num tokens (num_tokens=832 avail_mem=23.02 GB):  40%|███▉      | 23/58 [00:06<00:04,  7.61it/s]Capturing num tokens (num_tokens=768 avail_mem=22.74 GB):  40%|███▉      | 23/58 [00:06<00:04,  7.61it/s]Capturing num tokens (num_tokens=768 avail_mem=22.74 GB):  43%|████▎     | 25/58 [00:06<00:03,  8.93it/s]Capturing num tokens (num_tokens=704 avail_mem=23.01 GB):  43%|████▎     | 25/58 [00:06<00:03,  8.93it/s]

    Capturing num tokens (num_tokens=640 avail_mem=23.01 GB):  43%|████▎     | 25/58 [00:06<00:03,  8.93it/s]Capturing num tokens (num_tokens=640 avail_mem=23.01 GB):  47%|████▋     | 27/58 [00:06<00:03,  9.92it/s]Capturing num tokens (num_tokens=576 avail_mem=22.78 GB):  47%|████▋     | 27/58 [00:06<00:03,  9.92it/s]Capturing num tokens (num_tokens=512 avail_mem=23.00 GB):  47%|████▋     | 27/58 [00:06<00:03,  9.92it/s]

    Capturing num tokens (num_tokens=512 avail_mem=23.00 GB):  50%|█████     | 29/58 [00:06<00:02, 10.96it/s]Capturing num tokens (num_tokens=480 avail_mem=23.00 GB):  50%|█████     | 29/58 [00:06<00:02, 10.96it/s]Capturing num tokens (num_tokens=448 avail_mem=22.99 GB):  50%|█████     | 29/58 [00:06<00:02, 10.96it/s]

    Capturing num tokens (num_tokens=448 avail_mem=22.99 GB):  53%|█████▎    | 31/58 [00:06<00:02, 10.37it/s]Capturing num tokens (num_tokens=416 avail_mem=22.99 GB):  53%|█████▎    | 31/58 [00:06<00:02, 10.37it/s]Capturing num tokens (num_tokens=384 avail_mem=22.85 GB):  53%|█████▎    | 31/58 [00:06<00:02, 10.37it/s]Capturing num tokens (num_tokens=384 avail_mem=22.85 GB):  57%|█████▋    | 33/58 [00:06<00:02, 12.17it/s]Capturing num tokens (num_tokens=352 avail_mem=22.98 GB):  57%|█████▋    | 33/58 [00:06<00:02, 12.17it/s]Capturing num tokens (num_tokens=320 avail_mem=22.97 GB):  57%|█████▋    | 33/58 [00:06<00:02, 12.17it/s]

    Capturing num tokens (num_tokens=320 avail_mem=22.97 GB):  60%|██████    | 35/58 [00:06<00:01, 13.69it/s]Capturing num tokens (num_tokens=288 avail_mem=22.97 GB):  60%|██████    | 35/58 [00:06<00:01, 13.69it/s]Capturing num tokens (num_tokens=256 avail_mem=22.85 GB):  60%|██████    | 35/58 [00:06<00:01, 13.69it/s]Capturing num tokens (num_tokens=240 avail_mem=22.95 GB):  60%|██████    | 35/58 [00:07<00:01, 13.69it/s]Capturing num tokens (num_tokens=240 avail_mem=22.95 GB):  66%|██████▌   | 38/58 [00:07<00:01, 16.03it/s]Capturing num tokens (num_tokens=224 avail_mem=22.95 GB):  66%|██████▌   | 38/58 [00:07<00:01, 16.03it/s]Capturing num tokens (num_tokens=208 avail_mem=22.94 GB):  66%|██████▌   | 38/58 [00:07<00:01, 16.03it/s]

    Capturing num tokens (num_tokens=192 avail_mem=22.87 GB):  66%|██████▌   | 38/58 [00:07<00:01, 16.03it/s]Capturing num tokens (num_tokens=192 avail_mem=22.87 GB):  71%|███████   | 41/58 [00:07<00:00, 18.06it/s]Capturing num tokens (num_tokens=176 avail_mem=22.92 GB):  71%|███████   | 41/58 [00:07<00:00, 18.06it/s]Capturing num tokens (num_tokens=160 avail_mem=22.92 GB):  71%|███████   | 41/58 [00:07<00:00, 18.06it/s]Capturing num tokens (num_tokens=144 avail_mem=22.91 GB):  71%|███████   | 41/58 [00:07<00:00, 18.06it/s]Capturing num tokens (num_tokens=144 avail_mem=22.91 GB):  76%|███████▌  | 44/58 [00:07<00:00, 19.31it/s]Capturing num tokens (num_tokens=128 avail_mem=22.91 GB):  76%|███████▌  | 44/58 [00:07<00:00, 19.31it/s]

    Capturing num tokens (num_tokens=112 avail_mem=22.91 GB):  76%|███████▌  | 44/58 [00:07<00:00, 19.31it/s]Capturing num tokens (num_tokens=96 avail_mem=22.88 GB):  76%|███████▌  | 44/58 [00:07<00:00, 19.31it/s] Capturing num tokens (num_tokens=96 avail_mem=22.88 GB):  81%|████████  | 47/58 [00:07<00:00, 20.48it/s]Capturing num tokens (num_tokens=80 avail_mem=22.84 GB):  81%|████████  | 47/58 [00:07<00:00, 20.48it/s]Capturing num tokens (num_tokens=64 avail_mem=22.87 GB):  81%|████████  | 47/58 [00:07<00:00, 20.48it/s]Capturing num tokens (num_tokens=48 avail_mem=22.86 GB):  81%|████████  | 47/58 [00:07<00:00, 20.48it/s]

    Capturing num tokens (num_tokens=48 avail_mem=22.86 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.38it/s]Capturing num tokens (num_tokens=32 avail_mem=22.85 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.38it/s]Capturing num tokens (num_tokens=28 avail_mem=22.85 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.38it/s]Capturing num tokens (num_tokens=24 avail_mem=22.84 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.38it/s]Capturing num tokens (num_tokens=24 avail_mem=22.84 GB):  91%|█████████▏| 53/58 [00:07<00:00, 20.76it/s]Capturing num tokens (num_tokens=20 avail_mem=22.83 GB):  91%|█████████▏| 53/58 [00:07<00:00, 20.76it/s]

    Capturing num tokens (num_tokens=16 avail_mem=22.81 GB):  91%|█████████▏| 53/58 [00:07<00:00, 20.76it/s]Capturing num tokens (num_tokens=12 avail_mem=22.82 GB):  91%|█████████▏| 53/58 [00:07<00:00, 20.76it/s]Capturing num tokens (num_tokens=12 avail_mem=22.82 GB):  97%|█████████▋| 56/58 [00:07<00:00, 18.21it/s]Capturing num tokens (num_tokens=8 avail_mem=22.81 GB):  97%|█████████▋| 56/58 [00:07<00:00, 18.21it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=22.80 GB):  97%|█████████▋| 56/58 [00:08<00:00, 18.21it/s]Capturing num tokens (num_tokens=4 avail_mem=22.80 GB): 100%|██████████| 58/58 [00:08<00:00, 17.31it/s]Capturing num tokens (num_tokens=4 avail_mem=22.80 GB): 100%|██████████| 58/58 [00:08<00:00,  7.16it/s]


    [2026-04-23 04:28:28] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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



<strong style='color: #00008B;'>First, I identify the two numbers in the problem, which are 1 and 3.<br><br>Next, I add these two numbers together.<br><br>Finally, I calculate the sum, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I start by identifying the two numbers involved: 1 and 3.<br><br>Next, I add these two numbers together to find their total.<br><br>Finally, I conclude that the sum of 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition of the two numbers.<br><br>Finally, I arrive at the conclusion that the result of 1 plus 3 is 4.</strong>



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



<strong style='color: #00008B;'>I need to add the numbers 1 and 3 together.<br><br>First, I identify the two numbers to be added: 1 and 3.<br><br>Next, I perform the addition: 1 plus 3 equals 4.<br><br>Therefore, the sum of 1 and 3 is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I perform the addition of these two numbers.<br><br>Finally, I arrive at the result, which is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I perform the addition of these two numbers.<br><br>Finally, I arrive at the result, which is 4.<br></strong>



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
    [2026-04-23 04:28:50] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.58s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.55s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.56s/it]


    2026-04-23 04:29:05,630 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 04:29:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:02,  3.20s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:02,  3.20s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:27,  1.56s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:27,  1.56s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.14it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.14it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.77it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.77it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.48it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.27it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.27it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.14it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.14it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.14it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.81it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.81it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.81it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:05,  8.31it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:05,  8.31it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.31it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.88it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.88it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.88it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.76it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.76it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.76it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.76it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 14.93it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 14.93it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 14.93it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 14.93it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 14.93it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 27.53it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 27.53it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 27.53it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 27.53it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 27.53it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 27.53it/s]

    Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 27.53it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 35.35it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 35.35it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 35.35it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 35.35it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 35.35it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 35.35it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 35.35it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:00, 35.35it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 44.36it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 44.36it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 44.36it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 44.36it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 44.36it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 44.36it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 44.36it/s]

    Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 44.36it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 50.94it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 50.94it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 50.94it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 50.94it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 50.94it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 50.94it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 50.94it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 50.94it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:06<00:00, 50.94it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:06<00:00, 50.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00, 60.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.14 GB):   2%|▏         | 1/58 [00:00<00:21,  2.62it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.11 GB):   2%|▏         | 1/58 [00:00<00:21,  2.62it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.11 GB):   3%|▎         | 2/58 [00:00<00:17,  3.16it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.11 GB):   3%|▎         | 2/58 [00:00<00:17,  3.16it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.11 GB):   5%|▌         | 3/58 [00:00<00:15,  3.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.11 GB):   5%|▌         | 3/58 [00:00<00:15,  3.53it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.11 GB):   7%|▋         | 4/58 [00:01<00:17,  3.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.12 GB):   7%|▋         | 4/58 [00:01<00:17,  3.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.12 GB):   9%|▊         | 5/58 [00:01<00:19,  2.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.12 GB):   9%|▊         | 5/58 [00:01<00:19,  2.69it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=42.12 GB):  10%|█         | 6/58 [00:02<00:19,  2.64it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.12 GB):  10%|█         | 6/58 [00:02<00:19,  2.64it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.12 GB):  12%|█▏        | 7/58 [00:02<00:19,  2.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.13 GB):  12%|█▏        | 7/58 [00:02<00:19,  2.65it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=42.13 GB):  14%|█▍        | 8/58 [00:02<00:18,  2.75it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.13 GB):  14%|█▍        | 8/58 [00:02<00:18,  2.75it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.13 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.14 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.85it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=42.14 GB):  17%|█▋        | 10/58 [00:03<00:16,  2.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.13 GB):  17%|█▋        | 10/58 [00:03<00:16,  2.97it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.13 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.13 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.14it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=42.13 GB):  21%|██        | 12/58 [00:03<00:13,  3.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.13 GB):  21%|██        | 12/58 [00:03<00:13,  3.34it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=42.13 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.13 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.56it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=42.13 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.13 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.78it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=42.13 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.13 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.03it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=42.13 GB):  28%|██▊       | 16/58 [00:04<00:10,  4.00it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.13 GB):  28%|██▊       | 16/58 [00:04<00:10,  4.00it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.13 GB):  29%|██▉       | 17/58 [00:05<00:09,  4.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.13 GB):  29%|██▉       | 17/58 [00:05<00:09,  4.40it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=42.13 GB):  31%|███       | 18/58 [00:05<00:11,  3.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.13 GB):  31%|███       | 18/58 [00:05<00:11,  3.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.13 GB):  33%|███▎      | 19/58 [00:05<00:09,  4.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.13 GB):  33%|███▎      | 19/58 [00:05<00:09,  4.22it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=42.13 GB):  34%|███▍      | 20/58 [00:05<00:07,  5.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.13 GB):  34%|███▍      | 20/58 [00:05<00:07,  5.05it/s]Capturing num tokens (num_tokens=960 avail_mem=42.13 GB):  34%|███▍      | 20/58 [00:05<00:07,  5.05it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=42.13 GB):  38%|███▊      | 22/58 [00:06<00:05,  6.12it/s]Capturing num tokens (num_tokens=896 avail_mem=42.12 GB):  38%|███▊      | 22/58 [00:06<00:05,  6.12it/s]Capturing num tokens (num_tokens=832 avail_mem=42.12 GB):  38%|███▊      | 22/58 [00:06<00:05,  6.12it/s]Capturing num tokens (num_tokens=832 avail_mem=42.12 GB):  41%|████▏     | 24/58 [00:06<00:04,  8.09it/s]Capturing num tokens (num_tokens=768 avail_mem=42.11 GB):  41%|████▏     | 24/58 [00:06<00:04,  8.09it/s]

    Capturing num tokens (num_tokens=768 avail_mem=42.11 GB):  43%|████▎     | 25/58 [00:06<00:04,  8.17it/s]Capturing num tokens (num_tokens=704 avail_mem=42.10 GB):  43%|████▎     | 25/58 [00:06<00:04,  8.17it/s]Capturing num tokens (num_tokens=640 avail_mem=42.10 GB):  43%|████▎     | 25/58 [00:06<00:04,  8.17it/s]Capturing num tokens (num_tokens=640 avail_mem=42.10 GB):  47%|████▋     | 27/58 [00:06<00:03, 10.20it/s]Capturing num tokens (num_tokens=576 avail_mem=42.10 GB):  47%|████▋     | 27/58 [00:06<00:03, 10.20it/s]Capturing num tokens (num_tokens=512 avail_mem=42.09 GB):  47%|████▋     | 27/58 [00:06<00:03, 10.20it/s]Capturing num tokens (num_tokens=480 avail_mem=42.09 GB):  47%|████▋     | 27/58 [00:06<00:03, 10.20it/s]

    Capturing num tokens (num_tokens=448 avail_mem=42.09 GB):  47%|████▋     | 27/58 [00:06<00:03, 10.20it/s]Capturing num tokens (num_tokens=448 avail_mem=42.09 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.74it/s]Capturing num tokens (num_tokens=416 avail_mem=42.08 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.74it/s]Capturing num tokens (num_tokens=384 avail_mem=42.08 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.74it/s]Capturing num tokens (num_tokens=352 avail_mem=42.08 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.74it/s]

    Capturing num tokens (num_tokens=352 avail_mem=42.08 GB):  59%|█████▊    | 34/58 [00:06<00:01, 15.12it/s]Capturing num tokens (num_tokens=320 avail_mem=41.04 GB):  59%|█████▊    | 34/58 [00:06<00:01, 15.12it/s]Capturing num tokens (num_tokens=288 avail_mem=41.04 GB):  59%|█████▊    | 34/58 [00:06<00:01, 15.12it/s]

    Capturing num tokens (num_tokens=288 avail_mem=41.04 GB):  62%|██████▏   | 36/58 [00:06<00:01, 13.08it/s]Capturing num tokens (num_tokens=256 avail_mem=41.03 GB):  62%|██████▏   | 36/58 [00:06<00:01, 13.08it/s]Capturing num tokens (num_tokens=240 avail_mem=43.03 GB):  62%|██████▏   | 36/58 [00:07<00:01, 13.08it/s]Capturing num tokens (num_tokens=240 avail_mem=43.03 GB):  66%|██████▌   | 38/58 [00:07<00:01, 12.76it/s]Capturing num tokens (num_tokens=224 avail_mem=42.03 GB):  66%|██████▌   | 38/58 [00:07<00:01, 12.76it/s]

    Capturing num tokens (num_tokens=208 avail_mem=42.02 GB):  66%|██████▌   | 38/58 [00:07<00:01, 12.76it/s]Capturing num tokens (num_tokens=208 avail_mem=42.02 GB):  69%|██████▉   | 40/58 [00:07<00:01, 12.40it/s]Capturing num tokens (num_tokens=192 avail_mem=41.20 GB):  69%|██████▉   | 40/58 [00:07<00:01, 12.40it/s]

    Capturing num tokens (num_tokens=176 avail_mem=41.19 GB):  69%|██████▉   | 40/58 [00:07<00:01, 12.40it/s]Capturing num tokens (num_tokens=176 avail_mem=41.19 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.79it/s]Capturing num tokens (num_tokens=160 avail_mem=42.02 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.79it/s]Capturing num tokens (num_tokens=144 avail_mem=42.01 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.79it/s]

    Capturing num tokens (num_tokens=144 avail_mem=42.01 GB):  76%|███████▌  | 44/58 [00:07<00:01, 12.55it/s]Capturing num tokens (num_tokens=128 avail_mem=41.25 GB):  76%|███████▌  | 44/58 [00:07<00:01, 12.55it/s]Capturing num tokens (num_tokens=112 avail_mem=41.25 GB):  76%|███████▌  | 44/58 [00:07<00:01, 12.55it/s]

    Capturing num tokens (num_tokens=112 avail_mem=41.25 GB):  79%|███████▉  | 46/58 [00:07<00:01, 11.58it/s]Capturing num tokens (num_tokens=96 avail_mem=41.25 GB):  79%|███████▉  | 46/58 [00:07<00:01, 11.58it/s] Capturing num tokens (num_tokens=80 avail_mem=42.01 GB):  79%|███████▉  | 46/58 [00:07<00:01, 11.58it/s]Capturing num tokens (num_tokens=80 avail_mem=42.01 GB):  83%|████████▎ | 48/58 [00:07<00:00, 11.50it/s]Capturing num tokens (num_tokens=64 avail_mem=41.30 GB):  83%|████████▎ | 48/58 [00:07<00:00, 11.50it/s]

    Capturing num tokens (num_tokens=48 avail_mem=41.30 GB):  83%|████████▎ | 48/58 [00:08<00:00, 11.50it/s]Capturing num tokens (num_tokens=48 avail_mem=41.30 GB):  86%|████████▌ | 50/58 [00:08<00:00, 10.92it/s]Capturing num tokens (num_tokens=32 avail_mem=41.29 GB):  86%|████████▌ | 50/58 [00:08<00:00, 10.92it/s]Capturing num tokens (num_tokens=28 avail_mem=42.00 GB):  86%|████████▌ | 50/58 [00:08<00:00, 10.92it/s]

    Capturing num tokens (num_tokens=28 avail_mem=42.00 GB):  90%|████████▉ | 52/58 [00:08<00:00, 11.23it/s]Capturing num tokens (num_tokens=24 avail_mem=41.34 GB):  90%|████████▉ | 52/58 [00:08<00:00, 11.23it/s]Capturing num tokens (num_tokens=20 avail_mem=41.34 GB):  90%|████████▉ | 52/58 [00:08<00:00, 11.23it/s]Capturing num tokens (num_tokens=20 avail_mem=41.34 GB):  93%|█████████▎| 54/58 [00:08<00:00, 11.10it/s]Capturing num tokens (num_tokens=16 avail_mem=42.00 GB):  93%|█████████▎| 54/58 [00:08<00:00, 11.10it/s]

    Capturing num tokens (num_tokens=12 avail_mem=41.39 GB):  93%|█████████▎| 54/58 [00:08<00:00, 11.10it/s]Capturing num tokens (num_tokens=12 avail_mem=41.39 GB):  97%|█████████▋| 56/58 [00:08<00:00, 11.35it/s]Capturing num tokens (num_tokens=8 avail_mem=41.39 GB):  97%|█████████▋| 56/58 [00:08<00:00, 11.35it/s] Capturing num tokens (num_tokens=4 avail_mem=41.99 GB):  97%|█████████▋| 56/58 [00:08<00:00, 11.35it/s]

    Capturing num tokens (num_tokens=4 avail_mem=41.99 GB): 100%|██████████| 58/58 [00:08<00:00, 11.99it/s]Capturing num tokens (num_tokens=4 avail_mem=41.99 GB): 100%|██████████| 58/58 [00:08<00:00,  6.56it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I identify the two numbers in the problem: 1 and 3.<br><br>Next, I add these two numbers together.<br><br>Finally, I arrive at the sum of 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the answer is \(\boxed{4}\).</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I identify the two numbers in the problem: 1 and 3.<br><br>Next, I add these two numbers together.<br><br>Finally, I arrive at the sum of 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the answer is \(\boxed{4}\).</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
