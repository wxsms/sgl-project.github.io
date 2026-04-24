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
    [2026-04-24 00:32:36] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 00:32:38] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-24 00:32:38] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-24 00:32:40] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 00:32:46] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-24 00:32:47] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [2026-04-24 00:32:48] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.48s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.52s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.51s/it]


    2026-04-24 00:32:56,064 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 00:32:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:56,  3.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:56,  3.10s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:39,  1.78s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:39,  1.78s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:33,  1.54it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:33,  1.54it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:29,  1.74it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:29,  1.74it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:25,  1.94it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:25,  1.94it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:22,  2.17it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:22,  2.17it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:19,  2.44it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:19,  2.44it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:17,  2.66it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:17,  2.66it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:15,  2.94it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:15,  2.94it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:14,  3.20it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:14,  3.20it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:12,  3.51it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:12,  3.51it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:11,  3.86it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:11,  3.86it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:09,  4.24it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:09,  4.24it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:08<00:08,  4.74it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:08<00:08,  4.74it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:08<00:07,  5.25it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:08<00:07,  5.25it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:09<00:06,  5.80it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:09<00:06,  5.80it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:05,  6.50it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:05,  6.50it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:05,  6.50it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:09<00:04,  8.14it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:09<00:04,  8.14it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:09<00:04,  8.14it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:09<00:03,  9.39it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:09<00:03,  9.39it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:09<00:03,  9.39it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:09<00:02, 10.72it/s]

    Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:09<00:02, 10.72it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:09<00:02, 10.72it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:09<00:02, 12.15it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:09<00:02, 12.15it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:09<00:02, 12.15it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:09<00:02, 13.85it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:09<00:02, 13.85it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:09<00:02, 13.85it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:09<00:01, 15.23it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:09<00:01, 15.23it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:10<00:01, 15.23it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:10<00:01, 15.23it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:10<00:01, 16.99it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:10<00:01, 16.99it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:10<00:01, 16.99it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:10<00:01, 16.99it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:10<00:01, 19.47it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:10<00:01, 19.47it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:10<00:01, 19.47it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:10<00:01, 19.47it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:10<00:00, 21.16it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:10<00:00, 21.16it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:10<00:00, 21.16it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:10<00:00, 21.16it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:10<00:00, 21.80it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:10<00:00, 21.80it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:10<00:00, 21.80it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:10<00:00, 21.80it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:10<00:00, 23.75it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:10<00:00, 23.75it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:10<00:00, 23.75it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:10<00:00, 23.75it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:10<00:00, 24.14it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:10<00:00, 24.14it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:10<00:00, 24.14it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:10<00:00, 24.14it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:10<00:00, 25.41it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:10<00:00, 25.41it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:10<00:00, 25.41it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:10<00:00, 25.41it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:10<00:00, 25.41it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:10<00:00, 27.77it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:10<00:00, 27.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=84.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=84.12 GB):   2%|▏         | 1/58 [00:00<00:41,  1.38it/s]Capturing num tokens (num_tokens=7680 avail_mem=84.26 GB):   2%|▏         | 1/58 [00:00<00:41,  1.38it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=84.26 GB):   3%|▎         | 2/58 [00:01<00:37,  1.50it/s]Capturing num tokens (num_tokens=7168 avail_mem=84.32 GB):   3%|▎         | 2/58 [00:01<00:37,  1.50it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=84.32 GB):   5%|▌         | 3/58 [00:01<00:32,  1.69it/s]Capturing num tokens (num_tokens=6656 avail_mem=83.36 GB):   5%|▌         | 3/58 [00:01<00:32,  1.69it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=83.36 GB):   7%|▋         | 4/58 [00:02<00:33,  1.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=83.44 GB):   7%|▋         | 4/58 [00:02<00:33,  1.60it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=83.44 GB):   9%|▊         | 5/58 [00:03<00:32,  1.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=83.69 GB):   9%|▊         | 5/58 [00:03<00:32,  1.63it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=83.69 GB):  10%|█         | 6/58 [00:03<00:30,  1.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=84.58 GB):  10%|█         | 6/58 [00:03<00:30,  1.69it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=84.58 GB):  12%|█▏        | 7/58 [00:04<00:28,  1.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=83.89 GB):  12%|█▏        | 7/58 [00:04<00:28,  1.79it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=83.89 GB):  14%|█▍        | 8/58 [00:04<00:26,  1.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=84.74 GB):  14%|█▍        | 8/58 [00:04<00:26,  1.90it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=84.74 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=84.06 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.01it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=84.06 GB):  17%|█▋        | 10/58 [00:05<00:22,  2.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=84.15 GB):  17%|█▋        | 10/58 [00:05<00:22,  2.14it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=84.15 GB):  19%|█▉        | 11/58 [00:05<00:20,  2.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=84.84 GB):  19%|█▉        | 11/58 [00:05<00:20,  2.27it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=84.84 GB):  21%|██        | 12/58 [00:06<00:18,  2.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=84.27 GB):  21%|██        | 12/58 [00:06<00:18,  2.46it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=84.27 GB):  22%|██▏       | 13/58 [00:06<00:17,  2.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=84.91 GB):  22%|██▏       | 13/58 [00:06<00:17,  2.64it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=84.91 GB):  24%|██▍       | 14/58 [00:06<00:15,  2.80it/s]Capturing num tokens (num_tokens=2560 avail_mem=84.40 GB):  24%|██▍       | 14/58 [00:06<00:15,  2.80it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=84.40 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=84.49 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.12it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=84.49 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=85.02 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=85.02 GB):  29%|██▉       | 17/58 [00:07<00:10,  3.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=84.61 GB):  29%|██▉       | 17/58 [00:07<00:10,  3.77it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=84.61 GB):  31%|███       | 18/58 [00:07<00:09,  4.18it/s]Capturing num tokens (num_tokens=1536 avail_mem=85.07 GB):  31%|███       | 18/58 [00:07<00:09,  4.18it/s]Capturing num tokens (num_tokens=1536 avail_mem=85.07 GB):  33%|███▎      | 19/58 [00:07<00:08,  4.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=84.69 GB):  33%|███▎      | 19/58 [00:07<00:08,  4.41it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=84.69 GB):  34%|███▍      | 20/58 [00:07<00:07,  5.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=84.88 GB):  34%|███▍      | 20/58 [00:07<00:07,  5.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=84.88 GB):  36%|███▌      | 21/58 [00:08<00:06,  5.53it/s]Capturing num tokens (num_tokens=960 avail_mem=84.77 GB):  36%|███▌      | 21/58 [00:08<00:06,  5.53it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=84.77 GB):  38%|███▊      | 22/58 [00:08<00:05,  6.11it/s]Capturing num tokens (num_tokens=896 avail_mem=84.79 GB):  38%|███▊      | 22/58 [00:08<00:05,  6.11it/s]Capturing num tokens (num_tokens=896 avail_mem=84.79 GB):  40%|███▉      | 23/58 [00:08<00:05,  6.54it/s]Capturing num tokens (num_tokens=832 avail_mem=85.15 GB):  40%|███▉      | 23/58 [00:08<00:05,  6.54it/s]

    Capturing num tokens (num_tokens=832 avail_mem=85.15 GB):  41%|████▏     | 24/58 [00:08<00:04,  7.20it/s]Capturing num tokens (num_tokens=768 avail_mem=84.82 GB):  41%|████▏     | 24/58 [00:08<00:04,  7.20it/s]Capturing num tokens (num_tokens=768 avail_mem=84.82 GB):  43%|████▎     | 25/58 [00:08<00:04,  7.59it/s]Capturing num tokens (num_tokens=704 avail_mem=85.24 GB):  43%|████▎     | 25/58 [00:08<00:04,  7.59it/s]

    Capturing num tokens (num_tokens=704 avail_mem=85.24 GB):  45%|████▍     | 26/58 [00:08<00:04,  7.99it/s]Capturing num tokens (num_tokens=640 avail_mem=84.84 GB):  45%|████▍     | 26/58 [00:08<00:04,  7.99it/s]Capturing num tokens (num_tokens=640 avail_mem=84.84 GB):  47%|████▋     | 27/58 [00:08<00:03,  8.49it/s]Capturing num tokens (num_tokens=576 avail_mem=85.15 GB):  47%|████▋     | 27/58 [00:08<00:03,  8.49it/s]

    Capturing num tokens (num_tokens=576 avail_mem=85.15 GB):  48%|████▊     | 28/58 [00:08<00:03,  8.76it/s]Capturing num tokens (num_tokens=512 avail_mem=84.90 GB):  48%|████▊     | 28/58 [00:08<00:03,  8.76it/s]Capturing num tokens (num_tokens=480 avail_mem=85.19 GB):  48%|████▊     | 28/58 [00:08<00:03,  8.76it/s]Capturing num tokens (num_tokens=480 avail_mem=85.19 GB):  52%|█████▏    | 30/58 [00:09<00:02,  9.82it/s]Capturing num tokens (num_tokens=448 avail_mem=85.16 GB):  52%|█████▏    | 30/58 [00:09<00:02,  9.82it/s]

    Capturing num tokens (num_tokens=416 avail_mem=84.98 GB):  52%|█████▏    | 30/58 [00:09<00:02,  9.82it/s]Capturing num tokens (num_tokens=416 avail_mem=84.98 GB):  55%|█████▌    | 32/58 [00:09<00:02, 10.99it/s]Capturing num tokens (num_tokens=384 avail_mem=85.17 GB):  55%|█████▌    | 32/58 [00:09<00:02, 10.99it/s]Capturing num tokens (num_tokens=352 avail_mem=85.16 GB):  55%|█████▌    | 32/58 [00:09<00:02, 10.99it/s]

    Capturing num tokens (num_tokens=352 avail_mem=85.16 GB):  59%|█████▊    | 34/58 [00:09<00:02, 11.71it/s]Capturing num tokens (num_tokens=320 avail_mem=85.00 GB):  59%|█████▊    | 34/58 [00:09<00:02, 11.71it/s]Capturing num tokens (num_tokens=288 avail_mem=85.16 GB):  59%|█████▊    | 34/58 [00:09<00:02, 11.71it/s]Capturing num tokens (num_tokens=288 avail_mem=85.16 GB):  62%|██████▏   | 36/58 [00:09<00:01, 12.80it/s]Capturing num tokens (num_tokens=256 avail_mem=85.16 GB):  62%|██████▏   | 36/58 [00:09<00:01, 12.80it/s]Capturing num tokens (num_tokens=240 avail_mem=85.15 GB):  62%|██████▏   | 36/58 [00:09<00:01, 12.80it/s]

    Capturing num tokens (num_tokens=240 avail_mem=85.15 GB):  66%|██████▌   | 38/58 [00:09<00:01, 13.93it/s]Capturing num tokens (num_tokens=224 avail_mem=85.02 GB):  66%|██████▌   | 38/58 [00:09<00:01, 13.93it/s]Capturing num tokens (num_tokens=208 avail_mem=85.12 GB):  66%|██████▌   | 38/58 [00:09<00:01, 13.93it/s]Capturing num tokens (num_tokens=208 avail_mem=85.12 GB):  69%|██████▉   | 40/58 [00:09<00:01, 15.33it/s]Capturing num tokens (num_tokens=192 avail_mem=85.11 GB):  69%|██████▉   | 40/58 [00:09<00:01, 15.33it/s]Capturing num tokens (num_tokens=176 avail_mem=85.10 GB):  69%|██████▉   | 40/58 [00:09<00:01, 15.33it/s]

    Capturing num tokens (num_tokens=176 avail_mem=85.10 GB):  72%|███████▏  | 42/58 [00:09<00:01, 15.85it/s]Capturing num tokens (num_tokens=160 avail_mem=85.10 GB):  72%|███████▏  | 42/58 [00:09<00:01, 15.85it/s]Capturing num tokens (num_tokens=144 avail_mem=85.03 GB):  72%|███████▏  | 42/58 [00:09<00:01, 15.85it/s]Capturing num tokens (num_tokens=144 avail_mem=85.03 GB):  76%|███████▌  | 44/58 [00:09<00:00, 16.39it/s]Capturing num tokens (num_tokens=128 avail_mem=85.04 GB):  76%|███████▌  | 44/58 [00:09<00:00, 16.39it/s]Capturing num tokens (num_tokens=112 avail_mem=85.03 GB):  76%|███████▌  | 44/58 [00:09<00:00, 16.39it/s]

    Capturing num tokens (num_tokens=112 avail_mem=85.03 GB):  79%|███████▉  | 46/58 [00:10<00:00, 17.03it/s]Capturing num tokens (num_tokens=96 avail_mem=85.02 GB):  79%|███████▉  | 46/58 [00:10<00:00, 17.03it/s] Capturing num tokens (num_tokens=80 avail_mem=85.03 GB):  79%|███████▉  | 46/58 [00:10<00:00, 17.03it/s]Capturing num tokens (num_tokens=64 avail_mem=85.05 GB):  79%|███████▉  | 46/58 [00:10<00:00, 17.03it/s]Capturing num tokens (num_tokens=64 avail_mem=85.05 GB):  84%|████████▍ | 49/58 [00:10<00:00, 18.09it/s]Capturing num tokens (num_tokens=48 avail_mem=85.04 GB):  84%|████████▍ | 49/58 [00:10<00:00, 18.09it/s]

    Capturing num tokens (num_tokens=32 avail_mem=85.02 GB):  84%|████████▍ | 49/58 [00:10<00:00, 18.09it/s]Capturing num tokens (num_tokens=32 avail_mem=85.02 GB):  88%|████████▊ | 51/58 [00:10<00:00, 18.52it/s]Capturing num tokens (num_tokens=28 avail_mem=85.02 GB):  88%|████████▊ | 51/58 [00:10<00:00, 18.52it/s]Capturing num tokens (num_tokens=24 avail_mem=85.01 GB):  88%|████████▊ | 51/58 [00:10<00:00, 18.52it/s]Capturing num tokens (num_tokens=20 avail_mem=84.99 GB):  88%|████████▊ | 51/58 [00:10<00:00, 18.52it/s]Capturing num tokens (num_tokens=20 avail_mem=84.99 GB):  93%|█████████▎| 54/58 [00:10<00:00, 19.27it/s]Capturing num tokens (num_tokens=16 avail_mem=84.98 GB):  93%|█████████▎| 54/58 [00:10<00:00, 19.27it/s]

    Capturing num tokens (num_tokens=12 avail_mem=84.97 GB):  93%|█████████▎| 54/58 [00:10<00:00, 19.27it/s]Capturing num tokens (num_tokens=8 avail_mem=84.94 GB):  93%|█████████▎| 54/58 [00:10<00:00, 19.27it/s] Capturing num tokens (num_tokens=8 avail_mem=84.94 GB):  98%|█████████▊| 57/58 [00:10<00:00, 20.23it/s]Capturing num tokens (num_tokens=4 avail_mem=84.95 GB):  98%|█████████▊| 57/58 [00:10<00:00, 20.23it/s]Capturing num tokens (num_tokens=4 avail_mem=84.95 GB): 100%|██████████| 58/58 [00:10<00:00,  5.47it/s]


    [2026-04-24 00:33:19] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I perform the addition operation by adding these two numbers together.<br><br>Finally, I calculate that 1 plus 3 equals 4.<br></strong>



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



<strong style='color: #00008B;'>I need to calculate the sum of 1 and 3.<br><br>First, I recognize that adding 1 and 3 involves combining these two numbers.<br><br>By performing the addition, I find that 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>To find the sum of 1 and 3, follow these simple steps:<br><br>1. **Start with the first number:**<br>   <br>   \[<br>   1<br>   \]<br><br>2. **Add the second number to the first:**<br>   <br>   \[<br>   1 + 3<br>   \]<br><br>3. **Calculate the sum:**<br>   <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br><br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add the two numbers together to find their total.<br><br>Finally, I conclude that the sum of 1 and 3 is 4.</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Final Answer:**  <br>\(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>To solve the problem 1 + 3, I start by identifying the two numbers involved, which are 1 and 3.<br><br>Next, I add these two numbers together.<br><br>Finally, the sum of 1 and 3 is 4.<br></think><br><br>To solve the problem \(1 + 3\), follow these simple steps:<br><br>1. **Start with the first number:**<br>   \[<br>   1<br>   \]<br><br>2. **Add the second number:**<br>   \[<br>   1 + 3<br>   \]<br><br>3. **Calculate the sum:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I need to add the numbers 1 and 3.<br><br>Adding 1 and 3 together gives me 4.<br><br>So, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I need to add the numbers 1 and 3.<br><br>Adding 1 and 3 together gives me 4.<br><br>So, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>



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
    [2026-04-24 00:33:45] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.05s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.17s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.15s/it]


    2026-04-24 00:33:53,682 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 00:33:53] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:54,  3.06s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:54,  3.06s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:31,  1.64s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:31,  1.64s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:56,  1.03s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:56,  1.03s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:39,  1.37it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:39,  1.37it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:29,  1.83it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:29,  1.83it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:22,  2.35it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:22,  2.35it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:14,  3.50it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:14,  3.50it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:11,  4.18it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:11,  4.18it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:09,  4.90it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:09,  4.90it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:08,  5.61it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:08,  5.61it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:07,  6.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:07,  6.37it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  7.14it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  7.14it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  7.14it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  8.61it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  8.61it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  8.61it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04, 10.14it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04, 10.14it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04, 10.14it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:06<00:03, 12.17it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:06<00:03, 12.17it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:06<00:03, 12.17it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:06<00:03, 12.17it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:06<00:02, 15.66it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:06<00:02, 15.66it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:06<00:02, 15.66it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:06<00:02, 15.66it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:06<00:02, 15.66it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:06<00:01, 20.35it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:06<00:01, 20.35it/s]

    Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:06<00:01, 20.35it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:06<00:01, 20.35it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:06<00:01, 20.35it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:06<00:01, 20.35it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:06<00:00, 27.28it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:06<00:00, 27.28it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:06<00:00, 27.28it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:06<00:00, 27.28it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:06<00:00, 27.28it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:06<00:00, 27.28it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:06<00:00, 32.74it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:06<00:00, 32.74it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:06<00:00, 32.74it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:06<00:00, 32.74it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:06<00:00, 32.74it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:06<00:00, 32.74it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:06<00:00, 32.74it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:06<00:00, 32.74it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 40.82it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 40.82it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 40.82it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 40.82it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 40.82it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:06<00:00, 40.82it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:06<00:00, 40.82it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 45.01it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 45.01it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 45.01it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 45.01it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:06<00:00, 45.01it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:06<00:00, 45.01it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:06<00:00, 45.01it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 49.09it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 49.09it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 49.09it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 49.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=86.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=86.17 GB):   2%|▏         | 1/58 [00:00<00:16,  3.50it/s]Capturing num tokens (num_tokens=7680 avail_mem=86.14 GB):   2%|▏         | 1/58 [00:00<00:16,  3.50it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=86.14 GB):   3%|▎         | 2/58 [00:00<00:15,  3.62it/s]Capturing num tokens (num_tokens=7168 avail_mem=86.14 GB):   3%|▎         | 2/58 [00:00<00:15,  3.62it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=86.14 GB):   5%|▌         | 3/58 [00:00<00:14,  3.83it/s]Capturing num tokens (num_tokens=6656 avail_mem=86.14 GB):   5%|▌         | 3/58 [00:00<00:14,  3.83it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=86.14 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=86.14 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=86.14 GB):   9%|▊         | 5/58 [00:01<00:14,  3.57it/s]Capturing num tokens (num_tokens=5632 avail_mem=85.15 GB):   9%|▊         | 5/58 [00:01<00:14,  3.57it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=85.15 GB):  10%|█         | 6/58 [00:01<00:19,  2.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=85.09 GB):  10%|█         | 6/58 [00:01<00:19,  2.67it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=85.09 GB):  12%|█▏        | 7/58 [00:02<00:21,  2.35it/s]Capturing num tokens (num_tokens=4608 avail_mem=85.10 GB):  12%|█▏        | 7/58 [00:02<00:21,  2.35it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=85.10 GB):  14%|█▍        | 8/58 [00:02<00:22,  2.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=85.10 GB):  14%|█▍        | 8/58 [00:02<00:22,  2.25it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=85.10 GB):  16%|█▌        | 9/58 [00:03<00:21,  2.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=85.11 GB):  16%|█▌        | 9/58 [00:03<00:21,  2.25it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=85.11 GB):  17%|█▋        | 10/58 [00:03<00:21,  2.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=85.11 GB):  17%|█▋        | 10/58 [00:03<00:21,  2.27it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=85.11 GB):  19%|█▉        | 11/58 [00:04<00:19,  2.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=85.11 GB):  19%|█▉        | 11/58 [00:04<00:19,  2.44it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=85.11 GB):  21%|██        | 12/58 [00:04<00:18,  2.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=102.75 GB):  21%|██        | 12/58 [00:04<00:18,  2.51it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=102.75 GB):  22%|██▏       | 13/58 [00:04<00:15,  2.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=102.76 GB):  22%|██▏       | 13/58 [00:04<00:15,  2.85it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=102.76 GB):  24%|██▍       | 14/58 [00:05<00:14,  3.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=102.76 GB):  24%|██▍       | 14/58 [00:05<00:14,  3.12it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=102.76 GB):  26%|██▌       | 15/58 [00:05<00:12,  3.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=102.76 GB):  26%|██▌       | 15/58 [00:05<00:12,  3.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=102.76 GB):  28%|██▊       | 16/58 [00:05<00:10,  3.86it/s]Capturing num tokens (num_tokens=2048 avail_mem=102.75 GB):  28%|██▊       | 16/58 [00:05<00:10,  3.86it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=102.75 GB):  29%|██▉       | 17/58 [00:05<00:09,  4.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=102.76 GB):  29%|██▉       | 17/58 [00:05<00:09,  4.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=102.76 GB):  31%|███       | 18/58 [00:05<00:08,  4.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=102.75 GB):  31%|███       | 18/58 [00:05<00:08,  4.71it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=102.75 GB):  33%|███▎      | 19/58 [00:05<00:07,  5.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=102.75 GB):  33%|███▎      | 19/58 [00:05<00:07,  5.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=102.75 GB):  34%|███▍      | 20/58 [00:06<00:06,  5.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=102.75 GB):  34%|███▍      | 20/58 [00:06<00:06,  5.61it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=102.75 GB):  36%|███▌      | 21/58 [00:06<00:05,  6.26it/s]Capturing num tokens (num_tokens=960 avail_mem=102.75 GB):  36%|███▌      | 21/58 [00:06<00:05,  6.26it/s] Capturing num tokens (num_tokens=960 avail_mem=102.75 GB):  38%|███▊      | 22/58 [00:06<00:05,  7.03it/s]Capturing num tokens (num_tokens=896 avail_mem=102.74 GB):  38%|███▊      | 22/58 [00:06<00:05,  7.03it/s]

    Capturing num tokens (num_tokens=896 avail_mem=102.74 GB):  40%|███▉      | 23/58 [00:06<00:04,  7.66it/s]Capturing num tokens (num_tokens=832 avail_mem=102.74 GB):  40%|███▉      | 23/58 [00:06<00:04,  7.66it/s]Capturing num tokens (num_tokens=768 avail_mem=102.74 GB):  40%|███▉      | 23/58 [00:06<00:04,  7.66it/s]Capturing num tokens (num_tokens=768 avail_mem=102.74 GB):  43%|████▎     | 25/58 [00:06<00:03,  9.26it/s]Capturing num tokens (num_tokens=704 avail_mem=102.73 GB):  43%|████▎     | 25/58 [00:06<00:03,  9.26it/s]

    Capturing num tokens (num_tokens=640 avail_mem=102.73 GB):  43%|████▎     | 25/58 [00:06<00:03,  9.26it/s]Capturing num tokens (num_tokens=640 avail_mem=102.73 GB):  47%|████▋     | 27/58 [00:06<00:02, 11.12it/s]Capturing num tokens (num_tokens=576 avail_mem=102.73 GB):  47%|████▋     | 27/58 [00:06<00:02, 11.12it/s]Capturing num tokens (num_tokens=512 avail_mem=102.72 GB):  47%|████▋     | 27/58 [00:06<00:02, 11.12it/s]Capturing num tokens (num_tokens=480 avail_mem=102.72 GB):  47%|████▋     | 27/58 [00:06<00:02, 11.12it/s]Capturing num tokens (num_tokens=480 avail_mem=102.72 GB):  52%|█████▏    | 30/58 [00:06<00:01, 14.29it/s]Capturing num tokens (num_tokens=448 avail_mem=102.71 GB):  52%|█████▏    | 30/58 [00:06<00:01, 14.29it/s]

    Capturing num tokens (num_tokens=416 avail_mem=102.71 GB):  52%|█████▏    | 30/58 [00:06<00:01, 14.29it/s]Capturing num tokens (num_tokens=384 avail_mem=102.70 GB):  52%|█████▏    | 30/58 [00:06<00:01, 14.29it/s]Capturing num tokens (num_tokens=384 avail_mem=102.70 GB):  57%|█████▋    | 33/58 [00:06<00:01, 17.04it/s]Capturing num tokens (num_tokens=352 avail_mem=102.70 GB):  57%|█████▋    | 33/58 [00:06<00:01, 17.04it/s]Capturing num tokens (num_tokens=320 avail_mem=102.69 GB):  57%|█████▋    | 33/58 [00:06<00:01, 17.04it/s]Capturing num tokens (num_tokens=288 avail_mem=102.69 GB):  57%|█████▋    | 33/58 [00:06<00:01, 17.04it/s]Capturing num tokens (num_tokens=288 avail_mem=102.69 GB):  62%|██████▏   | 36/58 [00:07<00:01, 19.40it/s]Capturing num tokens (num_tokens=256 avail_mem=102.69 GB):  62%|██████▏   | 36/58 [00:07<00:01, 19.40it/s]

    Capturing num tokens (num_tokens=240 avail_mem=102.68 GB):  62%|██████▏   | 36/58 [00:07<00:01, 19.40it/s]Capturing num tokens (num_tokens=224 avail_mem=102.68 GB):  62%|██████▏   | 36/58 [00:07<00:01, 19.40it/s]Capturing num tokens (num_tokens=208 avail_mem=102.67 GB):  62%|██████▏   | 36/58 [00:07<00:01, 19.40it/s]Capturing num tokens (num_tokens=208 avail_mem=102.67 GB):  69%|██████▉   | 40/58 [00:07<00:00, 23.55it/s]Capturing num tokens (num_tokens=192 avail_mem=102.67 GB):  69%|██████▉   | 40/58 [00:07<00:00, 23.55it/s]Capturing num tokens (num_tokens=176 avail_mem=102.67 GB):  69%|██████▉   | 40/58 [00:07<00:00, 23.55it/s]Capturing num tokens (num_tokens=160 avail_mem=102.67 GB):  69%|██████▉   | 40/58 [00:07<00:00, 23.55it/s]Capturing num tokens (num_tokens=144 avail_mem=102.66 GB):  69%|██████▉   | 40/58 [00:07<00:00, 23.55it/s]Capturing num tokens (num_tokens=144 avail_mem=102.66 GB):  76%|███████▌  | 44/58 [00:07<00:00, 26.79it/s]Capturing num tokens (num_tokens=128 avail_mem=102.67 GB):  76%|███████▌  | 44/58 [00:07<00:00, 26.79it/s]

    Capturing num tokens (num_tokens=112 avail_mem=102.67 GB):  76%|███████▌  | 44/58 [00:07<00:00, 26.79it/s]Capturing num tokens (num_tokens=96 avail_mem=102.66 GB):  76%|███████▌  | 44/58 [00:07<00:00, 26.79it/s] Capturing num tokens (num_tokens=80 avail_mem=102.66 GB):  76%|███████▌  | 44/58 [00:07<00:00, 26.79it/s]Capturing num tokens (num_tokens=80 avail_mem=102.66 GB):  83%|████████▎ | 48/58 [00:07<00:00, 29.18it/s]Capturing num tokens (num_tokens=64 avail_mem=102.65 GB):  83%|████████▎ | 48/58 [00:07<00:00, 29.18it/s]Capturing num tokens (num_tokens=48 avail_mem=102.65 GB):  83%|████████▎ | 48/58 [00:07<00:00, 29.18it/s]Capturing num tokens (num_tokens=32 avail_mem=102.65 GB):  83%|████████▎ | 48/58 [00:07<00:00, 29.18it/s]Capturing num tokens (num_tokens=28 avail_mem=102.65 GB):  83%|████████▎ | 48/58 [00:07<00:00, 29.18it/s]Capturing num tokens (num_tokens=28 avail_mem=102.65 GB):  90%|████████▉ | 52/58 [00:07<00:00, 31.08it/s]Capturing num tokens (num_tokens=24 avail_mem=102.64 GB):  90%|████████▉ | 52/58 [00:07<00:00, 31.08it/s]

    Capturing num tokens (num_tokens=20 avail_mem=101.60 GB):  90%|████████▉ | 52/58 [00:07<00:00, 31.08it/s]Capturing num tokens (num_tokens=16 avail_mem=101.60 GB):  90%|████████▉ | 52/58 [00:07<00:00, 31.08it/s]

    Capturing num tokens (num_tokens=12 avail_mem=101.60 GB):  90%|████████▉ | 52/58 [00:07<00:00, 31.08it/s]Capturing num tokens (num_tokens=12 avail_mem=101.60 GB):  97%|█████████▋| 56/58 [00:07<00:00, 19.77it/s]Capturing num tokens (num_tokens=8 avail_mem=102.60 GB):  97%|█████████▋| 56/58 [00:07<00:00, 19.77it/s] Capturing num tokens (num_tokens=4 avail_mem=102.59 GB):  97%|█████████▋| 56/58 [00:07<00:00, 19.77it/s]

    Capturing num tokens (num_tokens=4 avail_mem=102.59 GB): 100%|██████████| 58/58 [00:08<00:00,  7.24it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I need to identify the two numbers involved in the addition, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>After performing the addition, I'll arrive at the final answer.<br></think><br><br>Sure! Let's solve the addition step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I need to identify the two numbers involved in the addition, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>After performing the addition, I'll arrive at the final answer.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Sure! Let's solve the addition step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
