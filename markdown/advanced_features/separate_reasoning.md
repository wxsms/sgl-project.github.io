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
    [2026-04-24 09:46:38] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 09:46:39] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-24 09:46:40] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-24 09:46:42] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 09:46:48] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-24 09:46:49] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False
    [2026-04-24 09:46:49] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.31s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.48s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.45s/it]


    2026-04-24 09:46:57,433 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 09:46:57] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:56,  3.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:56,  3.10s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.08it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.08it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:38,  1.40it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:38,  1.40it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:29,  1.79it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:29,  1.79it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:25,  2.07it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:25,  2.07it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:20,  2.43it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:20,  2.43it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:18,  2.73it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:18,  2.73it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:16,  2.89it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:16,  2.89it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:16,  2.97it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:16,  2.97it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:15,  3.12it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:15,  3.12it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:14,  3.25it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:14,  3.25it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:13,  3.45it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:13,  3.45it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:11,  3.79it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:11,  3.79it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:10,  4.03it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:10,  4.03it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:09,  4.36it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:09,  4.36it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:08,  4.96it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:08,  4.96it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:07<00:07,  5.32it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:07<00:07,  5.32it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:07<00:06,  5.71it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:07<00:06,  5.71it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:05,  6.38it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:05,  6.38it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:05,  6.38it/s] 

    Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:08<00:04,  8.10it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:08<00:04,  8.10it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:08<00:04,  8.31it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:08<00:04,  8.31it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:08<00:04,  8.31it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:08<00:03,  9.68it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:08<00:03,  9.68it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:08<00:03,  9.68it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:08<00:02, 11.10it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:08<00:02, 11.10it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:08<00:02, 11.10it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:08<00:02, 11.94it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:08<00:02, 11.94it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:08<00:02, 11.94it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:08<00:02, 13.43it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:08<00:02, 13.43it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:08<00:02, 13.43it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:08<00:01, 14.95it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:08<00:01, 14.95it/s]

    Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:08<00:01, 14.95it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:08<00:01, 15.56it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:08<00:01, 15.56it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:08<00:01, 15.56it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:09<00:01, 16.10it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:09<00:01, 16.10it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:09<00:01, 16.10it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:09<00:01, 16.10it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:09<00:00, 18.80it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:09<00:00, 18.80it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:09<00:00, 18.80it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:09<00:00, 18.55it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:09<00:00, 18.55it/s]

    Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:09<00:00, 18.55it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:09<00:00, 18.93it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:09<00:00, 18.93it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:09<00:00, 18.93it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:09<00:00, 18.93it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:09<00:00, 21.08it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:09<00:00, 21.08it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:09<00:00, 21.08it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:09<00:00, 21.08it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:09<00:00, 20.51it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:09<00:00, 20.51it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:09<00:00, 20.51it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:09<00:00, 20.51it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:09<00:00, 22.64it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:09<00:00, 22.64it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:09<00:00, 22.64it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:09<00:00, 22.64it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:09<00:00, 23.50it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:09<00:00, 23.50it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:09<00:00, 23.50it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  5.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=84.97 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=84.97 GB):   2%|▏         | 1/58 [00:00<00:44,  1.29it/s]Capturing num tokens (num_tokens=7680 avail_mem=85.04 GB):   2%|▏         | 1/58 [00:00<00:44,  1.29it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=85.04 GB):   3%|▎         | 2/58 [00:01<00:38,  1.46it/s]Capturing num tokens (num_tokens=7168 avail_mem=85.27 GB):   3%|▎         | 2/58 [00:01<00:38,  1.46it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=85.27 GB):   5%|▌         | 3/58 [00:01<00:33,  1.66it/s]Capturing num tokens (num_tokens=6656 avail_mem=85.26 GB):   5%|▌         | 3/58 [00:01<00:33,  1.66it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=85.26 GB):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=85.26 GB):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=85.26 GB):   9%|▊         | 5/58 [00:02<00:24,  2.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=85.25 GB):   9%|▊         | 5/58 [00:02<00:24,  2.17it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=85.25 GB):  10%|█         | 6/58 [00:02<00:20,  2.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=85.24 GB):  10%|█         | 6/58 [00:02<00:20,  2.50it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=85.24 GB):  12%|█▏        | 7/58 [00:03<00:17,  2.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=85.23 GB):  12%|█▏        | 7/58 [00:03<00:17,  2.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=85.23 GB):  14%|█▍        | 8/58 [00:03<00:14,  3.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=85.23 GB):  14%|█▍        | 8/58 [00:03<00:14,  3.42it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=85.23 GB):  16%|█▌        | 9/58 [00:03<00:14,  3.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=84.19 GB):  16%|█▌        | 9/58 [00:03<00:14,  3.42it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=84.19 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=85.19 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.38it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=85.19 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=84.36 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.52it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=84.36 GB):  21%|██        | 12/58 [00:04<00:13,  3.53it/s]Capturing num tokens (num_tokens=3072 avail_mem=84.35 GB):  21%|██        | 12/58 [00:04<00:13,  3.53it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=84.35 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=85.18 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.75it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=85.18 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.98it/s]Capturing num tokens (num_tokens=2560 avail_mem=84.41 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.98it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=84.41 GB):  26%|██▌       | 15/58 [00:05<00:10,  4.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=85.18 GB):  26%|██▌       | 15/58 [00:05<00:10,  4.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=85.18 GB):  28%|██▊       | 16/58 [00:05<00:09,  4.42it/s]Capturing num tokens (num_tokens=2048 avail_mem=84.45 GB):  28%|██▊       | 16/58 [00:05<00:09,  4.42it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=84.45 GB):  29%|██▉       | 17/58 [00:05<00:08,  4.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=84.45 GB):  29%|██▉       | 17/58 [00:05<00:08,  4.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=84.45 GB):  31%|███       | 18/58 [00:05<00:08,  4.87it/s]Capturing num tokens (num_tokens=1536 avail_mem=85.15 GB):  31%|███       | 18/58 [00:05<00:08,  4.87it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=85.15 GB):  33%|███▎      | 19/58 [00:05<00:07,  5.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=84.49 GB):  33%|███▎      | 19/58 [00:05<00:07,  5.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=84.49 GB):  34%|███▍      | 20/58 [00:06<00:06,  5.63it/s]Capturing num tokens (num_tokens=1024 avail_mem=85.15 GB):  34%|███▍      | 20/58 [00:06<00:06,  5.63it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=85.15 GB):  36%|███▌      | 21/58 [00:06<00:05,  6.37it/s]Capturing num tokens (num_tokens=960 avail_mem=84.54 GB):  36%|███▌      | 21/58 [00:06<00:05,  6.37it/s] Capturing num tokens (num_tokens=960 avail_mem=84.54 GB):  38%|███▊      | 22/58 [00:06<00:05,  6.38it/s]Capturing num tokens (num_tokens=896 avail_mem=84.53 GB):  38%|███▊      | 22/58 [00:06<00:05,  6.38it/s]

    Capturing num tokens (num_tokens=896 avail_mem=84.53 GB):  40%|███▉      | 23/58 [00:06<00:05,  6.99it/s]Capturing num tokens (num_tokens=832 avail_mem=85.13 GB):  40%|███▉      | 23/58 [00:06<00:05,  6.99it/s]Capturing num tokens (num_tokens=832 avail_mem=85.13 GB):  41%|████▏     | 24/58 [00:06<00:04,  7.64it/s]Capturing num tokens (num_tokens=768 avail_mem=84.58 GB):  41%|████▏     | 24/58 [00:06<00:04,  7.64it/s]

    Capturing num tokens (num_tokens=768 avail_mem=84.58 GB):  43%|████▎     | 25/58 [00:06<00:04,  7.73it/s]Capturing num tokens (num_tokens=704 avail_mem=84.57 GB):  43%|████▎     | 25/58 [00:06<00:04,  7.73it/s]Capturing num tokens (num_tokens=640 avail_mem=85.12 GB):  43%|████▎     | 25/58 [00:06<00:04,  7.73it/s]Capturing num tokens (num_tokens=640 avail_mem=85.12 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.79it/s]Capturing num tokens (num_tokens=576 avail_mem=84.62 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.79it/s]

    Capturing num tokens (num_tokens=512 avail_mem=85.11 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.79it/s]Capturing num tokens (num_tokens=512 avail_mem=85.11 GB):  50%|█████     | 29/58 [00:06<00:02,  9.82it/s]Capturing num tokens (num_tokens=480 avail_mem=84.68 GB):  50%|█████     | 29/58 [00:06<00:02,  9.82it/s]Capturing num tokens (num_tokens=448 avail_mem=85.12 GB):  50%|█████     | 29/58 [00:07<00:02,  9.82it/s]

    Capturing num tokens (num_tokens=448 avail_mem=85.12 GB):  53%|█████▎    | 31/58 [00:07<00:02, 10.66it/s]Capturing num tokens (num_tokens=416 avail_mem=84.80 GB):  53%|█████▎    | 31/58 [00:07<00:02, 10.66it/s]Capturing num tokens (num_tokens=384 avail_mem=84.69 GB):  53%|█████▎    | 31/58 [00:07<00:02, 10.66it/s]Capturing num tokens (num_tokens=384 avail_mem=84.69 GB):  57%|█████▋    | 33/58 [00:07<00:02, 10.75it/s]Capturing num tokens (num_tokens=352 avail_mem=85.10 GB):  57%|█████▋    | 33/58 [00:07<00:02, 10.75it/s]

    Capturing num tokens (num_tokens=320 avail_mem=84.71 GB):  57%|█████▋    | 33/58 [00:07<00:02, 10.75it/s]Capturing num tokens (num_tokens=320 avail_mem=84.71 GB):  60%|██████    | 35/58 [00:07<00:02, 11.11it/s]Capturing num tokens (num_tokens=288 avail_mem=85.10 GB):  60%|██████    | 35/58 [00:07<00:02, 11.11it/s]Capturing num tokens (num_tokens=256 avail_mem=84.73 GB):  60%|██████    | 35/58 [00:07<00:02, 11.11it/s]

    Capturing num tokens (num_tokens=256 avail_mem=84.73 GB):  64%|██████▍   | 37/58 [00:07<00:01, 11.63it/s]Capturing num tokens (num_tokens=240 avail_mem=85.09 GB):  64%|██████▍   | 37/58 [00:07<00:01, 11.63it/s]Capturing num tokens (num_tokens=224 avail_mem=84.75 GB):  64%|██████▍   | 37/58 [00:07<00:01, 11.63it/s]Capturing num tokens (num_tokens=224 avail_mem=84.75 GB):  67%|██████▋   | 39/58 [00:07<00:01, 12.19it/s]Capturing num tokens (num_tokens=208 avail_mem=85.08 GB):  67%|██████▋   | 39/58 [00:07<00:01, 12.19it/s]

    Capturing num tokens (num_tokens=192 avail_mem=84.77 GB):  67%|██████▋   | 39/58 [00:07<00:01, 12.19it/s]Capturing num tokens (num_tokens=192 avail_mem=84.77 GB):  71%|███████   | 41/58 [00:07<00:01, 12.91it/s]Capturing num tokens (num_tokens=176 avail_mem=85.07 GB):  71%|███████   | 41/58 [00:07<00:01, 12.91it/s]Capturing num tokens (num_tokens=160 avail_mem=84.97 GB):  71%|███████   | 41/58 [00:07<00:01, 12.91it/s]Capturing num tokens (num_tokens=160 avail_mem=84.97 GB):  74%|███████▍  | 43/58 [00:08<00:01, 13.82it/s]Capturing num tokens (num_tokens=144 avail_mem=85.06 GB):  74%|███████▍  | 43/58 [00:08<00:01, 13.82it/s]

    Capturing num tokens (num_tokens=128 avail_mem=85.11 GB):  74%|███████▍  | 43/58 [00:08<00:01, 13.82it/s]Capturing num tokens (num_tokens=128 avail_mem=85.11 GB):  78%|███████▊  | 45/58 [00:08<00:00, 14.70it/s]Capturing num tokens (num_tokens=112 avail_mem=85.07 GB):  78%|███████▊  | 45/58 [00:08<00:00, 14.70it/s]Capturing num tokens (num_tokens=96 avail_mem=85.07 GB):  78%|███████▊  | 45/58 [00:08<00:00, 14.70it/s] Capturing num tokens (num_tokens=96 avail_mem=85.07 GB):  81%|████████  | 47/58 [00:08<00:00, 15.15it/s]Capturing num tokens (num_tokens=80 avail_mem=84.86 GB):  81%|████████  | 47/58 [00:08<00:00, 15.15it/s]

    Capturing num tokens (num_tokens=64 avail_mem=85.05 GB):  81%|████████  | 47/58 [00:08<00:00, 15.15it/s]Capturing num tokens (num_tokens=64 avail_mem=85.05 GB):  84%|████████▍ | 49/58 [00:08<00:00, 15.92it/s]Capturing num tokens (num_tokens=48 avail_mem=85.05 GB):  84%|████████▍ | 49/58 [00:08<00:00, 15.92it/s]Capturing num tokens (num_tokens=32 avail_mem=85.07 GB):  84%|████████▍ | 49/58 [00:08<00:00, 15.92it/s]Capturing num tokens (num_tokens=28 avail_mem=84.93 GB):  84%|████████▍ | 49/58 [00:08<00:00, 15.92it/s]Capturing num tokens (num_tokens=28 avail_mem=84.93 GB):  90%|████████▉ | 52/58 [00:08<00:00, 17.77it/s]Capturing num tokens (num_tokens=24 avail_mem=85.04 GB):  90%|████████▉ | 52/58 [00:08<00:00, 17.77it/s]

    Capturing num tokens (num_tokens=20 avail_mem=85.03 GB):  90%|████████▉ | 52/58 [00:08<00:00, 17.77it/s]Capturing num tokens (num_tokens=16 avail_mem=85.02 GB):  90%|████████▉ | 52/58 [00:08<00:00, 17.77it/s]Capturing num tokens (num_tokens=16 avail_mem=85.02 GB):  95%|█████████▍| 55/58 [00:08<00:00, 18.66it/s]Capturing num tokens (num_tokens=12 avail_mem=85.02 GB):  95%|█████████▍| 55/58 [00:08<00:00, 18.66it/s]Capturing num tokens (num_tokens=8 avail_mem=85.01 GB):  95%|█████████▍| 55/58 [00:08<00:00, 18.66it/s] Capturing num tokens (num_tokens=4 avail_mem=84.94 GB):  95%|█████████▍| 55/58 [00:08<00:00, 18.66it/s]

    Capturing num tokens (num_tokens=4 avail_mem=84.94 GB): 100%|██████████| 58/58 [00:08<00:00, 20.07it/s]Capturing num tokens (num_tokens=4 avail_mem=84.94 GB): 100%|██████████| 58/58 [00:08<00:00,  6.59it/s]


    [2026-04-24 09:47:18] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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



<strong style='color: #00008B;'>I start by identifying the two numbers in the problem, which are 1 and 3.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum to find the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3<br>   \]<br><br>3. **Calculate the result:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>To solve this, I add the two numbers together.<br><br>1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I need to identify the two numbers in the addition problem: 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Finally, I'll calculate the sum to find the result.</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I'll add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of \(1\) and \(3\).<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>I need to calculate the sum of 1 and 3.<br><br>First, I'll identify the two numbers involved in the addition.<br><br>Then, I'll add them together to find the total.<br><br>Finally, I'll present the result as the answer.<br></think><br><br>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the answer is \(\boxed{4}\).</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>I need to calculate the sum of 1 and 3.<br><br>First, I'll identify the two numbers involved in the addition.<br><br>Then, I'll add them together to find the total.<br><br>Finally, I'll present the result as the answer.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the answer is \(\boxed{4}\).</strong>



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
    [2026-04-24 09:47:42] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.33s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.52s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.49s/it]


    2026-04-24 09:47:51,817 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 09:47:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:55,  3.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:55,  3.08s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:32,  1.66s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:32,  1.66s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:54,  1.01it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:54,  1.01it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:36,  1.50it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:36,  1.50it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:25,  2.07it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:25,  2.07it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:19,  2.70it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:19,  2.70it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.43it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.43it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.20it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.20it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.08it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.08it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.08it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:06,  6.78it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:05,  8.34it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:05,  8.34it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.34it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.87it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.87it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.87it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.82it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.82it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.82it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.82it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.26it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.26it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.26it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.26it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.26it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.66it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:06<00:00, 31.17it/s]

    Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:06<00:00, 31.17it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:06<00:00, 31.17it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 30.29it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 30.29it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 30.29it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 30.29it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 30.29it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 30.62it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 30.62it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 30.62it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 30.62it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:06<00:00, 30.62it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 30.60it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 30.60it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 30.60it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 30.60it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 30.60it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 31.23it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 31.23it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 31.23it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 31.23it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=102.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=102.76 GB):   2%|▏         | 1/58 [00:00<00:36,  1.55it/s]Capturing num tokens (num_tokens=7680 avail_mem=102.73 GB):   2%|▏         | 1/58 [00:00<00:36,  1.55it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=102.73 GB):   3%|▎         | 2/58 [00:01<00:33,  1.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=102.73 GB):   3%|▎         | 2/58 [00:01<00:33,  1.69it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=102.73 GB):   5%|▌         | 3/58 [00:01<00:30,  1.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=102.74 GB):   5%|▌         | 3/58 [00:01<00:30,  1.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=102.74 GB):   7%|▋         | 4/58 [00:02<00:27,  1.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=102.74 GB):   7%|▋         | 4/58 [00:02<00:27,  1.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=102.74 GB):   9%|▊         | 5/58 [00:02<00:25,  2.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=102.74 GB):   9%|▊         | 5/58 [00:02<00:25,  2.07it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=102.74 GB):  10%|█         | 6/58 [00:02<00:22,  2.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=102.74 GB):  10%|█         | 6/58 [00:02<00:22,  2.27it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=102.74 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=102.75 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.49it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=102.75 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=102.75 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.80it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=102.75 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=102.76 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=102.76 GB):  17%|█▋        | 10/58 [00:03<00:13,  3.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=102.76 GB):  17%|█▋        | 10/58 [00:03<00:13,  3.62it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=102.76 GB):  19%|█▉        | 11/58 [00:04<00:11,  4.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=102.75 GB):  19%|█▉        | 11/58 [00:04<00:11,  4.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=102.75 GB):  21%|██        | 12/58 [00:04<00:09,  4.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=102.75 GB):  21%|██        | 12/58 [00:04<00:09,  4.64it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=102.75 GB):  22%|██▏       | 13/58 [00:04<00:08,  5.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=102.75 GB):  22%|██▏       | 13/58 [00:04<00:08,  5.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=102.75 GB):  22%|██▏       | 13/58 [00:04<00:08,  5.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=102.75 GB):  26%|██▌       | 15/58 [00:04<00:05,  7.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=102.75 GB):  26%|██▌       | 15/58 [00:04<00:05,  7.31it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=102.75 GB):  26%|██▌       | 15/58 [00:04<00:05,  7.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=102.75 GB):  29%|██▉       | 17/58 [00:04<00:04,  9.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=102.75 GB):  29%|██▉       | 17/58 [00:04<00:04,  9.12it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=101.71 GB):  29%|██▉       | 17/58 [00:04<00:04,  9.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=101.71 GB):  33%|███▎      | 19/58 [00:04<00:04,  7.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=101.71 GB):  33%|███▎      | 19/58 [00:04<00:04,  7.88it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=101.71 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.10it/s]Capturing num tokens (num_tokens=1024 avail_mem=102.72 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.10it/s]Capturing num tokens (num_tokens=1024 avail_mem=102.72 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s]Capturing num tokens (num_tokens=960 avail_mem=102.71 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=102.71 GB):  38%|███▊      | 22/58 [00:05<00:04,  8.48it/s]Capturing num tokens (num_tokens=896 avail_mem=101.89 GB):  38%|███▊      | 22/58 [00:05<00:04,  8.48it/s]Capturing num tokens (num_tokens=896 avail_mem=101.89 GB):  40%|███▉      | 23/58 [00:05<00:04,  8.24it/s]Capturing num tokens (num_tokens=832 avail_mem=101.88 GB):  40%|███▉      | 23/58 [00:05<00:04,  8.24it/s]

    Capturing num tokens (num_tokens=832 avail_mem=101.88 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.49it/s]Capturing num tokens (num_tokens=768 avail_mem=102.70 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.49it/s]Capturing num tokens (num_tokens=704 avail_mem=102.70 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.49it/s]Capturing num tokens (num_tokens=704 avail_mem=102.70 GB):  45%|████▍     | 26/58 [00:05<00:03,  9.58it/s]Capturing num tokens (num_tokens=640 avail_mem=101.93 GB):  45%|████▍     | 26/58 [00:05<00:03,  9.58it/s]

    Capturing num tokens (num_tokens=640 avail_mem=101.93 GB):  47%|████▋     | 27/58 [00:05<00:03,  9.31it/s]Capturing num tokens (num_tokens=576 avail_mem=101.93 GB):  47%|████▋     | 27/58 [00:05<00:03,  9.31it/s]Capturing num tokens (num_tokens=576 avail_mem=101.93 GB):  48%|████▊     | 28/58 [00:05<00:03,  9.24it/s]Capturing num tokens (num_tokens=512 avail_mem=102.75 GB):  48%|████▊     | 28/58 [00:05<00:03,  9.24it/s]

    Capturing num tokens (num_tokens=512 avail_mem=102.75 GB):  50%|█████     | 29/58 [00:06<00:03,  9.43it/s]Capturing num tokens (num_tokens=480 avail_mem=102.69 GB):  50%|█████     | 29/58 [00:06<00:03,  9.43it/s]Capturing num tokens (num_tokens=480 avail_mem=102.69 GB):  52%|█████▏    | 30/58 [00:06<00:02,  9.55it/s]Capturing num tokens (num_tokens=448 avail_mem=101.98 GB):  52%|█████▏    | 30/58 [00:06<00:02,  9.55it/s]

    Capturing num tokens (num_tokens=448 avail_mem=101.98 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.31it/s]Capturing num tokens (num_tokens=416 avail_mem=101.97 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.31it/s]Capturing num tokens (num_tokens=384 avail_mem=102.68 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.31it/s]Capturing num tokens (num_tokens=384 avail_mem=102.68 GB):  57%|█████▋    | 33/58 [00:06<00:02, 10.17it/s]Capturing num tokens (num_tokens=352 avail_mem=102.02 GB):  57%|█████▋    | 33/58 [00:06<00:02, 10.17it/s]

    Capturing num tokens (num_tokens=352 avail_mem=102.02 GB):  59%|█████▊    | 34/58 [00:06<00:02,  9.95it/s]Capturing num tokens (num_tokens=320 avail_mem=102.02 GB):  59%|█████▊    | 34/58 [00:06<00:02,  9.95it/s]Capturing num tokens (num_tokens=288 avail_mem=102.68 GB):  59%|█████▊    | 34/58 [00:06<00:02,  9.95it/s]Capturing num tokens (num_tokens=288 avail_mem=102.68 GB):  62%|██████▏   | 36/58 [00:06<00:02, 10.82it/s]Capturing num tokens (num_tokens=256 avail_mem=102.67 GB):  62%|██████▏   | 36/58 [00:06<00:02, 10.82it/s]

    Capturing num tokens (num_tokens=240 avail_mem=102.06 GB):  62%|██████▏   | 36/58 [00:06<00:02, 10.82it/s]Capturing num tokens (num_tokens=240 avail_mem=102.06 GB):  66%|██████▌   | 38/58 [00:06<00:01, 10.73it/s]Capturing num tokens (num_tokens=224 avail_mem=102.67 GB):  66%|██████▌   | 38/58 [00:06<00:01, 10.73it/s]Capturing num tokens (num_tokens=208 avail_mem=102.66 GB):  66%|██████▌   | 38/58 [00:06<00:01, 10.73it/s]

    Capturing num tokens (num_tokens=208 avail_mem=102.66 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.05it/s]Capturing num tokens (num_tokens=192 avail_mem=102.11 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.05it/s]Capturing num tokens (num_tokens=176 avail_mem=102.66 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.05it/s]Capturing num tokens (num_tokens=176 avail_mem=102.66 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.88it/s]Capturing num tokens (num_tokens=160 avail_mem=102.16 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.88it/s]

    Capturing num tokens (num_tokens=144 avail_mem=102.16 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.88it/s]Capturing num tokens (num_tokens=144 avail_mem=102.16 GB):  76%|███████▌  | 44/58 [00:07<00:01, 11.79it/s]Capturing num tokens (num_tokens=128 avail_mem=102.67 GB):  76%|███████▌  | 44/58 [00:07<00:01, 11.79it/s]Capturing num tokens (num_tokens=112 avail_mem=102.22 GB):  76%|███████▌  | 44/58 [00:07<00:01, 11.79it/s]

    Capturing num tokens (num_tokens=112 avail_mem=102.22 GB):  79%|███████▉  | 46/58 [00:07<00:01, 11.92it/s]Capturing num tokens (num_tokens=96 avail_mem=102.73 GB):  79%|███████▉  | 46/58 [00:07<00:01, 11.92it/s] Capturing num tokens (num_tokens=80 avail_mem=102.65 GB):  79%|███████▉  | 46/58 [00:07<00:01, 11.92it/s]Capturing num tokens (num_tokens=80 avail_mem=102.65 GB):  83%|████████▎ | 48/58 [00:07<00:00, 12.52it/s]Capturing num tokens (num_tokens=64 avail_mem=102.24 GB):  83%|████████▎ | 48/58 [00:07<00:00, 12.52it/s]

    Capturing num tokens (num_tokens=48 avail_mem=102.65 GB):  83%|████████▎ | 48/58 [00:07<00:00, 12.52it/s]Capturing num tokens (num_tokens=48 avail_mem=102.65 GB):  86%|████████▌ | 50/58 [00:07<00:00, 12.88it/s]Capturing num tokens (num_tokens=32 avail_mem=102.26 GB):  86%|████████▌ | 50/58 [00:07<00:00, 12.88it/s]Capturing num tokens (num_tokens=28 avail_mem=102.64 GB):  86%|████████▌ | 50/58 [00:07<00:00, 12.88it/s]

    Capturing num tokens (num_tokens=28 avail_mem=102.64 GB):  90%|████████▉ | 52/58 [00:07<00:00, 13.22it/s]Capturing num tokens (num_tokens=24 avail_mem=102.28 GB):  90%|████████▉ | 52/58 [00:07<00:00, 13.22it/s]Capturing num tokens (num_tokens=20 avail_mem=102.64 GB):  90%|████████▉ | 52/58 [00:08<00:00, 13.22it/s]Capturing num tokens (num_tokens=20 avail_mem=102.64 GB):  93%|█████████▎| 54/58 [00:08<00:00, 13.78it/s]Capturing num tokens (num_tokens=16 avail_mem=102.38 GB):  93%|█████████▎| 54/58 [00:08<00:00, 13.78it/s]

    Capturing num tokens (num_tokens=12 avail_mem=102.32 GB):  93%|█████████▎| 54/58 [00:08<00:00, 13.78it/s]Capturing num tokens (num_tokens=12 avail_mem=102.32 GB):  97%|█████████▋| 56/58 [00:08<00:00, 13.96it/s]Capturing num tokens (num_tokens=8 avail_mem=102.62 GB):  97%|█████████▋| 56/58 [00:08<00:00, 13.96it/s] Capturing num tokens (num_tokens=4 avail_mem=102.32 GB):  97%|█████████▋| 56/58 [00:08<00:00, 13.96it/s]Capturing num tokens (num_tokens=4 avail_mem=102.32 GB): 100%|██████████| 58/58 [00:08<00:00, 14.40it/s]Capturing num tokens (num_tokens=4 avail_mem=102.32 GB): 100%|██████████| 58/58 [00:08<00:00,  6.92it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I need to add the numbers 1 and 3.<br><br>Adding 1 and 3 gives me 4.<br><br>Therefore, the sum of 1 and 3 is 4.<br></think><br><br>Sure, let's solve the addition step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Answer:**  <br>\(\boxed{4}\)</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I need to add the numbers 1 and 3.<br><br>Adding 1 and 3 gives me 4.<br><br>Therefore, the sum of 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Sure, let's solve the addition step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Start with the first number:**  <br>   \(1\)<br><br>2. **Add the second number:**  <br>   \(1 + 3\)<br><br>3. **Calculate the sum:**  <br>   \(1 + 3 = 4\)<br><br>**Answer:**  <br>\(\boxed{4}\)</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
