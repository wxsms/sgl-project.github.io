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
    [2026-04-24 16:50:04] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 16:50:05] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-24 16:50:06] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-24 16:50:08] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 16:50:15] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-24 16:50:16] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False
    [2026-04-24 16:50:16] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.19s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.22s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.21s/it]


    2026-04-24 16:50:24,528 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 16:50:24] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:57,  3.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:57,  3.12s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:38,  1.76s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:38,  1.76s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:32,  1.62it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:26,  1.91it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:26,  1.91it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:22,  2.22it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:22,  2.22it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:19,  2.57it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:19,  2.57it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:16,  2.99it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:16,  2.99it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:14,  3.26it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:14,  3.26it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:13,  3.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:13,  3.37it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:12,  3.54it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:12,  3.54it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:11,  3.84it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:11,  3.84it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:10,  4.23it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:10,  4.23it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:09,  4.46it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:09,  4.46it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:08<00:08,  4.73it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:08<00:08,  4.73it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:08<00:07,  5.13it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:08<00:07,  5.13it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:08<00:06,  5.63it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:08<00:06,  5.63it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:08<00:06,  6.32it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:08<00:06,  6.32it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:08<00:05,  6.94it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:08<00:05,  6.94it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:08<00:05,  6.94it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:09<00:04,  8.12it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:09<00:04,  8.12it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:09<00:04,  8.12it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:09<00:03,  9.35it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:09<00:03,  9.35it/s]

    Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:09<00:03,  9.35it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:09<00:02, 10.81it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:09<00:02, 10.81it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:09<00:02, 10.81it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:09<00:02, 11.94it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:09<00:02, 11.94it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:09<00:02, 11.94it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:09<00:02, 13.33it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:09<00:02, 13.33it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:09<00:02, 13.33it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:09<00:01, 13.80it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:09<00:01, 13.80it/s]

    Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:09<00:01, 13.80it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:09<00:01, 15.09it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:09<00:01, 15.09it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:09<00:01, 15.09it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:09<00:01, 15.95it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:09<00:01, 15.95it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:09<00:01, 15.95it/s]

    Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:10<00:01, 15.95it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:10<00:00, 18.18it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:10<00:00, 18.18it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:10<00:00, 18.18it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:10<00:00, 18.27it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:10<00:00, 18.27it/s]

    Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:10<00:00, 18.27it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:10<00:00, 18.27it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:10<00:00, 18.27it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:10<00:00, 18.27it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:10<00:00, 18.27it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:10<00:00, 20.46it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:10<00:00, 20.46it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:10<00:00, 20.46it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:10<00:00, 20.46it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:10<00:00, 21.28it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:10<00:00, 21.28it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:10<00:00, 21.28it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:10<00:00, 21.28it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:10<00:00, 21.29it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:10<00:00, 21.29it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:10<00:00, 21.29it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:10<00:00, 21.29it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:10<00:00, 22.68it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:10<00:00, 22.68it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:10<00:00, 22.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=84.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=84.70 GB):   2%|▏         | 1/58 [00:00<00:38,  1.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=84.77 GB):   2%|▏         | 1/58 [00:00<00:38,  1.47it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=84.77 GB):   3%|▎         | 2/58 [00:01<00:32,  1.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=84.80 GB):   3%|▎         | 2/58 [00:01<00:32,  1.74it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=84.80 GB):   5%|▌         | 3/58 [00:01<00:27,  1.97it/s]Capturing num tokens (num_tokens=6656 avail_mem=83.84 GB):   5%|▌         | 3/58 [00:01<00:27,  1.97it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=83.84 GB):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=84.24 GB):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=84.24 GB):   9%|▊         | 5/58 [00:02<00:28,  1.87it/s]Capturing num tokens (num_tokens=5632 avail_mem=84.43 GB):   9%|▊         | 5/58 [00:02<00:28,  1.87it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=84.43 GB):  10%|█         | 6/58 [00:03<00:26,  1.94it/s]Capturing num tokens (num_tokens=5120 avail_mem=85.25 GB):  10%|█         | 6/58 [00:03<00:26,  1.94it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=85.25 GB):  12%|█▏        | 7/58 [00:03<00:24,  2.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=84.32 GB):  12%|█▏        | 7/58 [00:03<00:24,  2.10it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=84.32 GB):  14%|█▍        | 8/58 [00:03<00:22,  2.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=85.13 GB):  14%|█▍        | 8/58 [00:03<00:22,  2.24it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=85.13 GB):  16%|█▌        | 9/58 [00:04<00:20,  2.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=84.54 GB):  16%|█▌        | 9/58 [00:04<00:20,  2.41it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=84.54 GB):  17%|█▋        | 10/58 [00:04<00:18,  2.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=84.59 GB):  17%|█▋        | 10/58 [00:04<00:18,  2.60it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=84.59 GB):  19%|█▉        | 11/58 [00:04<00:17,  2.75it/s]Capturing num tokens (num_tokens=3328 avail_mem=85.25 GB):  19%|█▉        | 11/58 [00:04<00:17,  2.75it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=85.25 GB):  21%|██        | 12/58 [00:05<00:15,  3.00it/s]Capturing num tokens (num_tokens=3072 avail_mem=84.63 GB):  21%|██        | 12/58 [00:05<00:15,  3.00it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=84.63 GB):  22%|██▏       | 13/58 [00:05<00:13,  3.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=85.24 GB):  22%|██▏       | 13/58 [00:05<00:13,  3.25it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=85.24 GB):  24%|██▍       | 14/58 [00:05<00:12,  3.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=84.68 GB):  24%|██▍       | 14/58 [00:05<00:12,  3.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=84.68 GB):  26%|██▌       | 15/58 [00:05<00:11,  3.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=84.69 GB):  26%|██▌       | 15/58 [00:05<00:11,  3.90it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=84.69 GB):  28%|██▊       | 16/58 [00:06<00:10,  4.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=85.22 GB):  28%|██▊       | 16/58 [00:06<00:10,  4.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=85.22 GB):  29%|██▉       | 17/58 [00:06<00:09,  4.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=84.80 GB):  29%|██▉       | 17/58 [00:06<00:09,  4.54it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=84.80 GB):  31%|███       | 18/58 [00:06<00:08,  4.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=85.21 GB):  31%|███       | 18/58 [00:06<00:08,  4.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=85.21 GB):  33%|███▎      | 19/58 [00:06<00:07,  5.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=84.79 GB):  33%|███▎      | 19/58 [00:06<00:07,  5.33it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=84.79 GB):  34%|███▍      | 20/58 [00:06<00:06,  5.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=85.20 GB):  34%|███▍      | 20/58 [00:06<00:06,  5.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=85.20 GB):  36%|███▌      | 21/58 [00:06<00:05,  6.33it/s]Capturing num tokens (num_tokens=960 avail_mem=84.81 GB):  36%|███▌      | 21/58 [00:06<00:05,  6.33it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=85.12 GB):  36%|███▌      | 21/58 [00:06<00:05,  6.33it/s]Capturing num tokens (num_tokens=896 avail_mem=85.12 GB):  40%|███▉      | 23/58 [00:07<00:04,  7.38it/s]Capturing num tokens (num_tokens=832 avail_mem=85.20 GB):  40%|███▉      | 23/58 [00:07<00:04,  7.38it/s]

    Capturing num tokens (num_tokens=768 avail_mem=84.85 GB):  40%|███▉      | 23/58 [00:07<00:04,  7.38it/s]Capturing num tokens (num_tokens=768 avail_mem=84.85 GB):  43%|████▎     | 25/58 [00:07<00:03,  8.27it/s]Capturing num tokens (num_tokens=704 avail_mem=85.18 GB):  43%|████▎     | 25/58 [00:07<00:03,  8.27it/s]Capturing num tokens (num_tokens=640 avail_mem=84.86 GB):  43%|████▎     | 25/58 [00:07<00:03,  8.27it/s]

    Capturing num tokens (num_tokens=640 avail_mem=84.86 GB):  47%|████▋     | 27/58 [00:07<00:03,  9.10it/s]Capturing num tokens (num_tokens=576 avail_mem=85.14 GB):  47%|████▋     | 27/58 [00:07<00:03,  9.10it/s]Capturing num tokens (num_tokens=512 avail_mem=84.88 GB):  47%|████▋     | 27/58 [00:07<00:03,  9.10it/s]Capturing num tokens (num_tokens=512 avail_mem=84.88 GB):  50%|█████     | 29/58 [00:07<00:02,  9.90it/s]Capturing num tokens (num_tokens=480 avail_mem=85.14 GB):  50%|█████     | 29/58 [00:07<00:02,  9.90it/s]

    Capturing num tokens (num_tokens=448 avail_mem=85.14 GB):  50%|█████     | 29/58 [00:07<00:02,  9.90it/s]Capturing num tokens (num_tokens=448 avail_mem=85.14 GB):  53%|█████▎    | 31/58 [00:07<00:02, 10.55it/s]Capturing num tokens (num_tokens=416 avail_mem=84.91 GB):  53%|█████▎    | 31/58 [00:07<00:02, 10.55it/s]Capturing num tokens (num_tokens=384 avail_mem=85.10 GB):  53%|█████▎    | 31/58 [00:07<00:02, 10.55it/s]

    Capturing num tokens (num_tokens=384 avail_mem=85.10 GB):  57%|█████▋    | 33/58 [00:07<00:02, 11.33it/s]Capturing num tokens (num_tokens=352 avail_mem=85.12 GB):  57%|█████▋    | 33/58 [00:07<00:02, 11.33it/s]Capturing num tokens (num_tokens=320 avail_mem=84.94 GB):  57%|█████▋    | 33/58 [00:07<00:02, 11.33it/s]Capturing num tokens (num_tokens=320 avail_mem=84.94 GB):  60%|██████    | 35/58 [00:08<00:01, 12.53it/s]Capturing num tokens (num_tokens=288 avail_mem=85.10 GB):  60%|██████    | 35/58 [00:08<00:01, 12.53it/s]Capturing num tokens (num_tokens=256 avail_mem=85.09 GB):  60%|██████    | 35/58 [00:08<00:01, 12.53it/s]

    Capturing num tokens (num_tokens=256 avail_mem=85.09 GB):  64%|██████▍   | 37/58 [00:08<00:01, 13.64it/s]Capturing num tokens (num_tokens=240 avail_mem=85.08 GB):  64%|██████▍   | 37/58 [00:08<00:01, 13.64it/s]Capturing num tokens (num_tokens=224 avail_mem=84.96 GB):  64%|██████▍   | 37/58 [00:08<00:01, 13.64it/s]Capturing num tokens (num_tokens=208 avail_mem=85.06 GB):  64%|██████▍   | 37/58 [00:08<00:01, 13.64it/s]Capturing num tokens (num_tokens=208 avail_mem=85.06 GB):  69%|██████▉   | 40/58 [00:08<00:01, 16.25it/s]Capturing num tokens (num_tokens=192 avail_mem=85.06 GB):  69%|██████▉   | 40/58 [00:08<00:01, 16.25it/s]Capturing num tokens (num_tokens=176 avail_mem=85.05 GB):  69%|██████▉   | 40/58 [00:08<00:01, 16.25it/s]

    Capturing num tokens (num_tokens=160 avail_mem=85.05 GB):  69%|██████▉   | 40/58 [00:08<00:01, 16.25it/s]Capturing num tokens (num_tokens=160 avail_mem=85.05 GB):  74%|███████▍  | 43/58 [00:08<00:00, 17.60it/s]Capturing num tokens (num_tokens=144 avail_mem=85.04 GB):  74%|███████▍  | 43/58 [00:08<00:00, 17.60it/s]Capturing num tokens (num_tokens=128 avail_mem=85.04 GB):  74%|███████▍  | 43/58 [00:08<00:00, 17.60it/s]Capturing num tokens (num_tokens=112 avail_mem=85.04 GB):  74%|███████▍  | 43/58 [00:08<00:00, 17.60it/s]Capturing num tokens (num_tokens=112 avail_mem=85.04 GB):  79%|███████▉  | 46/58 [00:08<00:00, 19.11it/s]Capturing num tokens (num_tokens=96 avail_mem=85.03 GB):  79%|███████▉  | 46/58 [00:08<00:00, 19.11it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=85.02 GB):  79%|███████▉  | 46/58 [00:08<00:00, 19.11it/s]Capturing num tokens (num_tokens=64 avail_mem=84.97 GB):  79%|███████▉  | 46/58 [00:08<00:00, 19.11it/s]Capturing num tokens (num_tokens=64 avail_mem=84.97 GB):  84%|████████▍ | 49/58 [00:08<00:00, 20.67it/s]Capturing num tokens (num_tokens=48 avail_mem=84.97 GB):  84%|████████▍ | 49/58 [00:08<00:00, 20.67it/s]Capturing num tokens (num_tokens=32 avail_mem=84.96 GB):  84%|████████▍ | 49/58 [00:08<00:00, 20.67it/s]Capturing num tokens (num_tokens=28 avail_mem=84.95 GB):  84%|████████▍ | 49/58 [00:08<00:00, 20.67it/s]Capturing num tokens (num_tokens=28 avail_mem=84.95 GB):  90%|████████▉ | 52/58 [00:08<00:00, 21.90it/s]Capturing num tokens (num_tokens=24 avail_mem=84.97 GB):  90%|████████▉ | 52/58 [00:08<00:00, 21.90it/s]

    Capturing num tokens (num_tokens=20 avail_mem=84.97 GB):  90%|████████▉ | 52/58 [00:08<00:00, 21.90it/s]Capturing num tokens (num_tokens=16 avail_mem=84.97 GB):  90%|████████▉ | 52/58 [00:08<00:00, 21.90it/s]Capturing num tokens (num_tokens=16 avail_mem=84.97 GB):  95%|█████████▍| 55/58 [00:08<00:00, 23.25it/s]Capturing num tokens (num_tokens=12 avail_mem=84.96 GB):  95%|█████████▍| 55/58 [00:08<00:00, 23.25it/s]Capturing num tokens (num_tokens=8 avail_mem=84.97 GB):  95%|█████████▍| 55/58 [00:08<00:00, 23.25it/s] Capturing num tokens (num_tokens=4 avail_mem=84.96 GB):  95%|█████████▍| 55/58 [00:08<00:00, 23.25it/s]Capturing num tokens (num_tokens=4 avail_mem=84.96 GB): 100%|██████████| 58/58 [00:09<00:00, 24.78it/s]Capturing num tokens (num_tokens=4 avail_mem=84.96 GB): 100%|██████████| 58/58 [00:09<00:00,  6.42it/s]


    [2026-04-24 16:50:46] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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



<strong style='color: #00008B;'>First, I recognize that I need to add the numbers 1 and 3.<br><br>By performing the addition, I combine the two quantities.<br><br>The sum of 1 and 3 is 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>Sure! Let's solve the addition problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I arrive at the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I need to identify the two numbers in the problem: 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Adding 1 and 3 gives me a total of 4.<br><br>Therefore, the final answer is 4.</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I start by identifying the two numbers involved, which are 1 and 3.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the result, which is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I start by identifying the two numbers involved, which are 1 and 3.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the result, which is 4.<br></strong>



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
    [2026-04-24 16:51:11] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.25s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.32s/it]


    2026-04-24 16:51:20,855 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 16:51:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:01,  3.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:01,  3.18s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:34,  1.58it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:34,  1.58it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.17it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.17it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.54it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.54it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.32it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.32it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.21it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.21it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.21it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.89it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.89it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.89it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.43it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.43it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:05,  8.38it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:05,  8.38it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:05,  8.30it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:05,  8.30it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:04,  8.41it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:04,  8.41it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04,  8.72it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04,  8.72it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04,  8.72it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:04,  9.71it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:04,  9.71it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:03,  9.57it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:03,  9.57it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:03,  9.63it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:03,  9.63it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:03,  9.63it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:06<00:03, 10.17it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:06<00:03, 10.17it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:06<00:03, 10.17it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:06<00:03, 10.94it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:06<00:03, 10.94it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:06<00:03, 10.94it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:06<00:02, 11.64it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:06<00:02, 11.64it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:06<00:02, 11.64it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:06<00:02, 12.55it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:06<00:02, 12.55it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:06<00:02, 12.55it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:06<00:01, 13.80it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:06<00:01, 13.80it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:06<00:01, 13.80it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:06<00:01, 14.23it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:06<00:01, 14.23it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:06<00:01, 14.23it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:06<00:01, 15.18it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:06<00:01, 15.18it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:06<00:01, 15.18it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:01, 16.36it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:01, 16.36it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:01, 16.36it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:01, 16.36it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:07<00:01, 17.88it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:07<00:01, 17.88it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:07<00:01, 17.88it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:07<00:01, 17.88it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 19.51it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 19.51it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:07<00:00, 19.51it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:07<00:00, 19.51it/s]

    Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:07<00:00, 19.51it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:07<00:00, 23.20it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:07<00:00, 23.20it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:07<00:00, 23.20it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:07<00:00, 23.20it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:07<00:00, 24.76it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:07<00:00, 24.76it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 24.76it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 24.76it/s]

    Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:07<00:00, 24.76it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:07<00:00, 27.26it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:07<00:00, 27.26it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:07<00:00, 27.26it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:07<00:00, 27.26it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:07<00:00, 27.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=102.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=102.76 GB):   2%|▏         | 1/58 [00:00<00:36,  1.54it/s]Capturing num tokens (num_tokens=7680 avail_mem=102.73 GB):   2%|▏         | 1/58 [00:00<00:36,  1.54it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=102.73 GB):   3%|▎         | 2/58 [00:01<00:32,  1.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=102.73 GB):   3%|▎         | 2/58 [00:01<00:32,  1.70it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=102.73 GB):   5%|▌         | 3/58 [00:01<00:30,  1.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=102.74 GB):   5%|▌         | 3/58 [00:01<00:30,  1.82it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=102.74 GB):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=102.74 GB):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=102.74 GB):   9%|▊         | 5/58 [00:02<00:24,  2.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=102.74 GB):   9%|▊         | 5/58 [00:02<00:24,  2.16it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=102.74 GB):  10%|█         | 6/58 [00:02<00:20,  2.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=102.74 GB):  10%|█         | 6/58 [00:02<00:20,  2.48it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=102.74 GB):  12%|█▏        | 7/58 [00:03<00:17,  2.84it/s]Capturing num tokens (num_tokens=4608 avail_mem=102.75 GB):  12%|█▏        | 7/58 [00:03<00:17,  2.84it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=102.75 GB):  14%|█▍        | 8/58 [00:03<00:15,  3.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=102.75 GB):  14%|█▍        | 8/58 [00:03<00:15,  3.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=102.75 GB):  16%|█▌        | 9/58 [00:03<00:13,  3.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=102.75 GB):  16%|█▌        | 9/58 [00:03<00:13,  3.73it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=102.75 GB):  17%|█▋        | 10/58 [00:03<00:10,  4.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=102.75 GB):  17%|█▋        | 10/58 [00:03<00:10,  4.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=102.75 GB):  19%|█▉        | 11/58 [00:03<00:09,  5.22it/s]Capturing num tokens (num_tokens=3328 avail_mem=102.75 GB):  19%|█▉        | 11/58 [00:03<00:09,  5.22it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=102.75 GB):  21%|██        | 12/58 [00:03<00:07,  6.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=102.75 GB):  21%|██        | 12/58 [00:03<00:07,  6.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=102.75 GB):  22%|██▏       | 13/58 [00:03<00:06,  6.86it/s]Capturing num tokens (num_tokens=2816 avail_mem=102.75 GB):  22%|██▏       | 13/58 [00:03<00:06,  6.86it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=102.75 GB):  24%|██▍       | 14/58 [00:04<00:06,  6.56it/s]Capturing num tokens (num_tokens=2560 avail_mem=101.72 GB):  24%|██▍       | 14/58 [00:04<00:06,  6.56it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=101.72 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=101.72 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=101.72 GB):  28%|██▊       | 16/58 [00:04<00:07,  5.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=102.72 GB):  28%|██▊       | 16/58 [00:04<00:07,  5.88it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=102.72 GB):  29%|██▉       | 17/58 [00:04<00:06,  6.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=101.90 GB):  29%|██▉       | 17/58 [00:04<00:06,  6.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=101.90 GB):  31%|███       | 18/58 [00:04<00:06,  5.91it/s]Capturing num tokens (num_tokens=1536 avail_mem=101.89 GB):  31%|███       | 18/58 [00:04<00:06,  5.91it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=101.89 GB):  33%|███▎      | 19/58 [00:04<00:06,  6.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=102.72 GB):  33%|███▎      | 19/58 [00:04<00:06,  6.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=101.95 GB):  33%|███▎      | 19/58 [00:05<00:06,  6.19it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=101.95 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.08it/s]Capturing num tokens (num_tokens=960 avail_mem=101.95 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.08it/s] Capturing num tokens (num_tokens=960 avail_mem=101.95 GB):  38%|███▊      | 22/58 [00:05<00:05,  7.15it/s]Capturing num tokens (num_tokens=896 avail_mem=101.95 GB):  38%|███▊      | 22/58 [00:05<00:05,  7.15it/s]

    Capturing num tokens (num_tokens=896 avail_mem=101.95 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.55it/s]Capturing num tokens (num_tokens=832 avail_mem=102.71 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.55it/s]Capturing num tokens (num_tokens=832 avail_mem=102.71 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.72it/s]Capturing num tokens (num_tokens=768 avail_mem=102.00 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.72it/s]

    Capturing num tokens (num_tokens=768 avail_mem=102.00 GB):  43%|████▎     | 25/58 [00:05<00:04,  7.83it/s]Capturing num tokens (num_tokens=704 avail_mem=101.99 GB):  43%|████▎     | 25/58 [00:05<00:04,  7.83it/s]Capturing num tokens (num_tokens=704 avail_mem=101.99 GB):  45%|████▍     | 26/58 [00:05<00:03,  8.18it/s]Capturing num tokens (num_tokens=640 avail_mem=102.70 GB):  45%|████▍     | 26/58 [00:05<00:03,  8.18it/s]

    Capturing num tokens (num_tokens=640 avail_mem=102.70 GB):  47%|████▋     | 27/58 [00:05<00:03,  8.62it/s]Capturing num tokens (num_tokens=576 avail_mem=102.04 GB):  47%|████▋     | 27/58 [00:05<00:03,  8.62it/s]Capturing num tokens (num_tokens=576 avail_mem=102.04 GB):  48%|████▊     | 28/58 [00:05<00:03,  8.57it/s]Capturing num tokens (num_tokens=512 avail_mem=102.04 GB):  48%|████▊     | 28/58 [00:05<00:03,  8.57it/s]

    Capturing num tokens (num_tokens=480 avail_mem=102.70 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.57it/s]Capturing num tokens (num_tokens=480 avail_mem=102.70 GB):  52%|█████▏    | 30/58 [00:06<00:02,  9.49it/s]Capturing num tokens (num_tokens=448 avail_mem=102.09 GB):  52%|█████▏    | 30/58 [00:06<00:02,  9.49it/s]

    Capturing num tokens (num_tokens=448 avail_mem=102.09 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.33it/s]Capturing num tokens (num_tokens=416 avail_mem=102.09 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.33it/s]Capturing num tokens (num_tokens=384 avail_mem=102.69 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.33it/s]Capturing num tokens (num_tokens=384 avail_mem=102.69 GB):  57%|█████▋    | 33/58 [00:06<00:02, 10.08it/s]Capturing num tokens (num_tokens=352 avail_mem=102.14 GB):  57%|█████▋    | 33/58 [00:06<00:02, 10.08it/s]

    Capturing num tokens (num_tokens=320 avail_mem=102.69 GB):  57%|█████▋    | 33/58 [00:06<00:02, 10.08it/s]Capturing num tokens (num_tokens=320 avail_mem=102.69 GB):  60%|██████    | 35/58 [00:06<00:02, 10.83it/s]Capturing num tokens (num_tokens=288 avail_mem=102.19 GB):  60%|██████    | 35/58 [00:06<00:02, 10.83it/s]Capturing num tokens (num_tokens=256 avail_mem=102.18 GB):  60%|██████    | 35/58 [00:06<00:02, 10.83it/s]

    Capturing num tokens (num_tokens=256 avail_mem=102.18 GB):  64%|██████▍   | 37/58 [00:06<00:01, 11.00it/s]Capturing num tokens (num_tokens=240 avail_mem=102.68 GB):  64%|██████▍   | 37/58 [00:06<00:01, 11.00it/s]Capturing num tokens (num_tokens=224 avail_mem=102.24 GB):  64%|██████▍   | 37/58 [00:06<00:01, 11.00it/s]Capturing num tokens (num_tokens=224 avail_mem=102.24 GB):  67%|██████▋   | 39/58 [00:06<00:01, 11.40it/s]Capturing num tokens (num_tokens=208 avail_mem=102.67 GB):  67%|██████▋   | 39/58 [00:06<00:01, 11.40it/s]

    Capturing num tokens (num_tokens=192 avail_mem=102.67 GB):  67%|██████▋   | 39/58 [00:07<00:01, 11.40it/s]Capturing num tokens (num_tokens=192 avail_mem=102.67 GB):  71%|███████   | 41/58 [00:07<00:01, 11.91it/s]Capturing num tokens (num_tokens=176 avail_mem=102.25 GB):  71%|███████   | 41/58 [00:07<00:01, 11.91it/s]Capturing num tokens (num_tokens=160 avail_mem=102.67 GB):  71%|███████   | 41/58 [00:07<00:01, 11.91it/s]

    Capturing num tokens (num_tokens=160 avail_mem=102.67 GB):  74%|███████▍  | 43/58 [00:07<00:01, 11.72it/s]Capturing num tokens (num_tokens=144 avail_mem=102.28 GB):  74%|███████▍  | 43/58 [00:07<00:01, 11.72it/s]Capturing num tokens (num_tokens=128 avail_mem=102.67 GB):  74%|███████▍  | 43/58 [00:07<00:01, 11.72it/s]Capturing num tokens (num_tokens=128 avail_mem=102.67 GB):  78%|███████▊  | 45/58 [00:07<00:01, 12.26it/s]Capturing num tokens (num_tokens=112 avail_mem=102.31 GB):  78%|███████▊  | 45/58 [00:07<00:01, 12.26it/s]

    Capturing num tokens (num_tokens=96 avail_mem=102.66 GB):  78%|███████▊  | 45/58 [00:07<00:01, 12.26it/s] Capturing num tokens (num_tokens=96 avail_mem=102.66 GB):  81%|████████  | 47/58 [00:07<00:00, 12.85it/s]Capturing num tokens (num_tokens=80 avail_mem=102.32 GB):  81%|████████  | 47/58 [00:07<00:00, 12.85it/s]Capturing num tokens (num_tokens=64 avail_mem=102.65 GB):  81%|████████  | 47/58 [00:07<00:00, 12.85it/s]

    Capturing num tokens (num_tokens=64 avail_mem=102.65 GB):  84%|████████▍ | 49/58 [00:07<00:00, 13.37it/s]Capturing num tokens (num_tokens=48 avail_mem=102.35 GB):  84%|████████▍ | 49/58 [00:07<00:00, 13.37it/s]Capturing num tokens (num_tokens=32 avail_mem=102.65 GB):  84%|████████▍ | 49/58 [00:07<00:00, 13.37it/s]Capturing num tokens (num_tokens=32 avail_mem=102.65 GB):  88%|████████▊ | 51/58 [00:07<00:00, 13.91it/s]Capturing num tokens (num_tokens=28 avail_mem=102.37 GB):  88%|████████▊ | 51/58 [00:07<00:00, 13.91it/s]Capturing num tokens (num_tokens=24 avail_mem=102.64 GB):  88%|████████▊ | 51/58 [00:07<00:00, 13.91it/s]

    Capturing num tokens (num_tokens=24 avail_mem=102.64 GB):  91%|█████████▏| 53/58 [00:07<00:00, 14.29it/s]Capturing num tokens (num_tokens=20 avail_mem=102.38 GB):  91%|█████████▏| 53/58 [00:07<00:00, 14.29it/s]Capturing num tokens (num_tokens=16 avail_mem=102.63 GB):  91%|█████████▏| 53/58 [00:08<00:00, 14.29it/s]Capturing num tokens (num_tokens=16 avail_mem=102.63 GB):  95%|█████████▍| 55/58 [00:08<00:00, 15.12it/s]Capturing num tokens (num_tokens=12 avail_mem=102.63 GB):  95%|█████████▍| 55/58 [00:08<00:00, 15.12it/s]Capturing num tokens (num_tokens=8 avail_mem=102.43 GB):  95%|█████████▍| 55/58 [00:08<00:00, 15.12it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=102.43 GB):  98%|█████████▊| 57/58 [00:08<00:00, 15.94it/s]Capturing num tokens (num_tokens=4 avail_mem=102.62 GB):  98%|█████████▊| 57/58 [00:08<00:00, 15.94it/s]Capturing num tokens (num_tokens=4 avail_mem=102.62 GB): 100%|██████████| 58/58 [00:08<00:00,  7.03it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition by combining these two numbers.<br><br>Finally, I arrive at the result, which is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = \boxed{4}<br>\]</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition by combining these two numbers.<br><br>Finally, I arrive at the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = \boxed{4}<br>\]</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
