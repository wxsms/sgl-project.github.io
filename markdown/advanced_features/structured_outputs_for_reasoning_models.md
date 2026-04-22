# Structured Outputs For Reasoning Models

When working with reasoning models that use special tokens like `<think>...</think>` to denote reasoning sections, you might want to allow free-form text within these sections while still enforcing grammar constraints on the rest of the output.

SGLang provides a feature to disable grammar restrictions within reasoning sections. This is particularly useful for models that need to perform complex reasoning steps before providing a structured output.

To enable this feature, use the `--reasoning-parser` flag which decide the think_end_token, such as `</think>`, when launching the server. You can also specify the reasoning parser using the `--reasoning-parser` flag.

## Supported Models

Currently, SGLang supports the following reasoning models:
- [DeepSeek R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d): The reasoning content is wrapped with `<think>` and `</think>` tags.
- [QwQ](https://huggingface.co/Qwen/QwQ-32B): The reasoning content is wrapped with `<think>` and `</think>` tags.


## Usage

## OpenAI Compatible API

Specify the `--grammar-backend`, `--reasoning-parser` option.


```python
import openai
import os

from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

os.environ["TOKENIZERS_PARALLELISM"] = "false"


server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --host 0.0.0.0 --reasoning-parser deepseek-r1 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
```

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(
    [2026-04-22 21:12:47] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-22 21:12:54] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-22 21:12:55] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-22 21:12:56] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-22 21:13:03] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-22 21:13:04] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.43s/it]

    [2026-04-22 21:13:15] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.42s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.43s/it]


    2026-04-22 21:13:17,636 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 21:13:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:04,  3.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:04,  3.24s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:28,  1.58s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:28,  1.58s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.06it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.06it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:34,  1.56it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:34,  1.56it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.14it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.14it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.77it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.77it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.50it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.50it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:10,  4.79it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:10,  4.79it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:10,  4.79it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:07,  6.49it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:07,  6.49it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:07,  6.49it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:05,  8.06it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:05,  8.06it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.06it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.60it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.60it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.60it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.58it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.58it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.58it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.58it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 14.89it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 14.89it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 14.89it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 14.89it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 14.89it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.43it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:00, 36.69it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 47.02it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 47.02it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 47.02it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 47.02it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 47.02it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 47.02it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 47.02it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 47.02it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:06<00:00, 47.02it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:06<00:00, 55.19it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:06<00:00, 55.19it/s]

    Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:06<00:00, 55.19it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:06<00:00, 55.19it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:06<00:00, 55.19it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:06<00:00, 55.19it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:06<00:00, 55.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.89 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.89 GB):   2%|▏         | 1/58 [00:00<00:16,  3.35it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.86 GB):   2%|▏         | 1/58 [00:00<00:16,  3.35it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.86 GB):   3%|▎         | 2/58 [00:00<00:17,  3.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.86 GB):   3%|▎         | 2/58 [00:00<00:17,  3.17it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.86 GB):   5%|▌         | 3/58 [00:00<00:16,  3.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.87 GB):   5%|▌         | 3/58 [00:00<00:16,  3.37it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.87 GB):   7%|▋         | 4/58 [00:01<00:14,  3.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.87 GB):   7%|▋         | 4/58 [00:01<00:14,  3.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.87 GB):   9%|▊         | 5/58 [00:01<00:13,  3.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.87 GB):   9%|▊         | 5/58 [00:01<00:13,  3.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.87 GB):  10%|█         | 6/58 [00:01<00:12,  4.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.87 GB):  10%|█         | 6/58 [00:01<00:12,  4.24it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.87 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.88 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.88 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.88 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.29it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.88 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.89 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.89 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.89 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.34it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.89 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.89 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.89 GB):  21%|██        | 12/58 [00:02<00:06,  7.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.88 GB):  21%|██        | 12/58 [00:02<00:06,  7.52it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=43.88 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.89 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.73it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.89 GB):  24%|██▍       | 14/58 [00:02<00:07,  5.80it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.89 GB):  24%|██▍       | 14/58 [00:02<00:07,  5.80it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=43.89 GB):  26%|██▌       | 15/58 [00:02<00:07,  5.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.89 GB):  26%|██▌       | 15/58 [00:02<00:07,  5.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.89 GB):  28%|██▊       | 16/58 [00:03<00:07,  5.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.88 GB):  28%|██▊       | 16/58 [00:03<00:07,  5.30it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=43.88 GB):  29%|██▉       | 17/58 [00:03<00:07,  5.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.89 GB):  29%|██▉       | 17/58 [00:03<00:07,  5.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.89 GB):  31%|███       | 18/58 [00:03<00:07,  5.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.88 GB):  31%|███       | 18/58 [00:03<00:07,  5.37it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=43.88 GB):  33%|███▎      | 19/58 [00:03<00:07,  5.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.88 GB):  33%|███▎      | 19/58 [00:03<00:07,  5.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.88 GB):  34%|███▍      | 20/58 [00:03<00:06,  6.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.88 GB):  34%|███▍      | 20/58 [00:03<00:06,  6.01it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=43.88 GB):  36%|███▌      | 21/58 [00:03<00:05,  6.69it/s]Capturing num tokens (num_tokens=960 avail_mem=43.88 GB):  36%|███▌      | 21/58 [00:03<00:05,  6.69it/s] Capturing num tokens (num_tokens=960 avail_mem=43.88 GB):  38%|███▊      | 22/58 [00:04<00:05,  6.88it/s]Capturing num tokens (num_tokens=896 avail_mem=43.87 GB):  38%|███▊      | 22/58 [00:04<00:05,  6.88it/s]

    Capturing num tokens (num_tokens=896 avail_mem=43.87 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.50it/s]Capturing num tokens (num_tokens=832 avail_mem=43.87 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.50it/s]Capturing num tokens (num_tokens=832 avail_mem=43.87 GB):  41%|████▏     | 24/58 [00:04<00:04,  7.73it/s]Capturing num tokens (num_tokens=768 avail_mem=43.87 GB):  41%|████▏     | 24/58 [00:04<00:04,  7.73it/s]

    Capturing num tokens (num_tokens=768 avail_mem=43.87 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.78it/s]Capturing num tokens (num_tokens=704 avail_mem=43.86 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.78it/s]Capturing num tokens (num_tokens=704 avail_mem=43.86 GB):  45%|████▍     | 26/58 [00:04<00:04,  7.85it/s]Capturing num tokens (num_tokens=640 avail_mem=43.86 GB):  45%|████▍     | 26/58 [00:04<00:04,  7.85it/s]

    Capturing num tokens (num_tokens=640 avail_mem=43.86 GB):  47%|████▋     | 27/58 [00:04<00:03,  7.95it/s]Capturing num tokens (num_tokens=576 avail_mem=43.86 GB):  47%|████▋     | 27/58 [00:04<00:03,  7.95it/s]Capturing num tokens (num_tokens=576 avail_mem=43.86 GB):  48%|████▊     | 28/58 [00:04<00:03,  8.07it/s]Capturing num tokens (num_tokens=512 avail_mem=43.85 GB):  48%|████▊     | 28/58 [00:04<00:03,  8.07it/s]

    Capturing num tokens (num_tokens=512 avail_mem=43.85 GB):  50%|█████     | 29/58 [00:04<00:03,  8.33it/s]Capturing num tokens (num_tokens=480 avail_mem=43.85 GB):  50%|█████     | 29/58 [00:04<00:03,  8.33it/s]Capturing num tokens (num_tokens=480 avail_mem=43.85 GB):  52%|█████▏    | 30/58 [00:05<00:03,  8.59it/s]Capturing num tokens (num_tokens=448 avail_mem=43.85 GB):  52%|█████▏    | 30/58 [00:05<00:03,  8.59it/s]Capturing num tokens (num_tokens=416 avail_mem=43.84 GB):  52%|█████▏    | 30/58 [00:05<00:03,  8.59it/s]

    Capturing num tokens (num_tokens=416 avail_mem=43.84 GB):  55%|█████▌    | 32/58 [00:05<00:02, 10.69it/s]Capturing num tokens (num_tokens=384 avail_mem=43.84 GB):  55%|█████▌    | 32/58 [00:05<00:02, 10.69it/s]Capturing num tokens (num_tokens=352 avail_mem=43.83 GB):  55%|█████▌    | 32/58 [00:05<00:02, 10.69it/s]

    Capturing num tokens (num_tokens=352 avail_mem=43.83 GB):  59%|█████▊    | 34/58 [00:05<00:02, 10.11it/s]Capturing num tokens (num_tokens=320 avail_mem=43.83 GB):  59%|█████▊    | 34/58 [00:05<00:02, 10.11it/s]Capturing num tokens (num_tokens=288 avail_mem=43.83 GB):  59%|█████▊    | 34/58 [00:05<00:02, 10.11it/s]

    Capturing num tokens (num_tokens=288 avail_mem=43.83 GB):  62%|██████▏   | 36/58 [00:05<00:02,  9.92it/s]Capturing num tokens (num_tokens=256 avail_mem=43.82 GB):  62%|██████▏   | 36/58 [00:05<00:02,  9.92it/s]Capturing num tokens (num_tokens=240 avail_mem=43.82 GB):  62%|██████▏   | 36/58 [00:05<00:02,  9.92it/s]Capturing num tokens (num_tokens=240 avail_mem=43.82 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.95it/s]Capturing num tokens (num_tokens=224 avail_mem=43.82 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.95it/s]

    Capturing num tokens (num_tokens=208 avail_mem=43.81 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.95it/s]Capturing num tokens (num_tokens=192 avail_mem=43.81 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.95it/s]Capturing num tokens (num_tokens=192 avail_mem=43.81 GB):  71%|███████   | 41/58 [00:05<00:01, 12.05it/s]Capturing num tokens (num_tokens=176 avail_mem=43.80 GB):  71%|███████   | 41/58 [00:05<00:01, 12.05it/s]

    Capturing num tokens (num_tokens=160 avail_mem=43.80 GB):  71%|███████   | 41/58 [00:06<00:01, 12.05it/s]Capturing num tokens (num_tokens=160 avail_mem=43.80 GB):  74%|███████▍  | 43/58 [00:06<00:01, 11.36it/s]Capturing num tokens (num_tokens=144 avail_mem=43.80 GB):  74%|███████▍  | 43/58 [00:06<00:01, 11.36it/s]

    Capturing num tokens (num_tokens=128 avail_mem=43.81 GB):  74%|███████▍  | 43/58 [00:06<00:01, 11.36it/s]Capturing num tokens (num_tokens=128 avail_mem=43.81 GB):  78%|███████▊  | 45/58 [00:06<00:01, 10.99it/s]Capturing num tokens (num_tokens=112 avail_mem=43.80 GB):  78%|███████▊  | 45/58 [00:06<00:01, 10.99it/s]Capturing num tokens (num_tokens=96 avail_mem=43.80 GB):  78%|███████▊  | 45/58 [00:06<00:01, 10.99it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=43.80 GB):  81%|████████  | 47/58 [00:06<00:01, 10.68it/s]Capturing num tokens (num_tokens=80 avail_mem=43.79 GB):  81%|████████  | 47/58 [00:06<00:01, 10.68it/s]Capturing num tokens (num_tokens=64 avail_mem=43.79 GB):  81%|████████  | 47/58 [00:06<00:01, 10.68it/s]Capturing num tokens (num_tokens=64 avail_mem=43.79 GB):  84%|████████▍ | 49/58 [00:06<00:00, 10.54it/s]Capturing num tokens (num_tokens=48 avail_mem=43.79 GB):  84%|████████▍ | 49/58 [00:06<00:00, 10.54it/s]

    Capturing num tokens (num_tokens=32 avail_mem=43.78 GB):  84%|████████▍ | 49/58 [00:06<00:00, 10.54it/s]Capturing num tokens (num_tokens=32 avail_mem=43.78 GB):  88%|████████▊ | 51/58 [00:06<00:00, 10.49it/s]Capturing num tokens (num_tokens=28 avail_mem=43.78 GB):  88%|████████▊ | 51/58 [00:06<00:00, 10.49it/s]Capturing num tokens (num_tokens=24 avail_mem=43.78 GB):  88%|████████▊ | 51/58 [00:06<00:00, 10.49it/s]Capturing num tokens (num_tokens=20 avail_mem=43.77 GB):  88%|████████▊ | 51/58 [00:06<00:00, 10.49it/s]Capturing num tokens (num_tokens=16 avail_mem=43.77 GB):  88%|████████▊ | 51/58 [00:07<00:00, 10.49it/s]Capturing num tokens (num_tokens=12 avail_mem=43.77 GB):  88%|████████▊ | 51/58 [00:07<00:00, 10.49it/s]

    Capturing num tokens (num_tokens=12 avail_mem=43.77 GB):  97%|█████████▋| 56/58 [00:07<00:00, 16.99it/s]Capturing num tokens (num_tokens=8 avail_mem=43.76 GB):  97%|█████████▋| 56/58 [00:07<00:00, 16.99it/s] Capturing num tokens (num_tokens=4 avail_mem=43.76 GB):  97%|█████████▋| 56/58 [00:07<00:00, 16.99it/s]Capturing num tokens (num_tokens=4 avail_mem=43.76 GB): 100%|██████████| 58/58 [00:07<00:00,  8.15it/s]


    [2026-04-22 21:13:33] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


### JSON

you can directly define a JSON schema or use [Pydantic](https://docs.pydantic.dev/latest/) to define and validate the response.

**Using Pydantic**


```python
from pydantic import BaseModel, Field


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[
        {
            "role": "assistant",
            "content": "Give me the information and population of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=2048,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "foo",
            # convert the pydantic model to json schema
            "schema": CapitalInfo.model_json_schema(),
        },
    },
)

print_highlight(
    f"reasoing_content: {response.choices[0].message.reasoning_content}\n\ncontent: {response.choices[0].message.content}"
)
```


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should check if that's correct.<br><br>Wait, I think the population might have changed a bit over the years. I recall reading somewhere that Paris has grown a lot, especially with the influx of people moving there for work. But I'm not sure if it's exactly 21 million or maybe a bit more. I should look up the latest data to confirm.<br><br>I also wonder if the population figure includes just the city proper or the entire metropolitan area. Sometimes, people talk about the metro area, which can be much larger. But the question specifically asks for the population of the capital, so I think it refers to the city limits. Still, I should make sure.<br><br>Another thing to consider is that population figures can vary depending on the source. Some might cite estimates from government agencies, while others might use more recent data from censuses or surveys. I should find a reliable source to get the most accurate number.<br><br>I think the population of Paris is around 21 million, but I'm not 100% sure. Maybe I should think about other major cities in France to compare. For example, Lyon is another big city, but it's much smaller. I believe its population is around 1.2 million. That gives me a sense that Paris is significantly larger.<br><br>Also, considering the economic activities in Paris, like the fashion industry and the entertainment sector, it makes sense that it's the capital and has a large population. The city hosts a lot of events, conventions, and businesses, which would attract a diverse population.<br><br>I should also think about the historical growth of Paris. It's been a major city for centuries, so its population has been increasing steadily. I think it's safe to say that it's over 20 million, but I'm still not certain about the exact number.<br><br>In summary, I'm pretty confident that the capital of France is Paris, and its population is around 21 million. However, to be precise, I should look up the latest statistics to confirm the exact figure.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21538000<br>}</strong>


**JSON Schema Directly**



```python
import json

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[
        {
            "role": "assistant",
            "content": "Give me the information and population of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=2048,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "foo", "schema": json.loads(json_schema)},
    },
)

print_highlight(
    f"reasoing_content: {response.choices[0].message.reasoning_content}\n\ncontent: {response.choices[0].message.content}"
)
```


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should check if that's correct.<br><br>Wait, I think the population might have changed a bit over the years. I recall reading somewhere that it's over 21 million, but I'm not sure if it's 21.5 or 22 million. Also, I should consider whether the population figure is as of a specific year, like 2021 or 2022, because populations can increase over time.<br><br>I also wonder if there are any other factors to consider, like whether the population includes just the city proper or the broader metropolitan area. Sometimes, population figures can include surrounding regions. But I think in this case, since the user asked for the population of the capital, it's probably referring to the city limits.<br><br>Another thing to consider is the source of the data. Is it from a reliable government website or a recent census? I should make sure the information is up-to-date and accurate. Maybe I can cross-reference this with a recent source to confirm the population number.<br><br>So, putting it all together, I'm pretty confident that Paris is the capital of France, and its population is around 21.5 million. I'll go with that and present it in the JSON format as requested.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21500000<br>}</strong>


### EBNF


```python
ebnf_grammar = """
root ::= city | description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[
        {"role": "system", "content": "You are a helpful geography bot."},
        {
            "role": "assistant",
            "content": "Give me the information and population of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=2048,
    extra_body={"ebnf": ebnf_grammar},
)

print_highlight(
    f"reasoing_content: {response.choices[0].message.reasoning_content}\n\ncontent: {response.choices[0].message.content}"
)
```


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not entirely sure about the population number. I think it's a big city, so maybe around 3 million? But I'm not certain. I should probably double-check that. Maybe I can recall that Paris is one of the largest cities in Europe, so 3.5 million sounds about right. I don't think it's more than that because I've heard it's a major tourist attraction but not the largest in the world. So, I'll go with Paris having a population of approximately 3.5 million.<br><br><br>content: London is the capital of France</strong>


### Regular expression


```python
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[
        {"role": "assistant", "content": "What is the capital of France?"},
    ],
    temperature=0,
    max_tokens=2048,
    extra_body={"regex": "(Paris|London)"},
)

print_highlight(
    f"reasoing_content: {response.choices[0].message.reasoning_content}\n\ncontent: {response.choices[0].message.content}"
)
```


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light." People go there for museums, landmarks like the Eiffel Tower, and it's a cultural hub. But is it the capital?<br><br>Wait, I think the capital is the official seat of government, right? So maybe Paris is both the capital and the most famous city. But I'm not entirely certain. I recall that some countries have their capital in a different city than their main tourist attraction. For example, I think Brazil's capital is not Rio de Janeiro, which is more famous. So maybe France is like that too.<br><br>Let me try to remember any specific information. I think the French government declares Paris as the capital. I've heard that before. Also, I remember learning that the Eiffel Tower is in Paris, which is a symbol of the city, but not necessarily the government building. The government buildings are probably in another part of the city or in a different city altogether.<br><br>Wait, no, I think the government buildings are in Paris. Maybe the Palace of Consultation or something like that. I'm not sure of the exact name, but I know it's in Paris. So if the government is based there, then Paris would be the capital. That makes sense because the capital is where all the official government work happens.<br><br>I also think that other capitals, like Germany's Berlin or Italy's Rome, are more than just tourist spots. So, yes, Paris being the capital of France fits into that pattern. It's both the main city and the government center.<br><br>I guess I'm pretty confident now. Paris is the capital of France because it's where the government is located, even though it's also a major tourist destination. So the answer should be Paris.<br><br><br>content: Paris</strong>


### Structural Tag


```python
tool_get_current_weather = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "state", "unit"],
        },
    },
}

tool_get_current_date = {
    "type": "function",
    "function": {
        "name": "get_current_date",
        "description": "Get the current date and time for a given timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                }
            },
            "required": ["timezone"],
        },
    },
}

schema_get_current_weather = tool_get_current_weather["function"]["parameters"]
schema_get_current_date = tool_get_current_date["function"]["parameters"]


def get_messages():
    return [
        {
            "role": "system",
            "content": f"""
# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant.""",
        },
        {
            "role": "assistant",
            "content": "You are in New York. Please get the current date and time, and the weather.",
        },
    ]


messages = get_messages()

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=messages,
    response_format={
        "type": "structural_tag",
        "max_new_tokens": 2048,
        "structures": [
            {
                "begin": "<function=get_current_weather>",
                "schema": schema_get_current_weather,
                "end": "</function>",
            },
            {
                "begin": "<function=get_current_date>",
                "schema": schema_get_current_date,
                "end": "</function>",
            },
        ],
        "triggers": ["<function="],
    },
)

print_highlight(
    f"reasoing_content: {response.choices[0].message.reasoning_content}\n\ncontent: {response.choices[0].message.content}"
)
```


<strong style='color: #00008B;'>reasoing_content: Okay, the user is in New York and wants the current date and time along with the weather. I need to figure out how to get both pieces of information using the functions provided.<br><br>First, for the date and time, I should use the get_current_date function. The required parameter is the timezone, which in this case is 'America/New_York'. So I'll structure the function call with that parameter.<br><br>Next, for the weather, I'll use get_current_weather. The parameters needed are city, state, and unit. The city is 'New York', the state is 'NY', and the unit should be 'fahrenheit' since the user didn't specify. I'll include all these in the function call.<br><br>I need to make sure each function call is separate and follows the specified format. So I'll first call get_current_date with the timezone parameter, then get_current_weather with the city, state, and unit parameters. Each function call will be on its own line with the correct syntax.<br><br>I should also remember to include the source links for each function in the response. For get_current_date, the OpenWeatherMap is the source, and for get_current_weather, it's also from OpenWeatherMap but under the 'Weather' section.<br><br>Putting it all together, I'll format each function call correctly, ensuring the parameters are JSON objects within the function syntax. This way, the user gets both the current date and time in New York and the weather details in Fahrenheit.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>Source: <br>- For get_current_date: [OpenWeatherMap](https://openweathermap.org/api)<br>- For get_current_weather: [OpenWeatherMap](https://openweathermap.org/api)</strong>


## Native API and SGLang Runtime (SRT)

> Note: For native API, as a work-around, you need to set `require_reasoning` argument to `True` to ensure the model will think before generating the structured output. It's not required for chat-completion API.

### JSON

**Using Pydantic**


```python
import requests
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


messages = [
    {
        "role": "assistant",
        "content": "Give me the information and population of the capital of France in the JSON format.",
    },
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
# Make API request
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "require_reasoning": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 2048,
            "json_schema": json.dumps(CapitalInfo.model_json_schema()),
        },
    },
)
print(response.json())


reasoing_content = response.json()["text"].split("</think>")[0]
content = response.json()["text"].split("</think>")[1]
print_highlight(f"reasoing_content: {reasoing_content}\n\ncontent: {content}")
```

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not exactly sure of the exact number. I think it\'s around 2 million, but I should double-check that.\n\nWait, maybe I should look up the latest statistics to be accurate. I recall that in recent years, Paris has seen a steady increase due to urban development and immigration. Let me see... I think the population is approximately 2,170,000 as of 2023. That seems about right.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the city name, its population, and maybe the country it\'s in. So, the keys could be "city", "population", and "country".\n\nPutting it all together, the JSON should look something like this: {"city": "Paris", "population": 2170000, "country": "France"}. I should make sure the syntax is correct, with proper commas and quotation marks.\n\nI should also consider if the user might need more details, like the metropolitan area population or other statistics, but since they specifically asked for the capital, I\'ll stick to that. Maybe mention that the data is up to date as of 2023 to provide context.\n\nAlright, I think that\'s all. I\'ll present the JSON and offer further assistance in case they need more information.\n</think>{\n  "name": "Paris",\n  "population": 2170000\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 382, 14190, 11, 7196, 358, 1265, 1401, 705, 279, 5535, 13142, 311, 387, 13382, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 24020, 5263, 4152, 311, 15662, 4401, 323, 15093, 13, 6771, 752, 1490, 1112, 358, 1744, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 3283, 829, 11, 1181, 7042, 11, 323, 7196, 279, 3146, 432, 594, 304, 13, 2055, 11, 279, 6894, 1410, 387, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 11436, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 5212, 8926, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 11, 330, 11141, 788, 330, 49000, 1, 7810, 358, 1265, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 57406, 3082, 7042, 476, 1008, 13142, 11, 714, 2474, 807, 11689, 4588, 369, 279, 6722, 11, 358, 3278, 9214, 311, 429, 13, 10696, 6286, 429, 279, 821, 374, 705, 311, 2400, 438, 315, 220, 17, 15, 17, 18, 311, 3410, 2266, 382, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3042, 279, 4718, 323, 3010, 4623, 12994, 304, 1142, 807, 1184, 803, 1995, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 198, 92, 151643], 'meta_info': {'id': 'b85478d2b86140ae8f152956a66dd06f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 369, 'completion_tokens': 392, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.1836919682100415, 'response_sent_to_client_ts': 1776892440.904833}}



<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.<br><br>First, I need to identify the capital of France. I know that Paris is the capital, so that's straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I'm not exactly sure of the exact number. I think it's around 2 million, but I should double-check that.<br><br>Wait, maybe I should look up the latest statistics to be accurate. I recall that in recent years, Paris has seen a steady increase due to urban development and immigration. Let me see... I think the population is approximately 2,170,000 as of 2023. That seems about right.<br><br>Now, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the city name, its population, and maybe the country it's in. So, the keys could be "city", "population", and "country".<br><br>Putting it all together, the JSON should look something like this: {"city": "Paris", "population": 2170000, "country": "France"}. I should make sure the syntax is correct, with proper commas and quotation marks.<br><br>I should also consider if the user might need more details, like the metropolitan area population or other statistics, but since they specifically asked for the capital, I'll stick to that. Maybe mention that the data is up to date as of 2023 to provide context.<br><br>Alright, I think that's all. I'll present the JSON and offer further assistance in case they need more information.<br><br><br>content: {<br>  "name": "Paris",<br>  "population": 2170000<br>}</strong>


**JSON Schema Directly**


```python
json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

# JSON
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "require_reasoning": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 2048,
            "json_schema": json_schema,
        },
    },
)

print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see... Yes, according to the latest data, the population is approximately 2,174,300 as of 2023. That seems accurate.\n\nNext, I need to structure this information into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and "country". The city is Paris, the population is the number I found, and the country is France.\n\nI should make sure the JSON syntax is correct. Each key should be in quotes, and the values as well. Also, the entire structure should be enclosed in curly braces. I\'ll format it properly to ensure there are no syntax errors.\n\nPutting it all together, the JSON object will have the city, population, and country. I\'ll present this to the user, making sure it\'s clear and easy to understand. I don\'t think the user needs anything else, but if they have more questions, I can provide additional information.\n\nI should also consider if the user might need the population figure in a different format, like a string or an integer, but since they asked for JSON, I\'ll stick with the number as an integer. \n\nAlright, I think that\'s all. I\'ll format the JSON correctly and present it to the user.\n</think>{\n\n"name": "Paris",\n"population": 2174300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 1112, 7414, 11, 4092, 311, 279, 5535, 821, 11, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 13382, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 3263, 576, 3283, 374, 12095, 11, 279, 7042, 374, 279, 1372, 358, 1730, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 8886, 1376, 1265, 387, 304, 17194, 11, 323, 279, 2750, 438, 1632, 13, 7281, 11, 279, 4453, 5944, 1265, 387, 43810, 304, 68103, 59191, 13, 358, 3278, 3561, 432, 10277, 311, 5978, 1052, 525, 902, 19482, 5975, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 686, 614, 279, 3283, 11, 7042, 11, 323, 3146, 13, 358, 3278, 3042, 419, 311, 279, 1196, 11, 3259, 2704, 432, 594, 2797, 323, 4135, 311, 3535, 13, 358, 1513, 944, 1744, 279, 1196, 3880, 4113, 770, 11, 714, 421, 807, 614, 803, 4755, 11, 358, 646, 3410, 5107, 1995, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 279, 7042, 7071, 304, 264, 2155, 3561, 11, 1075, 264, 914, 476, 458, 7546, 11, 714, 2474, 807, 4588, 369, 4718, 11, 358, 3278, 9214, 448, 279, 1372, 438, 458, 7546, 13, 4710, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3561, 279, 4718, 12440, 323, 3042, 432, 311, 279, 1196, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '3be4903c82684a25aa119b36ba97d50b', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 383, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 17.515601872000843, 'response_sent_to_client_ts': 1776892458.4335775}}</strong>


### EBNF


```python
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": "Give me the information of the capital of France.",
        "require_reasoning": True,
        "sampling_params": {
            "max_new_tokens": 2048,
            "temperature": 0,
            "n": 3,
            "ebnf": (
                "root ::= city | description\n"
                'city ::= "London" | "Paris" | "Berlin" | "Rome"\n'
                'description ::= city " is " status\n'
                'status ::= "the capital of " country\n'
                'country ::= "England" | "France" | "Germany" | "Italy"'
            ),
        },
        "stream": False,
        "return_logprob": False,
    },
)

print(response.json())
```

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'fcb9cc88ab2b452c910a70417191fd19', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.08433928480371833, 'response_sent_to_client_ts': 1776892458.5445166}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '81d89cb091be41cd83e672ba7fa38955', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.08425461268052459, 'response_sent_to_client_ts': 1776892458.544532}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '5aac59dc8ffa45f7bfe9d4852d560b08', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.08419433096423745, 'response_sent_to_client_ts': 1776892458.5445383}}]


### Regular expression


```python
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": "Paris is the capital of",
        "require_reasoning": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 2048,
            "regex": "(France|England)",
        },
    },
)
print(response.json())
```

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '7f3e2be68cb64ae99d5b59530565a429', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 15.300480643752962, 'response_sent_to_client_ts': 1776892473.8518815}}


### Structural Tag


```python
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
payload = {
    "text": text,
    "require_reasoning": True,
    "sampling_params": {
        "max_new_tokens": 2048,
        "structural_tag": json.dumps(
            {
                "type": "structural_tag",
                "structures": [
                    {
                        "begin": "<function=get_current_weather>",
                        "schema": schema_get_current_weather,
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "schema": schema_get_current_date,
                        "end": "</function>",
                    },
                ],
                "triggers": ["<function="],
            }
        ),
    },
}


# Send POST request to the API endpoint
response = requests.post(f"http://localhost:{port}/generate", json=payload)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'Okay, so I just got this query asking for the information and population of the capital of France in JSON format. Hmm, let me break this down.\n\nFirst off, I need to figure out what the user is actually looking for. They mention both information and population. I\'m assuming they want a brief overview, probably the most recent stats available. The capital city is straightforward—Paris.\n\nNow, considering the user wants this in JSON format, they\'re likely looking for structured data, maybe to integrate it into an app or use it for display purposes. So, I\'ll need to make sure the response is concise but comprehensive.\n\nPopulation stats can be tricky because they change over time. I should probably state that the numbers here are approximate and suggest checking the latest sources like Wikipedia for the most accurate info. That shows I\'m being helpful without overcommitting to specific figures.\n\nAlso, the user\'s request has been in French, so it\'s possible they\'re a French speaker or using a French app. I should keep the response clear and straightforward, avoiding any jargon that might confuse them.\n\nPutting it all together, I\'ll structure the JSON with a general info section that includes population,_surface_area, founded_year, and some notable landmarks. Then, a more detailed population section that breaks it down by age, sex, and migration trends. Adding another field for sources and notes ensures transparency about the data\'s reliability.\n\nI need to make sure the JSON syntax is correct—proper commas, quotes, and braces. Any mistake there could render the response invalid. So, I\'ll double-check that before finalizing.\n\nFinally, I\'ll write a brief response acknowledging the request, providing the JSON, and adding a friendly note about the data being approximate. That way, I cover all bases and make the user feel informed.\n</think>\n\nHere is the request for the information and population of the capital of France in JSON format:\n\n```json\n{\n  "general_info": {\n    "capital": "Paris",\n    "population": "Approximately 2,192,887 (as of 2023)",\n    "surface_area": "105.53 square kilometers",\n    "founded": "1347",\n    "notable_landmarks": ["Eiffel Tower", "Curie\t    \t\t Clipboarduento de France.diffusion Edifficultyku Lyon", "Louvre Museum"]\n  },\n  "population details": {\n    "total": "Approximately 2,192,887",\n    "male": "Approximately 1,095,000",\n    "female": "Approximately 1,097,887",\n    "aged_16-65": "Approximately 94% of the population",\n    "migrants": "Regular and irregular migration continues to impact the city\'s demographics."\n  },\n  "sources": "This information is approximate and may vary depending on the source. For the most accurate and up-to-date figures, consult official records or recent publications."\n}\n```\n\nThis JSON provides a general overview of Paris\'s information and population statistics. Population numbers are approximate and may vary depending on the source. For the most accurate and up-to-date figures, consult official records or recent publications.', 'output_ids': [32313, 11, 773, 358, 1101, 2684, 419, 3239, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 88190, 11, 1077, 752, 1438, 419, 1495, 382, 5338, 1007, 11, 358, 1184, 311, 7071, 700, 1128, 279, 1196, 374, 3520, 3330, 369, 13, 2379, 6286, 2176, 1995, 323, 7042, 13, 358, 2776, 25538, 807, 1366, 264, 9814, 23251, 11, 4658, 279, 1429, 3213, 10472, 2500, 13, 576, 6722, 3283, 374, 30339, 2293, 59604, 382, 7039, 11, 12831, 279, 1196, 6801, 419, 304, 4718, 3561, 11, 807, 2299, 4363, 3330, 369, 32930, 821, 11, 7196, 311, 31072, 432, 1119, 458, 906, 476, 990, 432, 369, 3037, 9895, 13, 2055, 11, 358, 3278, 1184, 311, 1281, 2704, 279, 2033, 374, 63594, 714, 15817, 382, 53371, 10472, 646, 387, 33453, 1576, 807, 2297, 916, 882, 13, 358, 1265, 4658, 1584, 429, 279, 5109, 1588, 525, 44868, 323, 4190, 13295, 279, 5535, 8173, 1075, 26587, 369, 279, 1429, 13382, 3546, 13, 2938, 4933, 358, 2776, 1660, 10950, 2041, 916, 17413, 1280, 311, 3151, 12396, 382, 13394, 11, 279, 1196, 594, 1681, 702, 1012, 304, 8585, 11, 773, 432, 594, 3204, 807, 2299, 264, 8585, 18601, 476, 1667, 264, 8585, 906, 13, 358, 1265, 2506, 279, 2033, 2797, 323, 30339, 11, 30426, 894, 502, 70821, 429, 2578, 58117, 1105, 382, 97904, 432, 678, 3786, 11, 358, 3278, 5944, 279, 4718, 448, 264, 4586, 3546, 3772, 429, 5646, 7042, 11, 31030, 15030, 11, 18047, 14645, 11, 323, 1045, 27190, 59924, 13, 5005, 11, 264, 803, 11682, 7042, 3772, 429, 18303, 432, 1495, 553, 4231, 11, 1839, 11, 323, 11906, 18339, 13, 30370, 2441, 2070, 369, 8173, 323, 8388, 25351, 27231, 911, 279, 821, 594, 30538, 382, 40, 1184, 311, 1281, 2704, 279, 4718, 19482, 374, 4396, 2293, 80668, 76602, 11, 17194, 11, 323, 59191, 13, 5765, 16523, 1052, 1410, 3141, 279, 2033, 8318, 13, 2055, 11, 358, 3278, 1990, 15934, 429, 1573, 1590, 4849, 382, 23949, 11, 358, 3278, 3270, 264, 9814, 2033, 60608, 279, 1681, 11, 8241, 279, 4718, 11, 323, 7842, 264, 11657, 5185, 911, 279, 821, 1660, 44868, 13, 2938, 1616, 11, 358, 3421, 678, 23092, 323, 1281, 279, 1196, 2666, 15987, 624, 151649, 271, 8420, 374, 279, 1681, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 24595, 3109, 788, 341, 262, 330, 65063, 788, 330, 59604, 756, 262, 330, 44441, 788, 330, 69520, 7108, 220, 17, 11, 16, 24, 17, 11, 23, 23, 22, 320, 300, 315, 220, 17, 15, 17, 18, 15752, 262, 330, 39746, 15030, 788, 330, 16, 15, 20, 13, 20, 18, 9334, 40568, 756, 262, 330, 69, 13082, 788, 330, 16, 18, 19, 22, 756, 262, 330, 1921, 480, 60506, 15544, 788, 4383, 36, 3092, 301, 21938, 497, 330, 16704, 645, 57041, 81327, 55288, 409, 9625, 40679, 7560, 468, 71864, 12133, 55201, 497, 330, 92806, 48506, 16328, 7026, 220, 1153, 220, 330, 44441, 3565, 788, 341, 262, 330, 5035, 788, 330, 69520, 7108, 220, 17, 11, 16, 24, 17, 11, 23, 23, 22, 756, 262, 330, 36476, 788, 330, 69520, 7108, 220, 16, 11, 15, 24, 20, 11, 15, 15, 15, 756, 262, 330, 42634, 788, 330, 69520, 7108, 220, 16, 11, 15, 24, 22, 11, 23, 23, 22, 756, 262, 330, 3279, 62, 16, 21, 12, 21, 20, 788, 330, 69520, 7108, 220, 24, 19, 4, 315, 279, 7042, 756, 262, 330, 76, 70236, 788, 330, 30404, 323, 41308, 11906, 9539, 311, 5421, 279, 3283, 594, 62234, 10040, 220, 1153, 220, 330, 39651, 788, 330, 1986, 1995, 374, 44868, 323, 1231, 13289, 11649, 389, 279, 2530, 13, 1752, 279, 1429, 13382, 323, 705, 4686, 18413, 12396, 11, 8498, 3946, 7424, 476, 3213, 27985, 10040, 532, 13874, 19324, 1986, 4718, 5707, 264, 4586, 23251, 315, 12095, 594, 1995, 323, 7042, 13142, 13, 39529, 5109, 525, 44868, 323, 1231, 13289, 11649, 389, 279, 2530, 13, 1752, 279, 1429, 13382, 323, 705, 4686, 18413, 12396, 11, 8498, 3946, 7424, 476, 3213, 27985, 13, 151643], 'meta_info': {'id': '734259f7f2464c95b380ecdd743001c1', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 365, 'completion_tokens': 670, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.684849270153791, 'response_sent_to_client_ts': 1776892479.5449142}}</strong>



```python
terminate_process(server_process)
```

## Offline Engine API


```python
import sglang as sgl

llm = sgl.Engine(
    model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    reasoning_parser="deepseek-r1",
    grammar_backend="xgrammar",
)
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
    [2026-04-22 21:14:54] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.55s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.53s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.54s/it]


    2026-04-22 21:15:03,375 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 21:15:03] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:47,  2.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:47,  2.94s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:57,  1.05s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:57,  1.05s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:42,  1.26it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:42,  1.26it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:27,  1.87it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:27,  1.87it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:23,  2.20it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:23,  2.20it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:19,  2.51it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:19,  2.51it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:17,  2.87it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:17,  2.87it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:14,  3.25it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:14,  3.25it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:12,  3.63it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:12,  3.63it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:11,  3.99it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:11,  3.99it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:09,  4.87it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:09,  4.87it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:07,  5.88it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:07,  5.88it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:06,  6.41it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:06,  6.41it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:06,  6.41it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:07<00:05,  7.69it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:07<00:05,  7.69it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:07<00:05,  7.69it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:07<00:04,  9.04it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:07<00:04,  9.04it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:07<00:04,  9.04it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:03, 10.85it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:03, 10.85it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:03, 10.85it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:07<00:02, 12.63it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:07<00:02, 12.63it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:07<00:02, 12.63it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:07<00:02, 12.63it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:07<00:02, 14.91it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:07<00:02, 14.91it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:07<00:02, 14.91it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:08<00:02, 14.91it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:08<00:01, 17.80it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:08<00:01, 17.80it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:08<00:01, 17.80it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:08<00:01, 18.30it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:08<00:01, 18.30it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:08<00:01, 18.30it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:08<00:01, 18.30it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:08<00:01, 20.34it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:08<00:01, 20.34it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:08<00:01, 20.34it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:08<00:01, 20.34it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:08<00:01, 20.34it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:08<00:00, 25.09it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:08<00:00, 25.09it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:08<00:00, 25.09it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:08<00:00, 25.09it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:08<00:00, 25.09it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:08<00:00, 26.86it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:08<00:00, 26.86it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:08<00:00, 26.86it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:08<00:00, 26.86it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:08<00:00, 26.86it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:08<00:00, 28.49it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:08<00:00, 28.49it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:08<00:00, 28.49it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:08<00:00, 28.49it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:08<00:00, 28.49it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:08<00:00, 30.99it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:08<00:00, 30.99it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:08<00:00, 30.99it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:08<00:00, 30.99it/s]

    Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:08<00:00, 30.99it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:08<00:00, 30.99it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:08<00:00, 34.18it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:08<00:00, 34.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=18.98 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=18.98 GB):   2%|▏         | 1/58 [00:00<00:16,  3.35it/s]Capturing num tokens (num_tokens=7680 avail_mem=18.95 GB):   2%|▏         | 1/58 [00:00<00:16,  3.35it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=18.95 GB):   3%|▎         | 2/58 [00:00<00:24,  2.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=17.92 GB):   3%|▎         | 2/58 [00:00<00:24,  2.31it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=17.92 GB):   5%|▌         | 3/58 [00:01<00:23,  2.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=18.10 GB):   5%|▌         | 3/58 [00:01<00:23,  2.37it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=18.10 GB):   7%|▋         | 4/58 [00:01<00:23,  2.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=18.93 GB):   7%|▋         | 4/58 [00:01<00:23,  2.28it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=18.93 GB):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=18.17 GB):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=18.17 GB):  10%|█         | 6/58 [00:02<00:21,  2.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=18.23 GB):  10%|█         | 6/58 [00:02<00:21,  2.40it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=18.23 GB):  12%|█▏        | 7/58 [00:02<00:20,  2.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=18.95 GB):  12%|█▏        | 7/58 [00:02<00:20,  2.48it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=18.95 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=18.34 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.63it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=18.34 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=18.36 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.88it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=18.36 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=18.87 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.12it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=18.87 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=18.95 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.14it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=18.95 GB):  21%|██        | 12/58 [00:04<00:15,  3.06it/s]Capturing num tokens (num_tokens=3072 avail_mem=18.42 GB):  21%|██        | 12/58 [00:04<00:15,  3.06it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=18.42 GB):  22%|██▏       | 13/58 [00:04<00:13,  3.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=18.48 GB):  22%|██▏       | 13/58 [00:04<00:13,  3.40it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=18.48 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=18.93 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=18.93 GB):  26%|██▌       | 15/58 [00:05<00:10,  4.05it/s]Capturing num tokens (num_tokens=2304 avail_mem=18.51 GB):  26%|██▌       | 15/58 [00:05<00:10,  4.05it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=18.51 GB):  28%|██▊       | 16/58 [00:05<00:09,  4.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=18.54 GB):  28%|██▊       | 16/58 [00:05<00:09,  4.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=18.54 GB):  29%|██▉       | 17/58 [00:05<00:08,  4.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=18.92 GB):  29%|██▉       | 17/58 [00:05<00:08,  4.98it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=18.92 GB):  31%|███       | 18/58 [00:05<00:08,  4.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=18.56 GB):  31%|███       | 18/58 [00:05<00:08,  4.64it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=18.56 GB):  33%|███▎      | 19/58 [00:05<00:08,  4.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=18.59 GB):  33%|███▎      | 19/58 [00:05<00:08,  4.65it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=18.59 GB):  34%|███▍      | 20/58 [00:06<00:08,  4.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=18.92 GB):  34%|███▍      | 20/58 [00:06<00:08,  4.59it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=18.92 GB):  36%|███▌      | 21/58 [00:06<00:08,  4.54it/s]Capturing num tokens (num_tokens=960 avail_mem=18.61 GB):  36%|███▌      | 21/58 [00:06<00:08,  4.54it/s] Capturing num tokens (num_tokens=896 avail_mem=18.64 GB):  36%|███▌      | 21/58 [00:06<00:08,  4.54it/s]Capturing num tokens (num_tokens=896 avail_mem=18.64 GB):  40%|███▉      | 23/58 [00:06<00:05,  6.23it/s]Capturing num tokens (num_tokens=832 avail_mem=18.91 GB):  40%|███▉      | 23/58 [00:06<00:05,  6.23it/s]

    Capturing num tokens (num_tokens=768 avail_mem=18.65 GB):  40%|███▉      | 23/58 [00:06<00:05,  6.23it/s]Capturing num tokens (num_tokens=768 avail_mem=18.65 GB):  43%|████▎     | 25/58 [00:06<00:04,  7.82it/s]Capturing num tokens (num_tokens=704 avail_mem=18.90 GB):  43%|████▎     | 25/58 [00:06<00:04,  7.82it/s]

    Capturing num tokens (num_tokens=704 avail_mem=18.90 GB):  45%|████▍     | 26/58 [00:06<00:04,  7.25it/s]Capturing num tokens (num_tokens=640 avail_mem=42.10 GB):  45%|████▍     | 26/58 [00:06<00:04,  7.25it/s]Capturing num tokens (num_tokens=576 avail_mem=42.09 GB):  45%|████▍     | 26/58 [00:06<00:04,  7.25it/s]Capturing num tokens (num_tokens=576 avail_mem=42.09 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.86it/s]Capturing num tokens (num_tokens=512 avail_mem=41.92 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.86it/s]Capturing num tokens (num_tokens=480 avail_mem=42.08 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.86it/s]

    Capturing num tokens (num_tokens=480 avail_mem=42.08 GB):  52%|█████▏    | 30/58 [00:07<00:02, 10.89it/s]Capturing num tokens (num_tokens=448 avail_mem=42.08 GB):  52%|█████▏    | 30/58 [00:07<00:02, 10.89it/s]Capturing num tokens (num_tokens=416 avail_mem=42.08 GB):  52%|█████▏    | 30/58 [00:07<00:02, 10.89it/s]Capturing num tokens (num_tokens=416 avail_mem=42.08 GB):  55%|█████▌    | 32/58 [00:07<00:02, 12.37it/s]Capturing num tokens (num_tokens=384 avail_mem=41.96 GB):  55%|█████▌    | 32/58 [00:07<00:02, 12.37it/s]Capturing num tokens (num_tokens=352 avail_mem=41.95 GB):  55%|█████▌    | 32/58 [00:07<00:02, 12.37it/s]

    Capturing num tokens (num_tokens=320 avail_mem=42.05 GB):  55%|█████▌    | 32/58 [00:07<00:02, 12.37it/s]Capturing num tokens (num_tokens=320 avail_mem=42.05 GB):  60%|██████    | 35/58 [00:07<00:01, 14.86it/s]Capturing num tokens (num_tokens=288 avail_mem=42.05 GB):  60%|██████    | 35/58 [00:07<00:01, 14.86it/s]Capturing num tokens (num_tokens=256 avail_mem=42.04 GB):  60%|██████    | 35/58 [00:07<00:01, 14.86it/s]Capturing num tokens (num_tokens=240 avail_mem=42.03 GB):  60%|██████    | 35/58 [00:07<00:01, 14.86it/s]Capturing num tokens (num_tokens=240 avail_mem=42.03 GB):  66%|██████▌   | 38/58 [00:07<00:01, 17.01it/s]Capturing num tokens (num_tokens=224 avail_mem=41.97 GB):  66%|██████▌   | 38/58 [00:07<00:01, 17.01it/s]

    Capturing num tokens (num_tokens=208 avail_mem=42.02 GB):  66%|██████▌   | 38/58 [00:07<00:01, 17.01it/s]Capturing num tokens (num_tokens=192 avail_mem=42.01 GB):  66%|██████▌   | 38/58 [00:07<00:01, 17.01it/s]Capturing num tokens (num_tokens=192 avail_mem=42.01 GB):  71%|███████   | 41/58 [00:07<00:00, 18.74it/s]Capturing num tokens (num_tokens=176 avail_mem=42.00 GB):  71%|███████   | 41/58 [00:07<00:00, 18.74it/s]Capturing num tokens (num_tokens=160 avail_mem=42.00 GB):  71%|███████   | 41/58 [00:07<00:00, 18.74it/s]Capturing num tokens (num_tokens=144 avail_mem=41.99 GB):  71%|███████   | 41/58 [00:07<00:00, 18.74it/s]

    Capturing num tokens (num_tokens=144 avail_mem=41.99 GB):  76%|███████▌  | 44/58 [00:07<00:00, 20.19it/s]Capturing num tokens (num_tokens=128 avail_mem=41.96 GB):  76%|███████▌  | 44/58 [00:07<00:00, 20.19it/s]Capturing num tokens (num_tokens=112 avail_mem=41.99 GB):  76%|███████▌  | 44/58 [00:07<00:00, 20.19it/s]Capturing num tokens (num_tokens=96 avail_mem=41.98 GB):  76%|███████▌  | 44/58 [00:07<00:00, 20.19it/s] Capturing num tokens (num_tokens=96 avail_mem=41.98 GB):  81%|████████  | 47/58 [00:07<00:00, 21.85it/s]Capturing num tokens (num_tokens=80 avail_mem=41.97 GB):  81%|████████  | 47/58 [00:07<00:00, 21.85it/s]Capturing num tokens (num_tokens=64 avail_mem=41.97 GB):  81%|████████  | 47/58 [00:07<00:00, 21.85it/s]Capturing num tokens (num_tokens=48 avail_mem=41.96 GB):  81%|████████  | 47/58 [00:07<00:00, 21.85it/s]

    Capturing num tokens (num_tokens=48 avail_mem=41.96 GB):  86%|████████▌ | 50/58 [00:07<00:00, 23.05it/s]Capturing num tokens (num_tokens=32 avail_mem=41.95 GB):  86%|████████▌ | 50/58 [00:07<00:00, 23.05it/s]Capturing num tokens (num_tokens=28 avail_mem=41.93 GB):  86%|████████▌ | 50/58 [00:07<00:00, 23.05it/s]Capturing num tokens (num_tokens=24 avail_mem=41.94 GB):  86%|████████▌ | 50/58 [00:07<00:00, 23.05it/s]Capturing num tokens (num_tokens=20 avail_mem=41.93 GB):  86%|████████▌ | 50/58 [00:08<00:00, 23.05it/s]Capturing num tokens (num_tokens=20 avail_mem=41.93 GB):  93%|█████████▎| 54/58 [00:08<00:00, 25.21it/s]Capturing num tokens (num_tokens=16 avail_mem=41.92 GB):  93%|█████████▎| 54/58 [00:08<00:00, 25.21it/s]Capturing num tokens (num_tokens=12 avail_mem=41.92 GB):  93%|█████████▎| 54/58 [00:08<00:00, 25.21it/s]

    Capturing num tokens (num_tokens=8 avail_mem=41.91 GB):  93%|█████████▎| 54/58 [00:08<00:00, 25.21it/s] Capturing num tokens (num_tokens=8 avail_mem=41.91 GB):  98%|█████████▊| 57/58 [00:08<00:00, 26.24it/s]Capturing num tokens (num_tokens=4 avail_mem=41.90 GB):  98%|█████████▊| 57/58 [00:08<00:00, 26.24it/s]Capturing num tokens (num_tokens=4 avail_mem=41.90 GB): 100%|██████████| 58/58 [00:08<00:00,  7.10it/s]


### JSON

**Using Pydantic**


```python
import json
from pydantic import BaseModel, Field

prompts = [
    "Give me the information of the capital of China in the JSON format.",
    "Give me the information of the capital of France in the JSON format.",
    "Give me the information of the capital of Ireland in the JSON format.",
]


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


sampling_params = {
    "temperature": 0,
    "top_p": 0.95,
    "max_new_tokens": 2048,
    "json_schema": json.dumps(CapitalInfo.model_json_schema()),
}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

    ===============================
    Prompt: Give me the information of the capital of China in the JSON format.
    Generated text: {
      "name": "Beijing",
      "population": 316000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    ===============================
    Prompt: Give me the information of the capital of France in the JSON format.
    Generated text: {
      "name": "Paris",
      "population": 2154000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    ===============================
    Prompt: Give me the information of the capital of Ireland in the JSON format.
    Generated text: {
      "name": "Ireland",
      "population": 500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000


**JSON Schema Directly**


```python
prompts = [
    "Give me the information of the capital of China in the JSON format.",
    "Give me the information of the capital of France in the JSON format.",
    "Give me the information of the capital of Ireland in the JSON format.",
]

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

sampling_params = {"temperature": 0, "max_new_tokens": 2048, "json_schema": json_schema}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

    ===============================
    Prompt: Give me the information of the capital of China in the JSON format.
    Generated text: {
      "name": "Beijing",
      "population": 316000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    ===============================
    Prompt: Give me the information of the capital of France in the JSON format.
    Generated text: {
      "name": "Paris",
      "population": 2154000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    ===============================
    Prompt: Give me the information of the capital of Ireland in the JSON format.
    Generated text: {
      "name": "Ireland",
      "population": 500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000


### EBNF



```python
prompts = [
    "Give me the information of the capital of France.",
    "Give me the information of the capital of Germany.",
    "Give me the information of the capital of Italy.",
]

sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "ebnf": (
        "root ::= city | description\n"
        'city ::= "London" | "Paris" | "Berlin" | "Rome"\n'
        'description ::= city " is " status\n'
        'status ::= "the capital of " country\n'
        'country ::= "England" | "France" | "Germany" | "Italy"'
    ),
}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

    ===============================
    Prompt: Give me the information of the capital of France.
    Generated text: Berlin is the capital of France
    ===============================
    Prompt: Give me the information of the capital of Germany.
    Generated text: Berlin is the capital of Germany
    ===============================
    Prompt: Give me the information of the capital of Italy.
    Generated text: Berlin is the capital of Germany


### Regular expression


```python
prompts = [
    "Please provide information about London as a major global city:",
    "Please provide information about Paris as a major global city:",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95, "regex": "(France|England)"}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

    ===============================
    Prompt: Please provide information about London as a major global city:
    Generated text: France
    ===============================
    Prompt: Please provide information about Paris as a major global city:
    Generated text: France



```python
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
prompts = [text]


sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 2048,
    "structural_tag": json.dumps(
        {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": "<function=get_current_weather>",
                    "schema": schema_get_current_weather,
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "schema": schema_get_current_date,
                    "end": "</function>",
                },
            ],
            "triggers": ["<function="],
        }
    ),
}


# Send POST request to the API endpoint
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

    ===============================
    Prompt: <｜begin▁of▁sentence｜><｜Assistant｜>Give me the information and population of the capital of France in the JSON format.<｜end▁of▁sentence｜><｜Assistant｜><think>
    
    Generated text: Alright, so the user asked for the information and population of the capital of France in JSON format. Hmm, okay, the capital is Paris. I need to make sure I get all the accurate details. Let me think about what information is typically included. Population is one thing, but also maybe the administrative region, area, and some notable landmarks.
    
    Wait, do I have the current population? I think the population of Paris is over 2 million now. Let me double-check that number to be sure. Yeah, I believe it's around 2,155,000 as of 2023. That seems up-to-date.
    
    What about the administrative region? Paris is in the Ile-de-France region. I should include that. The area is usually given in both square kilometers and square miles for clarity. So, about 105 square kilometers or roughly 40 square miles. That's correct.
    
    Notable landmarks are a good touch. The Eiffel Tower is iconic, so I'll include that. The Louvre Museum is another must. Also, maybe mention the City of Light to highlight its nickname. Including those will give the user a nice overview.
    
    Putting this all together, I need to structure it in JSON. So, the keys would be "AdministrativeRegion", "Area", and "NotableLandmarks". Each key will have its corresponding value. I should make sure the JSON syntax is correct, with proper commas and quotation marks.
    
    Wait, should I include the population as a number or maybe a formatted number with commas for readability? JSON can handle numbers without commas, so it's fine as is. But maybe I should write it as 2,155,000 to make it clear it's a large number.
    
    Also, should I include any units? For area, I already specified square kilometers and square miles, so that's covered. I think that's all the information the user is asking for. I should present it neatly in a JSON structure without any extra fluff.
    
    Let me put it all together now. The key points are Paris as the capital, its population, administrative region, area, and notable landmarks. I think that's comprehensive. I'll make sure the JSON is properly formatted and easy to read.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "AdministrativeRegion": "Ile-de-France",
      "Area": {
        "SquareKilometers": 105.2,
        "SquareMiles": 40.75
      },
      "Population": 2155000,
      "NotableLandmarks": [
        "Eiffel Tower",
        "Louvre Museum",
        "City of Light"
      ]
    }
    ```



```python
llm.shutdown()
```
