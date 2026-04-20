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
    [2026-04-20 23:11:43] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-20 23:11:45] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-20 23:11:45] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-20 23:11:47] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-20 23:11:53] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-20 23:11:54] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False
    [2026-04-20 23:11:54] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.07s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.17s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.15s/it]


    2026-04-20 23:12:01,588 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-20 23:12:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:54,  3.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:54,  3.07s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:23,  1.49s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:23,  1.49s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.87it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.87it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.61it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.61it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.39it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.39it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.28it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.28it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.28it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.95it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.95it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:06,  6.89it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:06,  6.89it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  6.58it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  6.58it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:06,  6.47it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:06,  6.47it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:06,  6.64it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:06,  6.64it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:06,  6.89it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:06,  6.89it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:05,  7.42it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:05,  7.42it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:05,  7.42it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:04,  8.50it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:03,  9.66it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:03,  9.66it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:03,  9.66it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:06<00:03, 11.25it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:06<00:03, 11.25it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:06<00:03, 11.25it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:06<00:02, 12.82it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:06<00:02, 12.82it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:06<00:02, 12.82it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:06<00:02, 14.46it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:06<00:02, 14.46it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:06<00:02, 14.46it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:06<00:02, 14.46it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:06<00:01, 16.98it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:06<00:01, 16.98it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:06<00:01, 16.98it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:06<00:01, 16.98it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:06<00:01, 18.96it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:06<00:01, 18.96it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:06<00:01, 18.96it/s]

    Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:06<00:01, 18.96it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:06<00:00, 24.57it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:06<00:00, 24.57it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:06<00:00, 24.57it/s]

    Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:06<00:00, 24.57it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 25.31it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 25.31it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 25.31it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 25.31it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:06<00:00, 26.18it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:06<00:00, 26.18it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:06<00:00, 26.18it/s]

    Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:07<00:00, 26.18it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:07<00:00, 26.99it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:07<00:00, 26.99it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:07<00:00, 26.99it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:07<00:00, 26.99it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:07<00:00, 26.99it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:07<00:00, 28.63it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:07<00:00, 28.63it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:07<00:00, 28.63it/s]

    Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:07<00:00, 28.63it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:07<00:00, 28.63it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:07<00:00, 31.35it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:07<00:00, 31.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=86.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=86.22 GB):   2%|▏         | 1/58 [00:00<00:34,  1.67it/s]Capturing num tokens (num_tokens=7680 avail_mem=86.19 GB):   2%|▏         | 1/58 [00:00<00:34,  1.67it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=86.19 GB):   3%|▎         | 2/58 [00:01<00:28,  1.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=86.19 GB):   3%|▎         | 2/58 [00:01<00:28,  1.94it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=86.19 GB):   5%|▌         | 3/58 [00:01<00:24,  2.24it/s]Capturing num tokens (num_tokens=6656 avail_mem=86.19 GB):   5%|▌         | 3/58 [00:01<00:24,  2.24it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=86.19 GB):   7%|▋         | 4/58 [00:01<00:21,  2.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=86.19 GB):   7%|▋         | 4/58 [00:01<00:21,  2.57it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=86.19 GB):   9%|▊         | 5/58 [00:01<00:17,  3.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=86.19 GB):   9%|▊         | 5/58 [00:01<00:17,  3.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=86.19 GB):  10%|█         | 6/58 [00:02<00:14,  3.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=86.19 GB):  10%|█         | 6/58 [00:02<00:14,  3.59it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=86.19 GB):  12%|█▏        | 7/58 [00:02<00:13,  3.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=85.01 GB):  12%|█▏        | 7/58 [00:02<00:13,  3.73it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=85.01 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=86.17 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.59it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=86.17 GB):  16%|█▌        | 9/58 [00:02<00:13,  3.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=85.19 GB):  16%|█▌        | 9/58 [00:02<00:13,  3.68it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=85.19 GB):  17%|█▋        | 10/58 [00:03<00:13,  3.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=86.17 GB):  17%|█▋        | 10/58 [00:03<00:13,  3.60it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=86.17 GB):  19%|█▉        | 11/58 [00:03<00:12,  3.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=85.25 GB):  19%|█▉        | 11/58 [00:03<00:12,  3.65it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=85.25 GB):  21%|██        | 12/58 [00:03<00:12,  3.56it/s]Capturing num tokens (num_tokens=3072 avail_mem=84.32 GB):  21%|██        | 12/58 [00:03<00:12,  3.56it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=84.32 GB):  22%|██▏       | 13/58 [00:04<00:14,  3.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=84.32 GB):  22%|██▏       | 13/58 [00:04<00:14,  3.15it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=84.32 GB):  24%|██▍       | 14/58 [00:04<00:14,  3.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=84.39 GB):  24%|██▍       | 14/58 [00:04<00:14,  3.09it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=84.39 GB):  26%|██▌       | 15/58 [00:04<00:14,  3.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=85.19 GB):  26%|██▌       | 15/58 [00:04<00:14,  3.04it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=85.19 GB):  28%|██▊       | 16/58 [00:05<00:13,  3.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=84.46 GB):  28%|██▊       | 16/58 [00:05<00:13,  3.14it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=84.46 GB):  29%|██▉       | 17/58 [00:05<00:12,  3.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=85.19 GB):  29%|██▉       | 17/58 [00:05<00:12,  3.28it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=85.19 GB):  31%|███       | 18/58 [00:05<00:11,  3.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=84.52 GB):  31%|███       | 18/58 [00:05<00:11,  3.42it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=84.52 GB):  33%|███▎      | 19/58 [00:05<00:10,  3.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=84.59 GB):  33%|███▎      | 19/58 [00:05<00:10,  3.67it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=84.59 GB):  34%|███▍      | 20/58 [00:06<00:10,  3.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=84.59 GB):  34%|███▍      | 20/58 [00:06<00:10,  3.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=84.59 GB):  36%|███▌      | 21/58 [00:06<00:08,  4.18it/s]Capturing num tokens (num_tokens=960 avail_mem=84.66 GB):  36%|███▌      | 21/58 [00:06<00:08,  4.18it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=84.66 GB):  38%|███▊      | 22/58 [00:06<00:08,  4.25it/s]Capturing num tokens (num_tokens=896 avail_mem=85.19 GB):  38%|███▊      | 22/58 [00:06<00:08,  4.25it/s]Capturing num tokens (num_tokens=896 avail_mem=85.19 GB):  40%|███▉      | 23/58 [00:06<00:07,  4.58it/s]Capturing num tokens (num_tokens=832 avail_mem=84.69 GB):  40%|███▉      | 23/58 [00:06<00:07,  4.58it/s]

    Capturing num tokens (num_tokens=832 avail_mem=84.69 GB):  41%|████▏     | 24/58 [00:06<00:07,  4.68it/s]Capturing num tokens (num_tokens=768 avail_mem=85.18 GB):  41%|████▏     | 24/58 [00:06<00:07,  4.68it/s]Capturing num tokens (num_tokens=768 avail_mem=85.18 GB):  43%|████▎     | 25/58 [00:07<00:06,  4.88it/s]Capturing num tokens (num_tokens=704 avail_mem=84.71 GB):  43%|████▎     | 25/58 [00:07<00:06,  4.88it/s]

    Capturing num tokens (num_tokens=704 avail_mem=84.71 GB):  45%|████▍     | 26/58 [00:07<00:06,  5.15it/s]Capturing num tokens (num_tokens=640 avail_mem=85.17 GB):  45%|████▍     | 26/58 [00:07<00:06,  5.15it/s]Capturing num tokens (num_tokens=640 avail_mem=85.17 GB):  47%|████▋     | 27/58 [00:07<00:05,  5.26it/s]Capturing num tokens (num_tokens=576 avail_mem=84.73 GB):  47%|████▋     | 27/58 [00:07<00:05,  5.26it/s]

    Capturing num tokens (num_tokens=576 avail_mem=84.73 GB):  48%|████▊     | 28/58 [00:07<00:05,  5.68it/s]Capturing num tokens (num_tokens=512 avail_mem=84.76 GB):  48%|████▊     | 28/58 [00:07<00:05,  5.68it/s]Capturing num tokens (num_tokens=512 avail_mem=84.76 GB):  50%|█████     | 29/58 [00:07<00:05,  5.73it/s]Capturing num tokens (num_tokens=480 avail_mem=85.16 GB):  50%|█████     | 29/58 [00:07<00:05,  5.73it/s]

    Capturing num tokens (num_tokens=480 avail_mem=85.16 GB):  52%|█████▏    | 30/58 [00:07<00:04,  5.96it/s]Capturing num tokens (num_tokens=448 avail_mem=84.79 GB):  52%|█████▏    | 30/58 [00:07<00:04,  5.96it/s]Capturing num tokens (num_tokens=448 avail_mem=84.79 GB):  53%|█████▎    | 31/58 [00:08<00:04,  6.17it/s]Capturing num tokens (num_tokens=416 avail_mem=85.15 GB):  53%|█████▎    | 31/58 [00:08<00:04,  6.17it/s]

    Capturing num tokens (num_tokens=416 avail_mem=85.15 GB):  55%|█████▌    | 32/58 [00:08<00:04,  6.15it/s]Capturing num tokens (num_tokens=384 avail_mem=85.15 GB):  55%|█████▌    | 32/58 [00:08<00:04,  6.15it/s]Capturing num tokens (num_tokens=384 avail_mem=85.15 GB):  57%|█████▋    | 33/58 [00:08<00:03,  6.44it/s]Capturing num tokens (num_tokens=352 avail_mem=84.84 GB):  57%|█████▋    | 33/58 [00:08<00:03,  6.44it/s]

    Capturing num tokens (num_tokens=352 avail_mem=84.84 GB):  59%|█████▊    | 34/58 [00:08<00:03,  6.52it/s]Capturing num tokens (num_tokens=320 avail_mem=85.14 GB):  59%|█████▊    | 34/58 [00:08<00:03,  6.52it/s]Capturing num tokens (num_tokens=320 avail_mem=85.14 GB):  60%|██████    | 35/58 [00:08<00:03,  6.67it/s]Capturing num tokens (num_tokens=288 avail_mem=85.14 GB):  60%|██████    | 35/58 [00:08<00:03,  6.67it/s]

    Capturing num tokens (num_tokens=288 avail_mem=85.14 GB):  62%|██████▏   | 36/58 [00:08<00:03,  6.97it/s]Capturing num tokens (num_tokens=256 avail_mem=84.90 GB):  62%|██████▏   | 36/58 [00:08<00:03,  6.97it/s]Capturing num tokens (num_tokens=256 avail_mem=84.90 GB):  64%|██████▍   | 37/58 [00:08<00:02,  7.44it/s]Capturing num tokens (num_tokens=240 avail_mem=85.12 GB):  64%|██████▍   | 37/58 [00:08<00:02,  7.44it/s]

    Capturing num tokens (num_tokens=240 avail_mem=85.12 GB):  66%|██████▌   | 38/58 [00:09<00:02,  7.88it/s]Capturing num tokens (num_tokens=224 avail_mem=85.12 GB):  66%|██████▌   | 38/58 [00:09<00:02,  7.88it/s]Capturing num tokens (num_tokens=224 avail_mem=85.12 GB):  67%|██████▋   | 39/58 [00:09<00:02,  8.03it/s]Capturing num tokens (num_tokens=208 avail_mem=85.12 GB):  67%|██████▋   | 39/58 [00:09<00:02,  8.03it/s]

    Capturing num tokens (num_tokens=192 avail_mem=84.96 GB):  67%|██████▋   | 39/58 [00:09<00:02,  8.03it/s]Capturing num tokens (num_tokens=192 avail_mem=84.96 GB):  71%|███████   | 41/58 [00:09<00:01,  9.20it/s]Capturing num tokens (num_tokens=176 avail_mem=84.98 GB):  71%|███████   | 41/58 [00:09<00:01,  9.20it/s]Capturing num tokens (num_tokens=160 avail_mem=85.10 GB):  71%|███████   | 41/58 [00:09<00:01,  9.20it/s]

    Capturing num tokens (num_tokens=160 avail_mem=85.10 GB):  74%|███████▍  | 43/58 [00:09<00:01, 10.26it/s]Capturing num tokens (num_tokens=144 avail_mem=85.09 GB):  74%|███████▍  | 43/58 [00:09<00:01, 10.26it/s]Capturing num tokens (num_tokens=128 avail_mem=85.10 GB):  74%|███████▍  | 43/58 [00:09<00:01, 10.26it/s]Capturing num tokens (num_tokens=128 avail_mem=85.10 GB):  78%|███████▊  | 45/58 [00:09<00:01, 11.37it/s]Capturing num tokens (num_tokens=112 avail_mem=85.09 GB):  78%|███████▊  | 45/58 [00:09<00:01, 11.37it/s]

    Capturing num tokens (num_tokens=96 avail_mem=85.08 GB):  78%|███████▊  | 45/58 [00:09<00:01, 11.37it/s] Capturing num tokens (num_tokens=96 avail_mem=85.08 GB):  81%|████████  | 47/58 [00:09<00:00, 12.40it/s]Capturing num tokens (num_tokens=80 avail_mem=85.01 GB):  81%|████████  | 47/58 [00:09<00:00, 12.40it/s]Capturing num tokens (num_tokens=64 avail_mem=84.98 GB):  81%|████████  | 47/58 [00:09<00:00, 12.40it/s]Capturing num tokens (num_tokens=48 avail_mem=85.00 GB):  81%|████████  | 47/58 [00:09<00:00, 12.40it/s]

    Capturing num tokens (num_tokens=48 avail_mem=85.00 GB):  86%|████████▌ | 50/58 [00:09<00:00, 14.46it/s]Capturing num tokens (num_tokens=32 avail_mem=85.00 GB):  86%|████████▌ | 50/58 [00:09<00:00, 14.46it/s]Capturing num tokens (num_tokens=28 avail_mem=85.01 GB):  86%|████████▌ | 50/58 [00:10<00:00, 14.46it/s]Capturing num tokens (num_tokens=28 avail_mem=85.01 GB):  90%|████████▉ | 52/58 [00:10<00:00, 15.40it/s]Capturing num tokens (num_tokens=24 avail_mem=85.04 GB):  90%|████████▉ | 52/58 [00:10<00:00, 15.40it/s]Capturing num tokens (num_tokens=20 avail_mem=85.03 GB):  90%|████████▉ | 52/58 [00:10<00:00, 15.40it/s]

    Capturing num tokens (num_tokens=16 avail_mem=85.02 GB):  90%|████████▉ | 52/58 [00:10<00:00, 15.40it/s]Capturing num tokens (num_tokens=16 avail_mem=85.02 GB):  95%|█████████▍| 55/58 [00:10<00:00, 17.36it/s]Capturing num tokens (num_tokens=12 avail_mem=85.01 GB):  95%|█████████▍| 55/58 [00:10<00:00, 17.36it/s]Capturing num tokens (num_tokens=8 avail_mem=85.01 GB):  95%|█████████▍| 55/58 [00:10<00:00, 17.36it/s] Capturing num tokens (num_tokens=4 avail_mem=85.00 GB):  95%|█████████▍| 55/58 [00:10<00:00, 17.36it/s]Capturing num tokens (num_tokens=4 avail_mem=85.00 GB): 100%|██████████| 58/58 [00:10<00:00, 20.02it/s]Capturing num tokens (num_tokens=4 avail_mem=85.00 GB): 100%|██████████| 58/58 [00:10<00:00,  5.63it/s]


    [2026-04-20 23:12:21] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think I heard somewhere that Paris has a population over 21 million. Maybe 21.6 million? I'm not sure if that's accurate. I should look up the latest data to confirm. Let me think, where can I find reliable information? Maybe the official government website or a reputable news source. <br><br>I recall that France's population is around 40 million, so Paris being a major city would have a significant portion of that. If the total population is about 40 million, and Paris is the largest city, it's plausible that it's around 21.6 million. I think I've seen that number before, but I'm not 100% sure. <br><br>Also, I should consider if the population figure includes just the city proper or the metropolitan area. Sometimes, population counts can include surrounding suburbs and satellite towns. But I think in this case, the user is asking for the population of the capital, which is Paris, so it's probably just the city limits. <br><br>I should also think about how populations can change over time. Demographics can fluctuate due to births, deaths, and migration. So the number might not be exact and could vary slightly from year to year. <br><br>To sum up, I'm pretty confident that Paris is the capital of France and that its population is approximately 21.6 million. But to be thorough, I should verify this information to ensure accuracy.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21620000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think I heard somewhere that Paris has a population over 21 million. Maybe 21.6 million? I'm not sure if that's accurate. I should look up the latest data to confirm. Let me think, where can I find reliable information? Maybe the official government website or a reputable news source. <br><br>I recall that France's population is around 40 million, so Paris being a major city would have a significant portion of that. If the total population is about 40 million, and Paris is the largest city, it's plausible that it's around 21.6 million. I think I've seen that number before, but I'm not 100% sure. <br><br>Also, I should consider if the population figure includes just the city proper or the metropolitan area. Sometimes, population counts can include surrounding suburbs and satellite towns. But I think in this case, the user is asking for the population of the capital, which is Paris, so it's probably just the city limits. <br><br>I should also think about how populations can change over time. Demographics can fluctuate due to births, deaths, and migration. So the number might not be exact and could vary slightly from year to year. <br><br>To sum up, I'm pretty confident that Paris is the capital of France and that its population is approximately 21.6 million. But to be thorough, I should verify this information to ensure accuracy.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21620000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population. I think it's a big city, maybe around 3 million? But I'm not certain. I should probably check some reliable sources to confirm this. Maybe I can look up recent population data or news articles that mention Paris's population. I remember hearing that Paris is one of the most populous cities in the world, but I'm not sure if it's over 3 million or not. I should also consider factors like urbanization and migration that might affect the population numbers. Maybe the population has grown a bit since the last census. I think the most recent data might be from 2020 or 2021. I should make sure the number I provide is accurate and up-to-date. Also, I should present this information in a clear and concise way, maybe in JSON format as the user requested. I should double-check the population figure to ensure it's correct before finalizing the answer.<br><br><br>content: Rome is the capital of France</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning a bit about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light" because of the famous Eiffel Tower. But is it the capital? I think so, but I'm not entirely certain. <br><br>Wait, I also recall that there's another city called Lyon. Isn't that the capital? No, I think I'm mixing things up. Lyon is definitely a significant city in France, known for its gastronomy and being the second-largest city. But I'm pretty sure Paris is the capital. <br><br>Let me try to remember any other capitals I know. For example, Germany's capital is Berlin, Italy's is Rome, Spain's is Madrid, and so on. So, following that pattern, France's capital should be Paris. I think I heard it a lot in history classes, especially when talking about the French Revolution and Napoleon. Those events happened in Paris, which probably helped it become the capital.<br><br>I also remember that the Eiffel Tower is in Paris, and it's a symbol of the country. The tower was built in the 19th century, and it's a tourist attraction. So, if Paris has such a famous landmark, it's likely the capital. <br><br>Another way to think about it is the political aspect. The President of France is based in Paris, right? So that makes sense. The government quarters, like the Palace of Versailles, are in Paris. That would mean Paris is where the country's government is located, making it the capital.<br><br>I guess I'm pretty confident now. I don't think I've heard of any other city being the capital of France. Lyon is more of a regional capital or something. Maybe it's the regional capital for certain areas, but not the national one. <br><br>So, putting it all together, Paris is the capital of France because it's the most significant political, cultural, and symbolic center of the country. It's where major landmarks like the Eiffel Tower and government buildings are located, and it's the birthplace of many important historical events and figures.<br><br><br>content: Paris</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants the current date and time along with the weather. I need to figure out how to get this information using the provided functions.<br><br>First, I should use the get_current_date function. The parameters required are the timezone. Since the user is in New York, I'll set the timezone parameter to 'America/New_York'. That should give me the current date and time in their local time zone.<br><br>Next, I need to get the weather. The get_current_weather function requires the city, state, and unit. The city is 'New York', the state abbreviation is 'NY', and the unit should be 'fahrenheit' since the user might prefer Fahrenheit. I'll structure the parameters accordingly.<br><br>I should make sure to call each function separately and include the parameters in the correct format. I'll start with get_current_date, then move on to get_current_weather. I'll include the sources in the response to show where the information came from.<br><br>Putting it all together, I'll write the function calls with the appropriate parameters and add the sources to each part of the answer. That should cover everything the user is asking for.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>  <br><br>Source: get_current_date function uses the 'get_current_date' function with the timezone parameter set to 'America/New_York'. Source: [get_current_date function documentation](https://docs.example.com/get_current_date).  <br>Source: get_current_weather function uses the 'get_current_weather' function with city 'New York', state 'NY', and unit 'fahrenheit'. Source: [get_current_weather function documentation](https://docs.example.com/get_current_weather).</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nWait, maybe I should check the latest statistics. I recall that in recent years, Paris has seen a slight increase. Let me think... I believe the population is approximately 2,150,000 as of 2023. That seems about right, but I\'m not 100% certain. I should make sure to present this information accurately.\n\nNext, I need to structure this into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" for context. The city is Paris, the population is 2,150,000, and the country is France.\n\nI should also consider if the user might need more details, like the exact year of the population figure. Including that could be helpful, so I\'ll add "year": 2023. That way, the user knows the data is up to date.\n\nPutting it all together, the JSON should look clean and well-structured. I\'ll make sure the syntax is correct, with proper commas and quotation marks. No markdown, just plain JSON.\n\nI think that\'s all. The user probably just needs the information quickly, so keeping it concise is key. I\'ll present the JSON without any extra fluff.\n</think>{\n\n"name": "Paris",\n"population": 21500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 14190, 11, 7196, 358, 1265, 1779, 279, 5535, 13142, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 8112, 5263, 13, 6771, 752, 1744, 1112, 358, 4411, 279, 7042, 374, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 3654, 13, 358, 1265, 1281, 2704, 311, 3042, 419, 1995, 29257, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 369, 2266, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 4734, 1042, 315, 279, 7042, 7071, 13, 55121, 429, 1410, 387, 10950, 11, 773, 358, 3278, 912, 330, 3157, 788, 220, 17, 15, 17, 18, 13, 2938, 1616, 11, 279, 1196, 8788, 279, 821, 374, 705, 311, 2400, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 4240, 323, 1632, 12, 51143, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 13, 2308, 50494, 11, 1101, 14396, 4718, 382, 40, 1744, 429, 594, 678, 13, 576, 1196, 4658, 1101, 3880, 279, 1995, 6157, 11, 773, 10282, 432, 63594, 374, 1376, 13, 358, 3278, 3042, 279, 4718, 2041, 894, 4960, 1320, 1362, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '59c83b014f0d473cb14a382d37e204ea', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 367, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.963245110586286, 'response_sent_to_client_ts': 1776726788.5063884}}



<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.<br><br>First, I need to identify the capital of France. I know that Paris is the capital, so that's straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I'm not sure of the exact number. I think it's around 2 million, but I should verify that.<br><br>Wait, maybe I should check the latest statistics. I recall that in recent years, Paris has seen a slight increase. Let me think... I believe the population is approximately 2,150,000 as of 2023. That seems about right, but I'm not 100% certain. I should make sure to present this information accurately.<br><br>Next, I need to structure this into a JSON format. JSON requires key-value pairs, so I'll create an object with keys like "city", "population", and maybe "country" for context. The city is Paris, the population is 2,150,000, and the country is France.<br><br>I should also consider if the user might need more details, like the exact year of the population figure. Including that could be helpful, so I'll add "year": 2023. That way, the user knows the data is up to date.<br><br>Putting it all together, the JSON should look clean and well-structured. I'll make sure the syntax is correct, with proper commas and quotation marks. No markdown, just plain JSON.<br><br>I think that's all. The user probably just needs the information quickly, so keeping it concise is key. I'll present the JSON without any extra fluff.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000</strong>


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should verify that.\n\nWait, maybe I should check some reliable sources to confirm the population. I recall that the population figures can vary depending on the source and the year. For example, the 2020 census might have a slightly different number than the 2021 estimate. I think the population was around 2,165,000 in 2021, but I\'m not 100% certain. I should make sure to use the most accurate and up-to-date information.\n\nAlso, the user wants the information in JSON format. JSON is a data interchange format, so I\'ll need to structure the data accordingly. I should include the city name, population, and maybe the year of the data. It\'s important to present the information clearly and accurately, so I\'ll double-check the numbers to avoid any mistakes.\n\nI should also consider if there are any other relevant details the user might find useful, like the area of the city or some key facts about it. But since the user specifically asked for population, I\'ll focus on that. Maybe adding a note about the population figure being approximate would be helpful, just in case.\n\nPutting it all together, I\'ll structure the JSON with the city name, population, and the year. I\'ll make sure the syntax is correct, using quotation marks and commas appropriately. I\'ll also keep the language clear and straightforward so that the user can easily understand the information.\n\nFinally, I\'ll review the JSON to ensure there are no errors and that the data is accurate. This way, the user gets a reliable and well-formatted response to their query.\n</think>{"name": "Paris", "population": 2165000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 14190, 11, 7196, 358, 1265, 1779, 1045, 14720, 8173, 311, 7683, 279, 7042, 13, 358, 19091, 429, 279, 7042, 12396, 646, 13289, 11649, 389, 279, 2530, 323, 279, 1042, 13, 1752, 3110, 11, 279, 220, 17, 15, 17, 15, 43602, 2578, 614, 264, 10078, 2155, 1372, 1091, 279, 220, 17, 15, 17, 16, 16045, 13, 358, 1744, 279, 7042, 572, 2163, 220, 17, 11, 16, 21, 20, 11, 15, 15, 15, 304, 220, 17, 15, 17, 16, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 3654, 13, 358, 1265, 1281, 2704, 311, 990, 279, 1429, 13382, 323, 705, 4686, 18413, 1995, 382, 13394, 11, 279, 1196, 6801, 279, 1995, 304, 4718, 3561, 13, 4718, 374, 264, 821, 51263, 3561, 11, 773, 358, 3278, 1184, 311, 5944, 279, 821, 27079, 13, 358, 1265, 2924, 279, 3283, 829, 11, 7042, 11, 323, 7196, 279, 1042, 315, 279, 821, 13, 1084, 594, 2989, 311, 3042, 279, 1995, 9355, 323, 29257, 11, 773, 358, 3278, 1990, 15934, 279, 5109, 311, 5648, 894, 20643, 382, 40, 1265, 1083, 2908, 421, 1052, 525, 894, 1008, 9760, 3565, 279, 1196, 2578, 1477, 5390, 11, 1075, 279, 3082, 315, 279, 3283, 476, 1045, 1376, 13064, 911, 432, 13, 1988, 2474, 279, 1196, 11689, 4588, 369, 7042, 11, 358, 3278, 5244, 389, 429, 13, 10696, 7842, 264, 5185, 911, 279, 7042, 7071, 1660, 44868, 1035, 387, 10950, 11, 1101, 304, 1142, 382, 97904, 432, 678, 3786, 11, 358, 3278, 5944, 279, 4718, 448, 279, 3283, 829, 11, 7042, 11, 323, 279, 1042, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 1667, 54231, 15423, 323, 76602, 34901, 13, 358, 3278, 1083, 2506, 279, 4128, 2797, 323, 30339, 773, 429, 279, 1196, 646, 6707, 3535, 279, 1995, 382, 23949, 11, 358, 3278, 3395, 279, 4718, 311, 5978, 1052, 525, 902, 5975, 323, 429, 279, 821, 374, 13382, 13, 1096, 1616, 11, 279, 1196, 5221, 264, 14720, 323, 1632, 8460, 12127, 2033, 311, 862, 3239, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 21, 20, 15, 15, 15, 92, 151643], 'meta_info': {'id': '5e81d3974ac64b8a9aeede4d62638e7d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 428, 'completion_tokens': 447, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.8452998315915465, 'response_sent_to_client_ts': 1776726793.364361}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '26f52e80d6c2457883e1b297df7dde05', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11838617827743292, 'response_sent_to_client_ts': 1776726793.5153134}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '50a9517e450545dc9c31f4aa91c9b678', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11831965111196041, 'response_sent_to_client_ts': 1776726793.5153272}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '47c6d93217544a6b91eb5529fa98475b', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11699402332305908, 'response_sent_to_client_ts': 1776726793.5153315}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '8c011d5a8c614d56a6dfb31af0d89780', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.535003520548344, 'response_sent_to_client_ts': 1776726813.0582933}}


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


<strong style='color: #00008B;'>{'text': 'Alright, so I need to figure out the population of the capital of France. Hmm, the capital of France is Paris, right? Okay, so I\'m looking for the population of Paris. I remember hearing that Paris is the most populous city in France, but I\'m not exactly sure about the exact number.\n\nI think the population figures change every year because people move in and out of cities. I should probably check a reliable source or maybe look up the most recent data available. I know that France\'s national population is around 40 million, but Paris is much bigger. I remember a time when I saw a statistic that said Paris had over 12 million people, but I\'m not sure if that\'s still accurate.\n\nWait, maybe I should consider the metropolitan area instead of the city proper. Sometimes, the metropolitan area includes a larger area surrounding the city, which would include more people. I think the metropolitan area of Paris is bigger than the city itself. I\'m not certain about the exact figures, though.\n\nI should also think about whether the number includes both residents and those who are temporary residents. Probably, it includes everyone living in the area. Another thing to consider is that cities can have significant expatriates, so the population number might not be the same as the city\'s resident population.\n\nI recall reading somewhere that the population of Paris was around 12 million a few years back, but I think that number has grown. Maybe around 12.5 million now? But I\'m not sure. I should try to visualize the current statistics or think about recent news if any have been published.\n\nAlternatively, perhaps I can think about other cities in France and their populations to get a sense. For example, Lyon is a major city, but its population is smaller. Maybe around 1.2 million? And then there\'s Marseille, which might be a few hundred thousand. So, Paris being the largest makes sense.\n\nWait, but the metropolitan area of Paris is often cited as having a much larger population. I think it\'s around 25 million people. So, if the user is asking for just the city\'s population, it\'s probably around 12 million, but if they\'re referring to the metropolitan area, it\'s 25 million.\n\nI should verify if the initial question was about the city or the metropolitan area. The question says, "the capital of France" which is Paris, but I\'m not sure if they meant the city proper or the broader metropolitan area. If I go with just the city, it\'s around 12 million, but I should probably mention both to cover all bases.\n\nAlso, I should consider that population figures are approximate and can vary depending on the source. For example, the World Bank might have slightly different numbers compared to other organizations like UNESCO or even local authorities.\n\nIn summary, I\'m pretty confident that Paris has a population of over 12 million, and possibly around 12.5 million to 13 million. The metropolitan area is significantly larger, around 25 million, which includes a much broader geographic range beyond just the city limits.\n\nI should probably present both numbers in the response so the user knows that the capital of France can be on either end depending on what they\'re looking for. Also, mentioning that the exact figure can vary is important to avoid giving the wrong information.\n\nAnother consideration is whether the user is looking for the current year\'s data. Population numbers can fluctuate annually, so if they need the most recent figure, they might have to check an up-to-date source. However, I think the figure I have is recent enough, given that I haven\'t heard any major changes in recent years.\n\nIn any case, providing both the city and metropolitan area figures seems like a good approach. I\'ll structure it into two sections to make it clear, and maybe mention that the urban area population is approximately 12 million, while the metropolitan area is around 25 million.\n\nSo, to sum up, the capital, Paris, has a population of about 12 million in the city proper, and around 25 million in its metropolitan area. This way, the user can get a better understanding of the population numbers depending on their specific interest.\n</think>\n\nThe capital of France, Paris, has a population of approximately 12 million in the city proper, and about 25 million in its metropolitan area. These figures provide a clear distinction between the city and the broader metropolitan region.', 'output_ids': [71486, 11, 773, 358, 1184, 311, 7071, 700, 279, 7042, 315, 279, 6722, 315, 9625, 13, 88190, 11, 279, 6722, 315, 9625, 374, 12095, 11, 1290, 30, 35439, 11, 773, 358, 2776, 3330, 369, 279, 7042, 315, 12095, 13, 358, 6099, 10778, 429, 12095, 374, 279, 1429, 94451, 3283, 304, 9625, 11, 714, 358, 2776, 537, 6896, 2704, 911, 279, 4734, 1372, 382, 40, 1744, 279, 7042, 12396, 2297, 1449, 1042, 1576, 1251, 3271, 304, 323, 700, 315, 9720, 13, 358, 1265, 4658, 1779, 264, 14720, 2530, 476, 7196, 1401, 705, 279, 1429, 3213, 821, 2500, 13, 358, 1414, 429, 9625, 594, 5313, 7042, 374, 2163, 220, 19, 15, 3526, 11, 714, 12095, 374, 1753, 11243, 13, 358, 6099, 264, 882, 979, 358, 5485, 264, 42489, 429, 1053, 12095, 1030, 916, 220, 16, 17, 3526, 1251, 11, 714, 358, 2776, 537, 2704, 421, 429, 594, 2058, 13382, 382, 14190, 11, 7196, 358, 1265, 2908, 279, 57406, 3082, 4518, 315, 279, 3283, 6169, 13, 17688, 11, 279, 57406, 3082, 5646, 264, 8131, 3082, 14590, 279, 3283, 11, 892, 1035, 2924, 803, 1251, 13, 358, 1744, 279, 57406, 3082, 315, 12095, 374, 11243, 1091, 279, 3283, 5086, 13, 358, 2776, 537, 3654, 911, 279, 4734, 12396, 11, 3498, 382, 40, 1265, 1083, 1744, 911, 3425, 279, 1372, 5646, 2176, 10826, 323, 1846, 879, 525, 13340, 10826, 13, 37154, 11, 432, 5646, 5019, 5382, 304, 279, 3082, 13, 13293, 3166, 311, 2908, 374, 429, 9720, 646, 614, 5089, 1343, 2361, 973, 11, 773, 279, 7042, 1372, 2578, 537, 387, 279, 1852, 438, 279, 3283, 594, 18967, 7042, 382, 40, 19091, 5290, 14696, 429, 279, 7042, 315, 12095, 572, 2163, 220, 16, 17, 3526, 264, 2421, 1635, 1182, 11, 714, 358, 1744, 429, 1372, 702, 14700, 13, 10696, 2163, 220, 16, 17, 13, 20, 3526, 1431, 30, 1988, 358, 2776, 537, 2704, 13, 358, 1265, 1430, 311, 50087, 279, 1482, 13142, 476, 1744, 911, 3213, 3669, 421, 894, 614, 1012, 4652, 382, 92014, 11, 8365, 358, 646, 1744, 911, 1008, 9720, 304, 9625, 323, 862, 21910, 311, 633, 264, 5530, 13, 1752, 3110, 11, 55201, 374, 264, 3598, 3283, 11, 714, 1181, 7042, 374, 9155, 13, 10696, 2163, 220, 16, 13, 17, 3526, 30, 1597, 1221, 1052, 594, 87693, 11, 892, 2578, 387, 264, 2421, 7739, 16183, 13, 2055, 11, 12095, 1660, 279, 7772, 3643, 5530, 382, 14190, 11, 714, 279, 57406, 3082, 315, 12095, 374, 3545, 21870, 438, 3432, 264, 1753, 8131, 7042, 13, 358, 1744, 432, 594, 2163, 220, 17, 20, 3526, 1251, 13, 2055, 11, 421, 279, 1196, 374, 10161, 369, 1101, 279, 3283, 594, 7042, 11, 432, 594, 4658, 2163, 220, 16, 17, 3526, 11, 714, 421, 807, 2299, 22023, 311, 279, 57406, 3082, 11, 432, 594, 220, 17, 20, 3526, 382, 40, 1265, 10146, 421, 279, 2856, 3405, 572, 911, 279, 3283, 476, 279, 57406, 3082, 13, 576, 3405, 2727, 11, 330, 1782, 6722, 315, 9625, 1, 892, 374, 12095, 11, 714, 358, 2776, 537, 2704, 421, 807, 8791, 279, 3283, 6169, 476, 279, 26829, 57406, 3082, 13, 1416, 358, 728, 448, 1101, 279, 3283, 11, 432, 594, 2163, 220, 16, 17, 3526, 11, 714, 358, 1265, 4658, 6286, 2176, 311, 3421, 678, 23092, 382, 13394, 11, 358, 1265, 2908, 429, 7042, 12396, 525, 44868, 323, 646, 13289, 11649, 389, 279, 2530, 13, 1752, 3110, 11, 279, 4337, 8547, 2578, 614, 10078, 2155, 5109, 7707, 311, 1008, 11104, 1075, 80776, 476, 1496, 2205, 11276, 382, 641, 12126, 11, 358, 2776, 5020, 16506, 429, 12095, 702, 264, 7042, 315, 916, 220, 16, 17, 3526, 11, 323, 10767, 2163, 220, 16, 17, 13, 20, 3526, 311, 220, 16, 18, 3526, 13, 576, 57406, 3082, 374, 11941, 8131, 11, 2163, 220, 17, 20, 3526, 11, 892, 5646, 264, 1753, 26829, 45039, 2088, 7797, 1101, 279, 3283, 13388, 382, 40, 1265, 4658, 3042, 2176, 5109, 304, 279, 2033, 773, 279, 1196, 8788, 429, 279, 6722, 315, 9625, 646, 387, 389, 2987, 835, 11649, 389, 1128, 807, 2299, 3330, 369, 13, 7281, 11, 44291, 429, 279, 4734, 7071, 646, 13289, 374, 2989, 311, 5648, 7086, 279, 4969, 1995, 382, 14037, 17879, 374, 3425, 279, 1196, 374, 3330, 369, 279, 1482, 1042, 594, 821, 13, 39529, 5109, 646, 38288, 6292, 29071, 11, 773, 421, 807, 1184, 279, 1429, 3213, 7071, 11, 807, 2578, 614, 311, 1779, 458, 705, 4686, 18413, 2530, 13, 4354, 11, 358, 1744, 279, 7071, 358, 614, 374, 3213, 3322, 11, 2661, 429, 358, 8990, 944, 6617, 894, 3598, 4344, 304, 3213, 1635, 382, 641, 894, 1142, 11, 8241, 2176, 279, 3283, 323, 57406, 3082, 12396, 4977, 1075, 264, 1661, 5486, 13, 358, 3278, 5944, 432, 1119, 1378, 14158, 311, 1281, 432, 2797, 11, 323, 7196, 6286, 429, 279, 15662, 3082, 7042, 374, 13187, 220, 16, 17, 3526, 11, 1393, 279, 57406, 3082, 374, 2163, 220, 17, 20, 3526, 382, 4416, 11, 311, 2629, 705, 11, 279, 6722, 11, 12095, 11, 702, 264, 7042, 315, 911, 220, 16, 17, 3526, 304, 279, 3283, 6169, 11, 323, 2163, 220, 17, 20, 3526, 304, 1181, 57406, 3082, 13, 1096, 1616, 11, 279, 1196, 646, 633, 264, 2664, 8660, 315, 279, 7042, 5109, 11649, 389, 862, 3151, 2734, 624, 151649, 271, 785, 6722, 315, 9625, 11, 12095, 11, 702, 264, 7042, 315, 13187, 220, 16, 17, 3526, 304, 279, 3283, 6169, 11, 323, 911, 220, 17, 20, 3526, 304, 1181, 57406, 3082, 13, 4220, 12396, 3410, 264, 2797, 29196, 1948, 279, 3283, 323, 279, 26829, 57406, 5537, 13, 151643], 'meta_info': {'id': '1896e129853042cf90983e07a9e7d01c', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 865, 'completion_tokens': 914, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 8.707355097867548, 'response_sent_to_client_ts': 1776726821.7756383}}</strong>



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
    [2026-04-20 23:13:51] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.07s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.17s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.16s/it]


    2026-04-20 23:13:59,800 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-20 23:13:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:51,  3.01s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:51,  3.01s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:22,  1.47s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:22,  1.47s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:48,  1.13it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:48,  1.13it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:32,  1.65it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:32,  1.65it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:23,  2.26it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:23,  2.26it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:17,  2.90it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:17,  2.90it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:13,  3.65it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.43it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.43it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.32it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.32it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.32it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  7.00it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  7.00it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  7.00it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.52it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.52it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.52it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04, 10.10it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04, 10.10it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04, 10.10it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 12.02it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 12.02it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 12.02it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 12.02it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.43it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.43it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.43it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.43it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.43it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.97it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.97it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.97it/s]

    Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.97it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.97it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.97it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:01, 20.97it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:00, 29.87it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 39.87it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 39.87it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 39.87it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 39.87it/s]

    Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 39.87it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 39.87it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 39.87it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 39.87it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 39.87it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 49.02it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 49.02it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 49.02it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 49.02it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 49.02it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 49.02it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 49.02it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 49.02it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 54.66it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 54.66it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 54.66it/s]

    Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 54.66it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 54.66it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 54.66it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 54.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=122.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=122.24 GB):   2%|▏         | 1/58 [00:00<00:16,  3.51it/s]Capturing num tokens (num_tokens=7680 avail_mem=122.21 GB):   2%|▏         | 1/58 [00:00<00:16,  3.51it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=122.21 GB):   3%|▎         | 2/58 [00:00<00:15,  3.62it/s]Capturing num tokens (num_tokens=7168 avail_mem=122.21 GB):   3%|▎         | 2/58 [00:00<00:15,  3.62it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=122.21 GB):   5%|▌         | 3/58 [00:00<00:14,  3.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=122.22 GB):   5%|▌         | 3/58 [00:00<00:14,  3.84it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=122.22 GB):   7%|▋         | 4/58 [00:01<00:13,  4.12it/s]Capturing num tokens (num_tokens=6144 avail_mem=122.22 GB):   7%|▋         | 4/58 [00:01<00:13,  4.12it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=122.22 GB):   9%|▊         | 5/58 [00:01<00:12,  4.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=122.22 GB):   9%|▊         | 5/58 [00:01<00:12,  4.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=122.22 GB):  10%|█         | 6/58 [00:01<00:10,  4.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=122.22 GB):  10%|█         | 6/58 [00:01<00:10,  4.74it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=122.22 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=122.23 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=122.23 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=122.23 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.60it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=122.23 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=122.24 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=122.24 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=122.24 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.57it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=122.24 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=122.24 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=122.24 GB):  21%|██        | 12/58 [00:02<00:05,  7.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=122.23 GB):  21%|██        | 12/58 [00:02<00:05,  7.70it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=122.24 GB):  21%|██        | 12/58 [00:02<00:05,  7.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=122.24 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.83it/s]Capturing num tokens (num_tokens=2560 avail_mem=122.24 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=122.24 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.83it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=122.24 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=122.23 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=122.24 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=122.24 GB):  31%|███       | 18/58 [00:02<00:03, 11.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=122.23 GB):  31%|███       | 18/58 [00:02<00:03, 11.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=122.23 GB):  31%|███       | 18/58 [00:02<00:03, 11.61it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=122.23 GB):  31%|███       | 18/58 [00:02<00:03, 11.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=122.23 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.57it/s]Capturing num tokens (num_tokens=960 avail_mem=122.23 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.57it/s] Capturing num tokens (num_tokens=896 avail_mem=122.22 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.57it/s]Capturing num tokens (num_tokens=832 avail_mem=122.22 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.57it/s]Capturing num tokens (num_tokens=832 avail_mem=122.22 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.50it/s]Capturing num tokens (num_tokens=768 avail_mem=122.22 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.50it/s]Capturing num tokens (num_tokens=704 avail_mem=122.21 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.50it/s]

    Capturing num tokens (num_tokens=640 avail_mem=122.21 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.50it/s]Capturing num tokens (num_tokens=640 avail_mem=122.21 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.42it/s]Capturing num tokens (num_tokens=576 avail_mem=122.21 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.42it/s]Capturing num tokens (num_tokens=512 avail_mem=122.20 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.42it/s]Capturing num tokens (num_tokens=480 avail_mem=122.20 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.42it/s]Capturing num tokens (num_tokens=448 avail_mem=122.20 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.42it/s]Capturing num tokens (num_tokens=448 avail_mem=122.20 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=416 avail_mem=122.19 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=384 avail_mem=122.19 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]

    Capturing num tokens (num_tokens=352 avail_mem=122.18 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=320 avail_mem=122.18 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=320 avail_mem=122.18 GB):  60%|██████    | 35/58 [00:03<00:00, 27.22it/s]Capturing num tokens (num_tokens=288 avail_mem=122.18 GB):  60%|██████    | 35/58 [00:03<00:00, 27.22it/s]Capturing num tokens (num_tokens=256 avail_mem=122.17 GB):  60%|██████    | 35/58 [00:03<00:00, 27.22it/s]Capturing num tokens (num_tokens=240 avail_mem=122.17 GB):  60%|██████    | 35/58 [00:03<00:00, 27.22it/s]Capturing num tokens (num_tokens=224 avail_mem=122.17 GB):  60%|██████    | 35/58 [00:03<00:00, 27.22it/s]Capturing num tokens (num_tokens=224 avail_mem=122.17 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.86it/s]Capturing num tokens (num_tokens=208 avail_mem=122.16 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.86it/s]Capturing num tokens (num_tokens=192 avail_mem=122.16 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.86it/s]

    Capturing num tokens (num_tokens=176 avail_mem=122.15 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.86it/s]Capturing num tokens (num_tokens=160 avail_mem=122.15 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.86it/s]Capturing num tokens (num_tokens=160 avail_mem=122.15 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.87it/s]Capturing num tokens (num_tokens=144 avail_mem=122.15 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.87it/s]Capturing num tokens (num_tokens=128 avail_mem=122.16 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.87it/s]Capturing num tokens (num_tokens=112 avail_mem=122.15 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.87it/s]Capturing num tokens (num_tokens=96 avail_mem=122.15 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.87it/s] Capturing num tokens (num_tokens=96 avail_mem=122.15 GB):  81%|████████  | 47/58 [00:03<00:00, 33.16it/s]Capturing num tokens (num_tokens=80 avail_mem=122.14 GB):  81%|████████  | 47/58 [00:03<00:00, 33.16it/s]Capturing num tokens (num_tokens=64 avail_mem=122.14 GB):  81%|████████  | 47/58 [00:03<00:00, 33.16it/s]

    Capturing num tokens (num_tokens=48 avail_mem=122.14 GB):  81%|████████  | 47/58 [00:03<00:00, 33.16it/s]Capturing num tokens (num_tokens=32 avail_mem=122.13 GB):  81%|████████  | 47/58 [00:03<00:00, 33.16it/s]Capturing num tokens (num_tokens=32 avail_mem=122.13 GB):  88%|████████▊ | 51/58 [00:03<00:00, 34.22it/s]Capturing num tokens (num_tokens=28 avail_mem=122.13 GB):  88%|████████▊ | 51/58 [00:03<00:00, 34.22it/s]Capturing num tokens (num_tokens=24 avail_mem=122.13 GB):  88%|████████▊ | 51/58 [00:03<00:00, 34.22it/s]Capturing num tokens (num_tokens=20 avail_mem=122.12 GB):  88%|████████▊ | 51/58 [00:03<00:00, 34.22it/s]Capturing num tokens (num_tokens=16 avail_mem=122.12 GB):  88%|████████▊ | 51/58 [00:03<00:00, 34.22it/s]Capturing num tokens (num_tokens=16 avail_mem=122.12 GB):  95%|█████████▍| 55/58 [00:03<00:00, 34.97it/s]Capturing num tokens (num_tokens=12 avail_mem=122.12 GB):  95%|█████████▍| 55/58 [00:03<00:00, 34.97it/s]Capturing num tokens (num_tokens=8 avail_mem=122.11 GB):  95%|█████████▍| 55/58 [00:03<00:00, 34.97it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=122.11 GB):  95%|█████████▍| 55/58 [00:03<00:00, 34.97it/s]Capturing num tokens (num_tokens=4 avail_mem=122.11 GB): 100%|██████████| 58/58 [00:03<00:00, 15.00it/s]


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
    Generated text: Paris is the capital of France
    ===============================
    Prompt: Give me the information of the capital of Germany.
    Generated text: Berlin is the capital of Germany
    ===============================
    Prompt: Give me the information of the capital of Italy.
    Generated text: Paris is the capital of France


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
    
    Generated text: Alright, the user is asking for the information and population of the capital of France in JSON format. First, I need to identify who the user is and what their intent is. It seems like they might be a student or someone working on a project where they need structured data.
    
    I should start by recalling that the capital of France is Paris. Knowing this is straightforward, but I need to ensure the information is accurate. The population figure needs to be up-to-date. From what I remember, the population of Paris is around 2 million, but I should double-check that to be sure.
    
    Next, I should structure this information into a JSON format. JSON requires keys and values, so I'll create an object with a key like "capital" and the value as Paris. Then, another key for the population, making sure to include the number in a readable format, perhaps with commas for clarity, like 2,000,000.
    
    I should also consider if the user might need additional details, such as the percentage of the total French population or the metropolitan area. However, the user only asked for the capital and population, so it's best to stick to that unless they ask for more.
    
    I need to present the information clearly and concisely, using proper JSON syntax to avoid any errors. Proper indentation and formatting make it easier to read, so I'll make sure to include that.
    
    Finally, I'll offer to provide more details if needed, showing that I'm open to assisting further. This way, the user feels supported and knows they can reach out again if their query expands.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "capital": "Paris",
      "population": 2000000  // Population of Paris (as of recent estimates)
    }
    ```
    
    Let me know if you need more details!



```python
llm.shutdown()
```
