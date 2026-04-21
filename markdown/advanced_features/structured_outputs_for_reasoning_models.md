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
    [2026-04-21 02:27:10] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-21 02:27:11] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-21 02:27:12] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-21 02:27:14] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-21 02:27:20] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-21 02:27:21] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False
    [2026-04-21 02:27:21] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.48s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.45s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.45s/it]


    2026-04-21 02:27:29,903 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 02:27:29] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:08,  3.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:08,  3.31s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:36,  1.73s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:36,  1.73s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:57,  1.04s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:57,  1.04s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:29,  1.80it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:29,  1.80it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:24,  2.11it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:24,  2.11it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:20,  2.48it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:20,  2.48it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:18,  2.75it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:18,  2.75it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:15,  3.11it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:15,  3.11it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:11,  4.16it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:11,  4.16it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:11,  4.16it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:07,  6.03it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:07,  6.03it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:07,  6.03it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:05,  7.37it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:05,  7.37it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:05,  7.62it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:05,  7.62it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:05,  7.77it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:05,  7.77it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:06<00:04,  8.10it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:06<00:04,  8.10it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:06<00:04,  8.10it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:03,  9.78it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:03,  9.78it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:03,  9.78it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:07<00:03, 11.01it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:07<00:03, 11.01it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:07<00:03, 11.01it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:02, 12.27it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:02, 12.27it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:02, 12.27it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:02, 12.27it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:07<00:01, 16.02it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:07<00:01, 16.02it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:07<00:01, 16.02it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:07<00:01, 16.02it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:01, 17.79it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:01, 17.79it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:01, 17.79it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:01, 17.79it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:07<00:01, 19.71it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:07<00:01, 19.71it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:07<00:01, 19.71it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:07<00:01, 19.71it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:07<00:01, 19.71it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:07<00:00, 22.62it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:07<00:00, 22.62it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:07<00:00, 22.62it/s]

    Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:07<00:00, 22.62it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:07<00:00, 22.62it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:07<00:00, 25.39it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:07<00:00, 25.39it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:07<00:00, 25.39it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:07<00:00, 25.39it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:07<00:00, 25.39it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:07<00:00, 27.83it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:07<00:00, 27.83it/s]

    Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:07<00:00, 27.83it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:07<00:00, 27.83it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:08<00:00, 27.83it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 29.84it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 29.84it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 29.84it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 29.84it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 29.84it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:08<00:00, 30.39it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:08<00:00, 30.39it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:08<00:00, 30.39it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:08<00:00, 30.39it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:08<00:00, 30.39it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:08<00:00, 30.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.04it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.70 GB):   2%|▏         | 1/58 [00:00<00:26,  2.13it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.03 GB):   2%|▏         | 1/58 [00:00<00:26,  2.13it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.03 GB):   3%|▎         | 2/58 [00:00<00:24,  2.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=44.03 GB):   3%|▎         | 2/58 [00:00<00:24,  2.31it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=44.03 GB):   5%|▌         | 3/58 [00:01<00:22,  2.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.03 GB):   5%|▌         | 3/58 [00:01<00:22,  2.45it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=44.03 GB):   7%|▋         | 4/58 [00:01<00:20,  2.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.91 GB):   7%|▋         | 4/58 [00:01<00:20,  2.69it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.91 GB):   9%|▊         | 5/58 [00:01<00:18,  2.93it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.02 GB):   9%|▊         | 5/58 [00:01<00:18,  2.93it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=44.02 GB):  10%|█         | 6/58 [00:02<00:16,  3.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.01 GB):  10%|█         | 6/58 [00:02<00:16,  3.21it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=44.01 GB):  12%|█▏        | 7/58 [00:02<00:14,  3.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.00 GB):  12%|█▏        | 7/58 [00:02<00:14,  3.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.00 GB):  14%|█▍        | 8/58 [00:02<00:12,  3.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.00 GB):  14%|█▍        | 8/58 [00:02<00:12,  3.94it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=44.00 GB):  16%|█▌        | 9/58 [00:02<00:11,  4.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.00 GB):  16%|█▌        | 9/58 [00:02<00:11,  4.34it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=44.00 GB):  17%|█▋        | 10/58 [00:02<00:11,  4.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.99 GB):  17%|█▋        | 10/58 [00:02<00:11,  4.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.99 GB):  19%|█▉        | 11/58 [00:03<00:09,  4.83it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.98 GB):  19%|█▉        | 11/58 [00:03<00:09,  4.83it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=43.98 GB):  21%|██        | 12/58 [00:03<00:08,  5.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.97 GB):  21%|██        | 12/58 [00:03<00:08,  5.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.97 GB):  22%|██▏       | 13/58 [00:03<00:07,  5.76it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.98 GB):  22%|██▏       | 13/58 [00:03<00:07,  5.76it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.98 GB):  24%|██▍       | 14/58 [00:03<00:06,  6.36it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.97 GB):  24%|██▍       | 14/58 [00:03<00:06,  6.36it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.97 GB):  26%|██▌       | 15/58 [00:03<00:06,  7.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.96 GB):  26%|██▌       | 15/58 [00:03<00:06,  7.13it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=43.96 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.96 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.93 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.93 GB):  31%|███       | 18/58 [00:03<00:04,  8.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.94 GB):  31%|███       | 18/58 [00:03<00:04,  8.74it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=43.92 GB):  31%|███       | 18/58 [00:03<00:04,  8.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.92 GB):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.93 GB):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s]Capturing num tokens (num_tokens=960 avail_mem=43.92 GB):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s] Capturing num tokens (num_tokens=960 avail_mem=43.92 GB):  38%|███▊      | 22/58 [00:04<00:02, 12.15it/s]Capturing num tokens (num_tokens=896 avail_mem=43.92 GB):  38%|███▊      | 22/58 [00:04<00:02, 12.15it/s]

    Capturing num tokens (num_tokens=832 avail_mem=43.91 GB):  38%|███▊      | 22/58 [00:04<00:02, 12.15it/s]Capturing num tokens (num_tokens=768 avail_mem=43.90 GB):  38%|███▊      | 22/58 [00:04<00:02, 12.15it/s]Capturing num tokens (num_tokens=768 avail_mem=43.90 GB):  43%|████▎     | 25/58 [00:04<00:02, 15.15it/s]Capturing num tokens (num_tokens=704 avail_mem=43.90 GB):  43%|████▎     | 25/58 [00:04<00:02, 15.15it/s]Capturing num tokens (num_tokens=640 avail_mem=43.90 GB):  43%|████▎     | 25/58 [00:04<00:02, 15.15it/s]Capturing num tokens (num_tokens=576 avail_mem=43.89 GB):  43%|████▎     | 25/58 [00:04<00:02, 15.15it/s]Capturing num tokens (num_tokens=576 avail_mem=43.89 GB):  48%|████▊     | 28/58 [00:04<00:01, 18.21it/s]Capturing num tokens (num_tokens=512 avail_mem=43.89 GB):  48%|████▊     | 28/58 [00:04<00:01, 18.21it/s]

    Capturing num tokens (num_tokens=480 avail_mem=43.89 GB):  48%|████▊     | 28/58 [00:04<00:01, 18.21it/s]Capturing num tokens (num_tokens=448 avail_mem=43.88 GB):  48%|████▊     | 28/58 [00:04<00:01, 18.21it/s]Capturing num tokens (num_tokens=416 avail_mem=43.88 GB):  48%|████▊     | 28/58 [00:04<00:01, 18.21it/s]Capturing num tokens (num_tokens=416 avail_mem=43.88 GB):  55%|█████▌    | 32/58 [00:04<00:01, 22.11it/s]Capturing num tokens (num_tokens=384 avail_mem=43.88 GB):  55%|█████▌    | 32/58 [00:04<00:01, 22.11it/s]Capturing num tokens (num_tokens=352 avail_mem=43.87 GB):  55%|█████▌    | 32/58 [00:04<00:01, 22.11it/s]Capturing num tokens (num_tokens=320 avail_mem=43.87 GB):  55%|█████▌    | 32/58 [00:04<00:01, 22.11it/s]Capturing num tokens (num_tokens=288 avail_mem=43.86 GB):  55%|█████▌    | 32/58 [00:04<00:01, 22.11it/s]

    Capturing num tokens (num_tokens=288 avail_mem=43.86 GB):  62%|██████▏   | 36/58 [00:04<00:00, 25.50it/s]Capturing num tokens (num_tokens=256 avail_mem=43.86 GB):  62%|██████▏   | 36/58 [00:04<00:00, 25.50it/s]Capturing num tokens (num_tokens=240 avail_mem=43.85 GB):  62%|██████▏   | 36/58 [00:04<00:00, 25.50it/s]Capturing num tokens (num_tokens=224 avail_mem=43.85 GB):  62%|██████▏   | 36/58 [00:04<00:00, 25.50it/s]Capturing num tokens (num_tokens=208 avail_mem=43.85 GB):  62%|██████▏   | 36/58 [00:04<00:00, 25.50it/s]Capturing num tokens (num_tokens=208 avail_mem=43.85 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.59it/s]Capturing num tokens (num_tokens=192 avail_mem=43.84 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.59it/s]Capturing num tokens (num_tokens=176 avail_mem=43.84 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.59it/s]Capturing num tokens (num_tokens=160 avail_mem=43.84 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.59it/s]Capturing num tokens (num_tokens=144 avail_mem=43.83 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.59it/s]

    Capturing num tokens (num_tokens=144 avail_mem=43.83 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.93it/s]Capturing num tokens (num_tokens=128 avail_mem=43.84 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.93it/s]Capturing num tokens (num_tokens=112 avail_mem=43.84 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.93it/s]Capturing num tokens (num_tokens=96 avail_mem=43.83 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.93it/s] Capturing num tokens (num_tokens=80 avail_mem=43.83 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.93it/s]Capturing num tokens (num_tokens=80 avail_mem=43.83 GB):  83%|████████▎ | 48/58 [00:04<00:00, 32.77it/s]Capturing num tokens (num_tokens=64 avail_mem=43.83 GB):  83%|████████▎ | 48/58 [00:04<00:00, 32.77it/s]Capturing num tokens (num_tokens=48 avail_mem=43.83 GB):  83%|████████▎ | 48/58 [00:04<00:00, 32.77it/s]Capturing num tokens (num_tokens=32 avail_mem=43.82 GB):  83%|████████▎ | 48/58 [00:05<00:00, 32.77it/s]Capturing num tokens (num_tokens=28 avail_mem=43.82 GB):  83%|████████▎ | 48/58 [00:05<00:00, 32.77it/s]

    Capturing num tokens (num_tokens=28 avail_mem=43.82 GB):  90%|████████▉ | 52/58 [00:05<00:00, 34.46it/s]Capturing num tokens (num_tokens=24 avail_mem=43.82 GB):  90%|████████▉ | 52/58 [00:05<00:00, 34.46it/s]Capturing num tokens (num_tokens=20 avail_mem=43.81 GB):  90%|████████▉ | 52/58 [00:05<00:00, 34.46it/s]Capturing num tokens (num_tokens=16 avail_mem=43.81 GB):  90%|████████▉ | 52/58 [00:05<00:00, 34.46it/s]Capturing num tokens (num_tokens=12 avail_mem=43.80 GB):  90%|████████▉ | 52/58 [00:05<00:00, 34.46it/s]Capturing num tokens (num_tokens=12 avail_mem=43.80 GB):  97%|█████████▋| 56/58 [00:05<00:00, 35.37it/s]Capturing num tokens (num_tokens=8 avail_mem=43.80 GB):  97%|█████████▋| 56/58 [00:05<00:00, 35.37it/s] Capturing num tokens (num_tokens=4 avail_mem=43.80 GB):  97%|█████████▋| 56/58 [00:05<00:00, 35.37it/s]Capturing num tokens (num_tokens=4 avail_mem=43.80 GB): 100%|██████████| 58/58 [00:05<00:00, 11.10it/s]


    [2026-04-21 02:27:45] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants the current date and time, along with the weather. Let me break this down step by step.<br><br>First, I need to figure out which functions to use. From the instructions, there are two relevant functions: 'get_current_weather' and 'get_current_date'. <br><br>Starting with the date and time: the function requires a timezone parameter. Since the user is in New York, I should use 'America/New_York' as the timezone. I'll structure the function call as <function=date>{{"timezone": "America/New_York"}}</function>.<br><br>Next, for the weather, I need the city and state. The user mentioned being in New York, so the city is 'New York' and the state is 'NY'. The unit isn't specified, so I'll default to Celsius. The function call will be <function=weather>{{"city": "New York", "state": "NY", "unit": "celsius"}}</function>.<br><br>I should make sure to call each function separately, as per the instructions, and present both results clearly. Also, I'll add the sources where I got the functions from, which is the provided JSON data.<br><br><br>content: <function=get_current_weather>{  <br>  "city": "New York",  <br>  "state": "NY",  <br>  "unit": "celsius"  <br>}</function>  <br><function=get_current_date>{  <br>  "timezone": "America/New_York"  <br>}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not exactly sure of the exact number. I think it\'s around 2 million, but I should double-check that.\n\nWait, maybe I should look up the latest statistics to be accurate. I recall that in recent years, Paris has seen a steady increase due to urban development and immigration. Let me see... I think the population is approximately 2,170,000 as of 2023. That seems about right.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the city name, its population, and maybe the country it\'s in. So, the keys could be "city", "population", and "country".\n\nPutting it all together, the JSON should look something like this: {"city": "Paris", "population": 2170000, "country": "France"}. I should make sure the syntax is correct, with proper commas and quotation marks.\n\nI should also consider if the user might need more details, like the metropolitan area population or other statistics, but since they specifically asked for the capital, I\'ll stick to that. Maybe mention that the data is up to date as of 2023 to provide context.\n\nAlright, I think that\'s all. I\'ll present the JSON and offer further assistance in case they need more information.\n</think>{\n  "name": "Paris",\n  "population": 2170000\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 382, 14190, 11, 7196, 358, 1265, 1401, 705, 279, 5535, 13142, 311, 387, 13382, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 24020, 5263, 4152, 311, 15662, 4401, 323, 15093, 13, 6771, 752, 1490, 1112, 358, 1744, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 3283, 829, 11, 1181, 7042, 11, 323, 7196, 279, 3146, 432, 594, 304, 13, 2055, 11, 279, 6894, 1410, 387, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 11436, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 5212, 8926, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 11, 330, 11141, 788, 330, 49000, 1, 7810, 358, 1265, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 57406, 3082, 7042, 476, 1008, 13142, 11, 714, 2474, 807, 11689, 4588, 369, 279, 6722, 11, 358, 3278, 9214, 311, 429, 13, 10696, 6286, 429, 279, 821, 374, 705, 311, 2400, 438, 315, 220, 17, 15, 17, 18, 311, 3410, 2266, 382, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3042, 279, 4718, 323, 3010, 4623, 12994, 304, 1142, 807, 1184, 803, 1995, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 198, 92, 151643], 'meta_info': {'id': '41e7bdb0ef9e4a8d829cb349e913777e', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 369, 'completion_tokens': 392, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.4023115718737245, 'response_sent_to_client_ts': 1776738492.0385277}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see... Yes, according to the latest data, the population is approximately 2,174,300 as of 2023. That seems accurate.\n\nNext, I need to structure this information into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and "country". The city is Paris, the population is the number I found, and the country is France.\n\nI should make sure the JSON syntax is correct. Each key should be in quotes, and the values as well. Also, the entire structure should be enclosed in curly braces. I\'ll format it properly to ensure there are no syntax errors.\n\nPutting it all together, the JSON object will have the city, population, and country. I\'ll present this to the user, making sure it\'s clear and easy to understand. I don\'t think the user needs anything else, but if they have more questions, I can provide additional information.\n\nI should also consider if the user might need the population figure in a different format, like a string or an integer, but since they asked for JSON, I\'ll stick with the number as an integer. \n\nAlright, I think that\'s all. I\'ll format the JSON correctly and present it to the user.\n</think>{\n\n"name": "Paris",\n"population": 2174300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 1112, 7414, 11, 4092, 311, 279, 5535, 821, 11, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 13382, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 3263, 576, 3283, 374, 12095, 11, 279, 7042, 374, 279, 1372, 358, 1730, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 8886, 1376, 1265, 387, 304, 17194, 11, 323, 279, 2750, 438, 1632, 13, 7281, 11, 279, 4453, 5944, 1265, 387, 43810, 304, 68103, 59191, 13, 358, 3278, 3561, 432, 10277, 311, 5978, 1052, 525, 902, 19482, 5975, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 686, 614, 279, 3283, 11, 7042, 11, 323, 3146, 13, 358, 3278, 3042, 419, 311, 279, 1196, 11, 3259, 2704, 432, 594, 2797, 323, 4135, 311, 3535, 13, 358, 1513, 944, 1744, 279, 1196, 3880, 4113, 770, 11, 714, 421, 807, 614, 803, 4755, 11, 358, 646, 3410, 5107, 1995, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 279, 7042, 7071, 304, 264, 2155, 3561, 11, 1075, 264, 914, 476, 458, 7546, 11, 714, 2474, 807, 4588, 369, 4718, 11, 358, 3278, 9214, 448, 279, 1372, 438, 458, 7546, 13, 4710, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3561, 279, 4718, 12440, 323, 3042, 432, 311, 279, 1196, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': 'fd3fa9ae63234faa842a9925c4369b69', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 383, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 16.45556989731267, 'response_sent_to_client_ts': 1776738508.5074189}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'f6edcd235fed48a6a7bf63f7d17b968c', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14403194095939398, 'response_sent_to_client_ts': 1776738508.7009084}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'c16bf6c695f7466f8abe5de4bedb703f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14395741233602166, 'response_sent_to_client_ts': 1776738508.7009227}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '8f5d31a958f043d58a5f7a8fa9b35484', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14390002191066742, 'response_sent_to_client_ts': 1776738508.7009275}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '2822d37001c5409b9961d3b487d1812e', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 20.054542722180486, 'response_sent_to_client_ts': 1776738528.7641778}}


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


<strong style='color: #00008B;'>{'text': 'Alright, the user asked for the information and population of the capital of France in JSON format. Okay, so the capital is definitely Paris. I should probably structure the response with key-value pairs. Let me think about what details they want. They mentioned population, so I need the approximate number. I know the recent estimates are around 2.1-2.2 million. It\'s good to mention that it\'s fluctuating between different sources.\n\nI should also include some key facts about Paris. Maybe its location, nickname, one thing people love there, and the universal language since it\'s a Global City. That makes the info comprehensive and interesting. \n\nI wonder, is the user just looking for a straightforward data dump, or do they need more context? Since they specified JSON, maybe they plan to use it for data processing or integration elsewhere. So being clear and concise with the JSON structure is important. \n\nI should make sure the JSON is properly formatted without any syntax errors. Also, since population can change, it\'s good to provide a note indicating that the figure is approximate. Yeah, that\'s all covered. Let me put it all together neatly.\n</think>\n\nSure! Here\'s the information and population of the capital of France (Paris) in JSON format:\n\n```json\n{\n  "capital": "Paris",\n  "population": {\n    "approximate": "2.1 to 2.2 million",\n    "note": "Population figures are approximate and may fluctuate slightly depending on the source."\n  },\n  "key_facts": {\n    "location": "The capital of France, located in northern France, bordering the Seine River.",\n    "nickname": "La Ville Constraints",\n    "people_love": "Paris is often praised for its romantic charm, cultural richness, and historical significance.",\n    "universalLanguage": "English is widely spoken in Paris, reflecting its status as a global city."\n  }\n}\n```', 'output_ids': [71486, 11, 279, 1196, 4588, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 35439, 11, 773, 279, 6722, 374, 8491, 12095, 13, 358, 1265, 4658, 5944, 279, 2033, 448, 1376, 19083, 13530, 13, 6771, 752, 1744, 911, 1128, 3565, 807, 1366, 13, 2379, 9733, 7042, 11, 773, 358, 1184, 279, 44868, 1372, 13, 358, 1414, 279, 3213, 17530, 525, 2163, 220, 17, 13, 16, 12, 17, 13, 17, 3526, 13, 1084, 594, 1661, 311, 6286, 429, 432, 594, 38288, 53543, 1948, 2155, 8173, 382, 40, 1265, 1083, 2924, 1045, 1376, 13064, 911, 12095, 13, 10696, 1181, 3728, 11, 29399, 11, 825, 3166, 1251, 2948, 1052, 11, 323, 279, 20178, 4128, 2474, 432, 594, 264, 7962, 4311, 13, 2938, 3643, 279, 3546, 15817, 323, 7040, 13, 4710, 40, 5775, 11, 374, 279, 1196, 1101, 3330, 369, 264, 30339, 821, 10276, 11, 476, 653, 807, 1184, 803, 2266, 30, 8704, 807, 5189, 4718, 11, 7196, 807, 3119, 311, 990, 432, 369, 821, 8692, 476, 17590, 17920, 13, 2055, 1660, 2797, 323, 63594, 448, 279, 4718, 5944, 374, 2989, 13, 4710, 40, 1265, 1281, 2704, 279, 4718, 374, 10277, 23126, 2041, 894, 19482, 5975, 13, 7281, 11, 2474, 7042, 646, 2297, 11, 432, 594, 1661, 311, 3410, 264, 5185, 18860, 429, 279, 7071, 374, 44868, 13, 21607, 11, 429, 594, 678, 9761, 13, 6771, 752, 2182, 432, 678, 3786, 62166, 624, 151649, 271, 39814, 0, 5692, 594, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 320, 59604, 8, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 341, 262, 330, 48053, 3426, 788, 330, 17, 13, 16, 311, 220, 17, 13, 17, 3526, 756, 262, 330, 9974, 788, 330, 53371, 12396, 525, 44868, 323, 1231, 38288, 6292, 10078, 11649, 389, 279, 2530, 10040, 220, 1153, 220, 330, 792, 761, 11359, 788, 341, 262, 330, 2527, 788, 330, 785, 6722, 315, 9625, 11, 7407, 304, 18172, 9625, 11, 3886, 287, 279, 1345, 482, 10948, 10346, 262, 330, 39413, 788, 330, 8747, 89130, 87473, 756, 262, 330, 16069, 5560, 586, 788, 330, 59604, 374, 3545, 36375, 369, 1181, 23467, 31253, 11, 12752, 88930, 11, 323, 13656, 25361, 10346, 262, 330, 95300, 13806, 788, 330, 22574, 374, 13570, 21355, 304, 12095, 11, 41752, 1181, 2639, 438, 264, 3644, 3283, 10040, 220, 456, 532, 73594, 151643], 'meta_info': {'id': '14bce9b7be384398a4443d529f744492', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 235, 'completion_tokens': 396, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.426836788188666, 'response_sent_to_client_ts': 1776738532.199974}}</strong>



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
    [2026-04-21 02:29:01] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.77s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.54s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.57s/it]


    2026-04-21 02:29:11,666 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 02:29:11] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:52,  3.03s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:52,  3.03s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:23,  1.50s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:23,  1.50s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:25,  2.10it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:25,  2.10it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:22,  2.29it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:22,  2.29it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:20,  2.55it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:20,  2.55it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.83it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.83it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:15,  3.12it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:15,  3.12it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:13,  3.50it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:12,  3.79it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:12,  3.79it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:10,  4.18it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:10,  4.18it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:09,  4.54it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:09,  4.54it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:08,  4.94it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:08,  4.94it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:08,  5.36it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:08,  5.36it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:06,  6.00it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:06,  6.00it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:06,  6.52it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:06,  6.52it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:06,  6.52it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:06<00:04,  7.83it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:06<00:04,  7.83it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:06<00:04,  7.83it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:06<00:04,  9.19it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:06<00:04,  9.19it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:06<00:04,  9.19it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:03, 10.77it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:03, 10.77it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:03, 10.77it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:07<00:02, 12.31it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:07<00:02, 12.31it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:07<00:02, 12.31it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:07<00:02, 13.79it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:07<00:02, 13.79it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:07<00:02, 13.79it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:07<00:02, 13.79it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:01, 16.18it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:01, 16.18it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:01, 16.18it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:01, 16.18it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:07<00:01, 18.08it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:07<00:01, 18.08it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:07<00:01, 18.08it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:07<00:01, 18.08it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:01, 20.02it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:01, 20.02it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:01, 20.02it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:01, 20.02it/s]

    Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:07<00:00, 22.49it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:07<00:00, 22.49it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:07<00:00, 22.49it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:07<00:00, 22.49it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:07<00:00, 22.49it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 24.80it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 24.80it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:07<00:00, 24.80it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:07<00:00, 24.80it/s]

    Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:08<00:00, 24.80it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:08<00:00, 26.92it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:08<00:00, 26.92it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:08<00:00, 26.92it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:08<00:00, 26.92it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:08<00:00, 27.30it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:08<00:00, 27.30it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:08<00:00, 27.30it/s]

    Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:08<00:00, 27.30it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:08<00:00, 27.30it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:08<00:00, 29.27it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:08<00:00, 29.27it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:08<00:00, 29.27it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:08<00:00, 29.27it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:08<00:00, 29.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=28.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=28.41 GB):   2%|▏         | 1/58 [00:00<00:25,  2.23it/s]Capturing num tokens (num_tokens=7680 avail_mem=23.25 GB):   2%|▏         | 1/58 [00:00<00:25,  2.23it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=23.25 GB):   3%|▎         | 2/58 [00:00<00:19,  2.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=23.12 GB):   3%|▎         | 2/58 [00:00<00:19,  2.86it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=23.12 GB):   5%|▌         | 3/58 [00:01<00:23,  2.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=23.09 GB):   5%|▌         | 3/58 [00:01<00:23,  2.36it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=23.09 GB):   7%|▋         | 4/58 [00:01<00:23,  2.34it/s]Capturing num tokens (num_tokens=6144 avail_mem=22.11 GB):   7%|▋         | 4/58 [00:01<00:23,  2.34it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=22.11 GB):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=22.18 GB):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=22.18 GB):  10%|█         | 6/58 [00:02<00:21,  2.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=22.24 GB):  10%|█         | 6/58 [00:02<00:21,  2.37it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=22.24 GB):  12%|█▏        | 7/58 [00:02<00:21,  2.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=22.32 GB):  12%|█▏        | 7/58 [00:02<00:21,  2.42it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=22.32 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=23.12 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.62it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=23.12 GB):  16%|█▌        | 9/58 [00:03<00:18,  2.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=23.12 GB):  16%|█▌        | 9/58 [00:03<00:18,  2.68it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=23.12 GB):  17%|█▋        | 10/58 [00:03<00:16,  2.89it/s]Capturing num tokens (num_tokens=3584 avail_mem=23.13 GB):  17%|█▋        | 10/58 [00:03<00:16,  2.89it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=23.13 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=23.13 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.15it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=23.13 GB):  21%|██        | 12/58 [00:04<00:13,  3.49it/s]Capturing num tokens (num_tokens=3072 avail_mem=22.60 GB):  21%|██        | 12/58 [00:04<00:13,  3.49it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=22.60 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=22.64 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=22.64 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=22.67 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.13it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=22.67 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=22.70 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=22.70 GB):  28%|██▊       | 16/58 [00:05<00:08,  4.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=22.73 GB):  28%|██▊       | 16/58 [00:05<00:08,  4.97it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=22.73 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=23.13 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=23.13 GB):  31%|███       | 18/58 [00:05<00:06,  5.86it/s]Capturing num tokens (num_tokens=1536 avail_mem=23.13 GB):  31%|███       | 18/58 [00:05<00:06,  5.86it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=23.13 GB):  33%|███▎      | 19/58 [00:05<00:06,  6.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=23.16 GB):  33%|███▎      | 19/58 [00:05<00:06,  6.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=22.83 GB):  33%|███▎      | 19/58 [00:05<00:06,  6.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=22.83 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s]Capturing num tokens (num_tokens=960 avail_mem=22.85 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=23.12 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s]Capturing num tokens (num_tokens=896 avail_mem=23.12 GB):  40%|███▉      | 23/58 [00:05<00:03,  9.16it/s]Capturing num tokens (num_tokens=832 avail_mem=23.11 GB):  40%|███▉      | 23/58 [00:05<00:03,  9.16it/s]Capturing num tokens (num_tokens=768 avail_mem=23.13 GB):  40%|███▉      | 23/58 [00:05<00:03,  9.16it/s]

    Capturing num tokens (num_tokens=768 avail_mem=23.13 GB):  43%|████▎     | 25/58 [00:05<00:03, 10.26it/s]Capturing num tokens (num_tokens=704 avail_mem=22.93 GB):  43%|████▎     | 25/58 [00:05<00:03, 10.26it/s]Capturing num tokens (num_tokens=640 avail_mem=22.92 GB):  43%|████▎     | 25/58 [00:06<00:03, 10.26it/s]Capturing num tokens (num_tokens=640 avail_mem=22.92 GB):  47%|████▋     | 27/58 [00:06<00:02, 11.82it/s]Capturing num tokens (num_tokens=576 avail_mem=22.93 GB):  47%|████▋     | 27/58 [00:06<00:02, 11.82it/s]Capturing num tokens (num_tokens=512 avail_mem=22.93 GB):  47%|████▋     | 27/58 [00:06<00:02, 11.82it/s]

    Capturing num tokens (num_tokens=512 avail_mem=22.93 GB):  50%|█████     | 29/58 [00:06<00:02, 12.69it/s]Capturing num tokens (num_tokens=480 avail_mem=23.06 GB):  50%|█████     | 29/58 [00:06<00:02, 12.69it/s]Capturing num tokens (num_tokens=448 avail_mem=22.93 GB):  50%|█████     | 29/58 [00:06<00:02, 12.69it/s]Capturing num tokens (num_tokens=448 avail_mem=22.93 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.03it/s]Capturing num tokens (num_tokens=416 avail_mem=22.93 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.03it/s]Capturing num tokens (num_tokens=384 avail_mem=22.93 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.03it/s]

    Capturing num tokens (num_tokens=384 avail_mem=22.93 GB):  57%|█████▋    | 33/58 [00:06<00:01, 13.55it/s]Capturing num tokens (num_tokens=352 avail_mem=22.93 GB):  57%|█████▋    | 33/58 [00:06<00:01, 13.55it/s]Capturing num tokens (num_tokens=320 avail_mem=23.00 GB):  57%|█████▋    | 33/58 [00:06<00:01, 13.55it/s]Capturing num tokens (num_tokens=320 avail_mem=23.00 GB):  60%|██████    | 35/58 [00:06<00:01, 15.04it/s]Capturing num tokens (num_tokens=288 avail_mem=23.00 GB):  60%|██████    | 35/58 [00:06<00:01, 15.04it/s]Capturing num tokens (num_tokens=256 avail_mem=22.99 GB):  60%|██████    | 35/58 [00:06<00:01, 15.04it/s]Capturing num tokens (num_tokens=240 avail_mem=22.98 GB):  60%|██████    | 35/58 [00:06<00:01, 15.04it/s]

    Capturing num tokens (num_tokens=240 avail_mem=22.98 GB):  66%|██████▌   | 38/58 [00:06<00:01, 17.38it/s]Capturing num tokens (num_tokens=224 avail_mem=22.98 GB):  66%|██████▌   | 38/58 [00:06<00:01, 17.38it/s]Capturing num tokens (num_tokens=208 avail_mem=22.97 GB):  66%|██████▌   | 38/58 [00:06<00:01, 17.38it/s]Capturing num tokens (num_tokens=192 avail_mem=22.96 GB):  66%|██████▌   | 38/58 [00:06<00:01, 17.38it/s]Capturing num tokens (num_tokens=192 avail_mem=22.96 GB):  71%|███████   | 41/58 [00:06<00:00, 19.64it/s]Capturing num tokens (num_tokens=176 avail_mem=22.95 GB):  71%|███████   | 41/58 [00:06<00:00, 19.64it/s]Capturing num tokens (num_tokens=160 avail_mem=22.95 GB):  71%|███████   | 41/58 [00:06<00:00, 19.64it/s]Capturing num tokens (num_tokens=144 avail_mem=22.94 GB):  71%|███████   | 41/58 [00:06<00:00, 19.64it/s]

    Capturing num tokens (num_tokens=144 avail_mem=22.94 GB):  76%|███████▌  | 44/58 [00:06<00:00, 21.49it/s]Capturing num tokens (num_tokens=128 avail_mem=22.94 GB):  76%|███████▌  | 44/58 [00:06<00:00, 21.49it/s]Capturing num tokens (num_tokens=112 avail_mem=22.93 GB):  76%|███████▌  | 44/58 [00:07<00:00, 21.49it/s]Capturing num tokens (num_tokens=96 avail_mem=22.93 GB):  76%|███████▌  | 44/58 [00:07<00:00, 21.49it/s] Capturing num tokens (num_tokens=96 avail_mem=22.93 GB):  81%|████████  | 47/58 [00:07<00:00, 22.67it/s]Capturing num tokens (num_tokens=80 avail_mem=22.92 GB):  81%|████████  | 47/58 [00:07<00:00, 22.67it/s]Capturing num tokens (num_tokens=64 avail_mem=22.91 GB):  81%|████████  | 47/58 [00:07<00:00, 22.67it/s]Capturing num tokens (num_tokens=48 avail_mem=22.92 GB):  81%|████████  | 47/58 [00:07<00:00, 22.67it/s]

    Capturing num tokens (num_tokens=48 avail_mem=22.92 GB):  86%|████████▌ | 50/58 [00:07<00:00, 24.15it/s]Capturing num tokens (num_tokens=32 avail_mem=22.91 GB):  86%|████████▌ | 50/58 [00:07<00:00, 24.15it/s]Capturing num tokens (num_tokens=28 avail_mem=22.90 GB):  86%|████████▌ | 50/58 [00:07<00:00, 24.15it/s]Capturing num tokens (num_tokens=24 avail_mem=22.90 GB):  86%|████████▌ | 50/58 [00:07<00:00, 24.15it/s]Capturing num tokens (num_tokens=24 avail_mem=22.90 GB):  91%|█████████▏| 53/58 [00:07<00:00, 25.48it/s]Capturing num tokens (num_tokens=20 avail_mem=22.89 GB):  91%|█████████▏| 53/58 [00:07<00:00, 25.48it/s]Capturing num tokens (num_tokens=16 avail_mem=22.88 GB):  91%|█████████▏| 53/58 [00:07<00:00, 25.48it/s]Capturing num tokens (num_tokens=12 avail_mem=22.88 GB):  91%|█████████▏| 53/58 [00:07<00:00, 25.48it/s]

    Capturing num tokens (num_tokens=12 avail_mem=22.88 GB):  97%|█████████▋| 56/58 [00:07<00:00, 25.60it/s]Capturing num tokens (num_tokens=8 avail_mem=22.87 GB):  97%|█████████▋| 56/58 [00:07<00:00, 25.60it/s] Capturing num tokens (num_tokens=4 avail_mem=22.86 GB):  97%|█████████▋| 56/58 [00:07<00:00, 25.60it/s]Capturing num tokens (num_tokens=4 avail_mem=22.86 GB): 100%|██████████| 58/58 [00:07<00:00,  7.75it/s]


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
      "population": 300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
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
      "population": 300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
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
    Generated text: England
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
    
    Generated text: Okay, so I'm trying to figure out how to respond to the user's request. They asked for the information and population of the capital of France in JSON format. Hmm, let's break this down step by step.
    
    First, I know that the capital of France is Paris. That's pretty much a given, right? So I can start by identifying the capital as Paris. Now, the user wants the population, so I need to find out the current population of Paris. I remember that population numbers can change over time, so I should look for the most recent data available.
    
    I think the population of Paris is quite large because it's a major city. I recall it's over 3 million, but I'm not exactly sure of the exact number. Maybe around 3.5 million? I should verify that to make sure. Wait, I think it's approximately 3,500,000 people. But I'm not 100% certain, so I should look it up to confirm.
    
    Once I have the population number, I need to structure it in JSON format. JSON, or JavaScript Object Notation, is a way to represent data as key-value pairs. So, I'll create an object with a key for the capital and another for the population. The key should probably be "capital" and "population" to make it clear.
    
    Putting it all together, the JSON should look something like this:
    
    {
      "capital": "Paris",
      "population": 3500000
    }
    
    But I should double-check the population number to ensure accuracy. Maybe it's 3,520,000 or 3,490,000? I think I remember it being around 3.5 million, but I might be slightly off. I'll stick with 3,500,000 unless I find more precise data.
    
    Wait, another thought: sometimes population figures are given in thousands or millions. Should I write it as a number or include the comma for better readability? In JSON, it's usually better to use numbers without commas, so 3500000 is correct.
    
    Okay, I think I've got it now. The JSON object will clearly state the capital as Paris and its population as approximately 3.5 million. I should make sure the syntax is correct, with proper quotation marks and commas.
    </think>
    
    ```json
    {
      "capital": "Paris",
      "population": 3500000
    }
    ```



```python
llm.shutdown()
```
