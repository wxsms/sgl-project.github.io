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
    [2026-04-24 09:48:29] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 09:48:30] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-24 09:48:31] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-24 09:48:33] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 09:48:38] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-24 09:48:39] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False
    [2026-04-24 09:48:39] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.07s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.15s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.14s/it]


    2026-04-24 09:48:47,515 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 09:48:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:56,  3.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:56,  3.11s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:29,  1.59s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:29,  1.59s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:55,  1.00s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:55,  1.00s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:38,  1.40it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:38,  1.40it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.83it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.83it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:22,  2.26it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:22,  2.26it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.75it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.75it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:14,  3.43it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:14,  3.43it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:11,  4.29it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:11,  4.29it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:05<00:11,  4.29it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:07,  6.00it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:07,  6.00it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:07,  6.00it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:05,  7.61it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:05,  7.61it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  7.61it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.24it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.24it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.24it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.22it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.22it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.22it/s]

    Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.22it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 14.18it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 14.18it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:06<00:01, 19.65it/s]

    Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:06<00:01, 19.65it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:06<00:01, 19.65it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:06<00:01, 19.65it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:06<00:00, 28.49it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:06<00:00, 28.49it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:06<00:00, 28.49it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:06<00:00, 28.49it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:06<00:00, 28.49it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:06<00:00, 28.49it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:06<00:00, 28.49it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:06<00:00, 28.49it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:00, 38.51it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:00, 38.51it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:00, 38.51it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:00, 38.51it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:00, 38.51it/s]

    Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:06<00:00, 38.51it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:06<00:00, 38.51it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:06<00:00, 38.51it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 46.59it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 46.59it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 46.59it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 46.59it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:06<00:00, 46.59it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:06<00:00, 46.59it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:06<00:00, 46.59it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:06<00:00, 46.59it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 52.73it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 52.73it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 52.73it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 52.73it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 52.73it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 52.73it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 52.73it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 52.73it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=111.87 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=111.87 GB):   2%|▏         | 1/58 [00:00<00:16,  3.54it/s]Capturing num tokens (num_tokens=7680 avail_mem=111.84 GB):   2%|▏         | 1/58 [00:00<00:16,  3.54it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=111.84 GB):   3%|▎         | 2/58 [00:00<00:15,  3.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=111.84 GB):   3%|▎         | 2/58 [00:00<00:15,  3.64it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=111.84 GB):   5%|▌         | 3/58 [00:00<00:14,  3.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=111.73 GB):   5%|▌         | 3/58 [00:00<00:14,  3.85it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=111.73 GB):   7%|▋         | 4/58 [00:01<00:13,  3.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=108.78 GB):   7%|▋         | 4/58 [00:01<00:13,  3.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=108.78 GB):   9%|▊         | 5/58 [00:01<00:13,  4.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=108.20 GB):   9%|▊         | 5/58 [00:01<00:13,  4.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=108.20 GB):  10%|█         | 6/58 [00:01<00:11,  4.46it/s]Capturing num tokens (num_tokens=5120 avail_mem=108.19 GB):  10%|█         | 6/58 [00:01<00:11,  4.46it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=108.19 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=108.08 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=108.08 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=106.67 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.28it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=106.67 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.57it/s]Capturing num tokens (num_tokens=3840 avail_mem=104.80 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.57it/s]Capturing num tokens (num_tokens=3840 avail_mem=104.80 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.13it/s]Capturing num tokens (num_tokens=3584 avail_mem=104.80 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.13it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=104.80 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.74it/s]Capturing num tokens (num_tokens=3328 avail_mem=104.80 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.74it/s]Capturing num tokens (num_tokens=3328 avail_mem=104.80 GB):  21%|██        | 12/58 [00:02<00:06,  7.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=104.79 GB):  21%|██        | 12/58 [00:02<00:06,  7.39it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=104.80 GB):  21%|██        | 12/58 [00:02<00:06,  7.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=104.80 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=104.80 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=104.80 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.64it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=104.80 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=104.79 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=104.80 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=104.80 GB):  31%|███       | 18/58 [00:02<00:03, 11.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=104.79 GB):  31%|███       | 18/58 [00:02<00:03, 11.59it/s]Capturing num tokens (num_tokens=1280 avail_mem=104.79 GB):  31%|███       | 18/58 [00:02<00:03, 11.59it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=104.79 GB):  31%|███       | 18/58 [00:02<00:03, 11.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=104.79 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.52it/s]Capturing num tokens (num_tokens=960 avail_mem=104.79 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.52it/s] Capturing num tokens (num_tokens=896 avail_mem=104.79 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.52it/s]Capturing num tokens (num_tokens=832 avail_mem=104.78 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.52it/s]Capturing num tokens (num_tokens=832 avail_mem=104.78 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.49it/s]Capturing num tokens (num_tokens=768 avail_mem=104.78 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.49it/s]Capturing num tokens (num_tokens=704 avail_mem=104.77 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.49it/s]

    Capturing num tokens (num_tokens=640 avail_mem=104.77 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.49it/s]Capturing num tokens (num_tokens=640 avail_mem=104.77 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.42it/s]Capturing num tokens (num_tokens=576 avail_mem=104.77 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.42it/s]Capturing num tokens (num_tokens=512 avail_mem=104.76 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.42it/s]Capturing num tokens (num_tokens=480 avail_mem=104.76 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.42it/s]Capturing num tokens (num_tokens=448 avail_mem=104.76 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.42it/s]Capturing num tokens (num_tokens=448 avail_mem=104.76 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=416 avail_mem=104.75 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=384 avail_mem=104.75 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]

    Capturing num tokens (num_tokens=352 avail_mem=104.74 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=320 avail_mem=104.74 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.11it/s]Capturing num tokens (num_tokens=320 avail_mem=104.74 GB):  60%|██████    | 35/58 [00:03<00:00, 27.24it/s]Capturing num tokens (num_tokens=288 avail_mem=104.74 GB):  60%|██████    | 35/58 [00:03<00:00, 27.24it/s]Capturing num tokens (num_tokens=256 avail_mem=104.73 GB):  60%|██████    | 35/58 [00:03<00:00, 27.24it/s]Capturing num tokens (num_tokens=240 avail_mem=104.73 GB):  60%|██████    | 35/58 [00:03<00:00, 27.24it/s]Capturing num tokens (num_tokens=224 avail_mem=104.73 GB):  60%|██████    | 35/58 [00:03<00:00, 27.24it/s]Capturing num tokens (num_tokens=224 avail_mem=104.73 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]Capturing num tokens (num_tokens=208 avail_mem=104.72 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]Capturing num tokens (num_tokens=192 avail_mem=104.72 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]

    Capturing num tokens (num_tokens=176 avail_mem=104.71 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]Capturing num tokens (num_tokens=160 avail_mem=104.71 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]Capturing num tokens (num_tokens=160 avail_mem=104.71 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.88it/s]Capturing num tokens (num_tokens=144 avail_mem=104.71 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.88it/s]Capturing num tokens (num_tokens=128 avail_mem=104.72 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.88it/s]Capturing num tokens (num_tokens=112 avail_mem=104.71 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.88it/s]Capturing num tokens (num_tokens=96 avail_mem=104.71 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.88it/s] Capturing num tokens (num_tokens=96 avail_mem=104.71 GB):  81%|████████  | 47/58 [00:03<00:00, 33.20it/s]Capturing num tokens (num_tokens=80 avail_mem=104.70 GB):  81%|████████  | 47/58 [00:03<00:00, 33.20it/s]Capturing num tokens (num_tokens=64 avail_mem=104.70 GB):  81%|████████  | 47/58 [00:03<00:00, 33.20it/s]

    Capturing num tokens (num_tokens=48 avail_mem=104.70 GB):  81%|████████  | 47/58 [00:03<00:00, 33.20it/s]Capturing num tokens (num_tokens=32 avail_mem=104.69 GB):  81%|████████  | 47/58 [00:03<00:00, 33.20it/s]Capturing num tokens (num_tokens=32 avail_mem=104.69 GB):  88%|████████▊ | 51/58 [00:03<00:00, 34.33it/s]Capturing num tokens (num_tokens=28 avail_mem=104.69 GB):  88%|████████▊ | 51/58 [00:03<00:00, 34.33it/s]Capturing num tokens (num_tokens=24 avail_mem=104.69 GB):  88%|████████▊ | 51/58 [00:03<00:00, 34.33it/s]Capturing num tokens (num_tokens=20 avail_mem=104.68 GB):  88%|████████▊ | 51/58 [00:03<00:00, 34.33it/s]Capturing num tokens (num_tokens=16 avail_mem=104.68 GB):  88%|████████▊ | 51/58 [00:03<00:00, 34.33it/s]Capturing num tokens (num_tokens=16 avail_mem=104.68 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.03it/s]Capturing num tokens (num_tokens=12 avail_mem=104.68 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.03it/s]Capturing num tokens (num_tokens=8 avail_mem=104.67 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.03it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=104.67 GB):  95%|█████████▍| 55/58 [00:03<00:00, 35.03it/s]Capturing num tokens (num_tokens=4 avail_mem=104.67 GB): 100%|██████████| 58/58 [00:03<00:00, 14.64it/s]


    [2026-04-24 09:49:00] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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


<strong style='color: #00008B;'>reasoing_content: Okay, the user is in New York and wants the current date and time along with the weather. I need to figure out how to get this using the provided functions.<br><br>First, I should use the get_current_date function because it gives the date and time for a specific timezone. The user is in New York, so I'll set the timezone parameter to 'America/New_York'.<br><br>Next, I need the weather information. The get_current_weather function requires the city, state, and unit. The city is New York, the state is NY, and I'll choose Fahrenheit since the user didn't specify otherwise.<br><br>I'll structure the response by first calling get_current_date with the timezone and then get_current_weather with the required parameters. I'll make sure to include the function calls in the specified format and add the sources as per the instructions.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nWait, maybe I should check the latest statistics. I recall that in recent years, Paris has seen a slight increase. Let me think... I believe the population is approximately 2,150,000 as of 2023. That seems about right, but I\'m not 100% certain. I should make sure to present this information accurately.\n\nNext, I need to structure this into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" for context. The city is Paris, the population is 2,150,000, and the country is France.\n\nI should also consider if the user might need more details, like the exact year of the population figure. Including that could be helpful, so I\'ll add "year": 2023. That way, the user knows the data is up to date.\n\nPutting it all together, the JSON should look clean and well-structured. I\'ll make sure the syntax is correct, with proper commas and quotation marks. No markdown, just plain JSON.\n\nI think that\'s all. The user probably just needs the information quickly, so keeping it concise is key. I\'ll present the JSON without any extra fluff.\n</think>{\n\n"name": "Paris",\n"population": 21500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 14190, 11, 7196, 358, 1265, 1779, 279, 5535, 13142, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 8112, 5263, 13, 6771, 752, 1744, 1112, 358, 4411, 279, 7042, 374, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 3654, 13, 358, 1265, 1281, 2704, 311, 3042, 419, 1995, 29257, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 369, 2266, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 4734, 1042, 315, 279, 7042, 7071, 13, 55121, 429, 1410, 387, 10950, 11, 773, 358, 3278, 912, 330, 3157, 788, 220, 17, 15, 17, 18, 13, 2938, 1616, 11, 279, 1196, 8788, 279, 821, 374, 705, 311, 2400, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 4240, 323, 1632, 12, 51143, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 13, 2308, 50494, 11, 1101, 14396, 4718, 382, 40, 1744, 429, 594, 678, 13, 576, 1196, 4658, 1101, 3880, 279, 1995, 6157, 11, 773, 10282, 432, 63594, 374, 1376, 13, 358, 3278, 3042, 279, 4718, 2041, 894, 4960, 1320, 1362, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '05a5484711f640dfbbc29bbb727dc895', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 367, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 18.774960584938526, 'response_sent_to_client_ts': 1777024184.3290184}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should verify that.\n\nWait, maybe I should check some reliable sources to confirm the population. I recall that the population figures can vary depending on the source and the year. For example, the 2020 census might have a slightly different number than the 2021 estimate. I think the population was around 2,165,000 in 2021, but I\'m not 100% certain. I should make sure to use the most accurate and up-to-date information.\n\nAlso, the user wants the information in JSON format. JSON is a data interchange format, so I\'ll need to structure the data accordingly. I should include the city name, population, and maybe the year of the data. It\'s important to present the information clearly and accurately, so I\'ll double-check the numbers to avoid any mistakes.\n\nI should also consider if there are any other relevant details the user might find useful, like the area of the city or some key facts about it. But since the user specifically asked for population, I\'ll focus on that. Maybe adding a note about the population figure being approximate would be helpful, just in case.\n\nPutting it all together, I\'ll structure the JSON with the city name, population, and the year. I\'ll make sure the syntax is correct, using quotation marks and commas appropriately. I\'ll also keep the language clear and straightforward so that the user can easily understand the information.\n\nFinally, I\'ll review the JSON to ensure there are no errors and that the data is accurate. This way, the user gets a reliable and well-formatted response to their query.\n</think>{"name": "Paris", "population": 2165000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 14190, 11, 7196, 358, 1265, 1779, 1045, 14720, 8173, 311, 7683, 279, 7042, 13, 358, 19091, 429, 279, 7042, 12396, 646, 13289, 11649, 389, 279, 2530, 323, 279, 1042, 13, 1752, 3110, 11, 279, 220, 17, 15, 17, 15, 43602, 2578, 614, 264, 10078, 2155, 1372, 1091, 279, 220, 17, 15, 17, 16, 16045, 13, 358, 1744, 279, 7042, 572, 2163, 220, 17, 11, 16, 21, 20, 11, 15, 15, 15, 304, 220, 17, 15, 17, 16, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 3654, 13, 358, 1265, 1281, 2704, 311, 990, 279, 1429, 13382, 323, 705, 4686, 18413, 1995, 382, 13394, 11, 279, 1196, 6801, 279, 1995, 304, 4718, 3561, 13, 4718, 374, 264, 821, 51263, 3561, 11, 773, 358, 3278, 1184, 311, 5944, 279, 821, 27079, 13, 358, 1265, 2924, 279, 3283, 829, 11, 7042, 11, 323, 7196, 279, 1042, 315, 279, 821, 13, 1084, 594, 2989, 311, 3042, 279, 1995, 9355, 323, 29257, 11, 773, 358, 3278, 1990, 15934, 279, 5109, 311, 5648, 894, 20643, 382, 40, 1265, 1083, 2908, 421, 1052, 525, 894, 1008, 9760, 3565, 279, 1196, 2578, 1477, 5390, 11, 1075, 279, 3082, 315, 279, 3283, 476, 1045, 1376, 13064, 911, 432, 13, 1988, 2474, 279, 1196, 11689, 4588, 369, 7042, 11, 358, 3278, 5244, 389, 429, 13, 10696, 7842, 264, 5185, 911, 279, 7042, 7071, 1660, 44868, 1035, 387, 10950, 11, 1101, 304, 1142, 382, 97904, 432, 678, 3786, 11, 358, 3278, 5944, 279, 4718, 448, 279, 3283, 829, 11, 7042, 11, 323, 279, 1042, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 1667, 54231, 15423, 323, 76602, 34901, 13, 358, 3278, 1083, 2506, 279, 4128, 2797, 323, 30339, 773, 429, 279, 1196, 646, 6707, 3535, 279, 1995, 382, 23949, 11, 358, 3278, 3395, 279, 4718, 311, 5978, 1052, 525, 902, 5975, 323, 429, 279, 821, 374, 13382, 13, 1096, 1616, 11, 279, 1196, 5221, 264, 14720, 323, 1632, 8460, 12127, 2033, 311, 862, 3239, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 21, 20, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'e2b23a65cb284ca0a0c8b7106d9b3ecd', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 428, 'completion_tokens': 447, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.203268349170685, 'response_sent_to_client_ts': 1777024188.5469294}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '8e56056ad6874bb4954071005235525e', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1225547818467021, 'response_sent_to_client_ts': 1777024188.7004397}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'b17604fa8c4a414eb658d4fb36f15646', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.12246829271316528, 'response_sent_to_client_ts': 1777024188.700453}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'f274f0fb5fac46108a96657ad8f324f5', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.12219398468732834, 'response_sent_to_client_ts': 1777024188.7004583}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'e4dc34e1b55e4b5c8f09ca8735333419', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 20.544473477639258, 'response_sent_to_client_ts': 1777024209.2531996}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Hmm, let me break this down. First, I need to figure out what country\'s capital they\'re referring to, but since France is a country, its capital is definitely巴黎. I remember that Paris is the main city there.\n\nNow, the user wants the information and population in JSON. JSON stands for JavaScript Object Notation, which is a way to structure data. So, I need to create a JSON object with the relevant details. The main elements would be the population, the capital, the country, and maybe some other info like the area and time zone.\n\nI should start by gathering the most recent population data. From what I know, as of the latest data, Paris has a population around 2.1 million people. I\'ll include that. Also, mentioning the metropolitan area along with Paris might be helpful because its population is higher. I think the metropolitan area population was about 12 million if I remember correctly.\n\nNext, the location of the capital, Paris, is important. I\'ll write it out in both French and English for clarity. The country is naturally France, so that\'s straightforward.\n\nFor the city details, mentioning the prefecture helps. The prefecture of the Seine is indeed known as le cidre, so I\'ll include that. Also, Paris is famous for landmarks like the Eiffel Tower and the Louvre, so adding those keywords could make it more appealing, maybe for optimization in some systems.\n\nCoordinates are next. Paris\'s latitude and longitude are well-known. I\'ll put those in, making sure to include the degree symbols and decimal places for accuracy.\n\nTime zone is another useful piece of information. Paris is in Central European Time, so I\'ll specify that and include the offset notation, like +1:00. I think it\'s +1:00 because of the daylight saving time in some years, but it\'s standard in others, so I\'ll just put +1:00 for simplicity.\n\nI should make sure the JSON structure is correct. It should be a single object with keys like "capital", "country", etc., each having their respective values as strings or numbers. I\'ll avoid any markdown since the user requested plain JSON. \n\nAlso, considering the user\'s request, they might not just want the basic info—perhaps they need it for a project or a presentation. By providing all these details, it adds more value beyond just the population. \n\nI need to double-check the population numbers to ensure accuracy. Sometimes, cities like Paris can be tricky because the metropolitan area is different from the city proper. Including both makes the data more comprehensive.\n\nFinally, did the user specify any particular order for the keys? They didn\'t, so I\'ll arrange them alphabetically for uniformity. That way, it\'s easier to read and doesn\'t require the user to know the order.\n\nPutting it all together, I\'ll structure the JSON with each key, ensuring that strings are in quotes and numbers are without. Making sure everything is correctly formatted so it can be easily parsed if needed.\n\nI think that\'s about it. I\'ve covered all the necessary aspects the user asked for, presented in a clear and organized JSON format. Hopefully, this meets their needs and provides all the useful information they\'re looking for.\n</think>\n\nHere is the population of the capital of France (Paris) in JSON format:\n\n```json\n{\n  "capital": {\n    "name": "Paris",\n    "french_name": "Le cidre",\n    "coordinates": {\n      "latitude": "48°51\'14.88\'E",\n      "longitude": "2°21\'9.58\'N"\n    }\n  },\n  "country": "France",\n  "population": "2,157,903",\n  "metropolitan_area_population": "11,300,000",\n  "area": "10.55 km²",\n  "time_zone": "UTC+1:00 (Central European Time)"\n}\n```', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 88190, 11, 1077, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 7071, 700, 1128, 3146, 594, 6722, 807, 2299, 22023, 311, 11, 714, 2474, 9625, 374, 264, 3146, 11, 1181, 6722, 374, 8491, 106004, 13, 358, 6099, 429, 12095, 374, 279, 1887, 3283, 1052, 382, 7039, 11, 279, 1196, 6801, 279, 1995, 323, 7042, 304, 4718, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 2055, 11, 358, 1184, 311, 1855, 264, 4718, 1633, 448, 279, 9760, 3565, 13, 576, 1887, 5424, 1035, 387, 279, 7042, 11, 279, 6722, 11, 279, 3146, 11, 323, 7196, 1045, 1008, 3546, 1075, 279, 3082, 323, 882, 10143, 382, 40, 1265, 1191, 553, 22888, 279, 1429, 3213, 7042, 821, 13, 5542, 1128, 358, 1414, 11, 438, 315, 279, 5535, 821, 11, 12095, 702, 264, 7042, 2163, 220, 17, 13, 16, 3526, 1251, 13, 358, 3278, 2924, 429, 13, 7281, 11, 44291, 279, 57406, 3082, 3156, 448, 12095, 2578, 387, 10950, 1576, 1181, 7042, 374, 5080, 13, 358, 1744, 279, 57406, 3082, 7042, 572, 911, 220, 16, 17, 3526, 421, 358, 6099, 12440, 382, 5847, 11, 279, 3728, 315, 279, 6722, 11, 12095, 11, 374, 2989, 13, 358, 3278, 3270, 432, 700, 304, 2176, 8585, 323, 6364, 369, 31273, 13, 576, 3146, 374, 17712, 9625, 11, 773, 429, 594, 30339, 382, 2461, 279, 3283, 3565, 11, 44291, 279, 89445, 552, 8609, 13, 576, 89445, 552, 315, 279, 1345, 482, 374, 12824, 3881, 438, 512, 32141, 265, 11, 773, 358, 3278, 2924, 429, 13, 7281, 11, 12095, 374, 11245, 369, 59924, 1075, 279, 468, 3092, 301, 21938, 323, 279, 9729, 48506, 11, 773, 7842, 1846, 20844, 1410, 1281, 432, 803, 32252, 11, 7196, 369, 25262, 304, 1045, 5942, 382, 43876, 525, 1790, 13, 12095, 594, 20849, 323, 20515, 525, 1632, 21309, 13, 358, 3278, 2182, 1846, 304, 11, 3259, 2704, 311, 2924, 279, 8381, 17738, 323, 12122, 7482, 369, 13403, 382, 1462, 10143, 374, 2441, 5390, 6573, 315, 1995, 13, 12095, 374, 304, 10684, 7513, 4120, 11, 773, 358, 3278, 13837, 429, 323, 2924, 279, 4347, 44197, 11, 1075, 488, 16, 25, 15, 15, 13, 358, 1744, 432, 594, 488, 16, 25, 15, 15, 1576, 315, 279, 52021, 13997, 882, 304, 1045, 1635, 11, 714, 432, 594, 5297, 304, 3800, 11, 773, 358, 3278, 1101, 2182, 488, 16, 25, 15, 15, 369, 38975, 382, 40, 1265, 1281, 2704, 279, 4718, 5944, 374, 4396, 13, 1084, 1265, 387, 264, 3175, 1633, 448, 6894, 1075, 330, 65063, 497, 330, 11141, 497, 4992, 2572, 1817, 3432, 862, 19511, 2750, 438, 9069, 476, 5109, 13, 358, 3278, 5648, 894, 50494, 2474, 279, 1196, 11223, 14396, 4718, 13, 4710, 13394, 11, 12831, 279, 1196, 594, 1681, 11, 807, 2578, 537, 1101, 1366, 279, 6770, 3546, 2293, 65272, 807, 1184, 432, 369, 264, 2390, 476, 264, 15496, 13, 3216, 8241, 678, 1493, 3565, 11, 432, 11367, 803, 897, 7797, 1101, 279, 7042, 13, 4710, 40, 1184, 311, 1990, 15934, 279, 7042, 5109, 311, 5978, 13403, 13, 17688, 11, 9720, 1075, 12095, 646, 387, 33453, 1576, 279, 57406, 3082, 374, 2155, 504, 279, 3283, 6169, 13, 55121, 2176, 3643, 279, 821, 803, 15817, 382, 23949, 11, 1521, 279, 1196, 13837, 894, 3953, 1973, 369, 279, 6894, 30, 2379, 3207, 944, 11, 773, 358, 3278, 30893, 1105, 27790, 2673, 369, 13794, 487, 13, 2938, 1616, 11, 432, 594, 8661, 311, 1349, 323, 3171, 944, 1373, 279, 1196, 311, 1414, 279, 1973, 382, 97904, 432, 678, 3786, 11, 358, 3278, 5944, 279, 4718, 448, 1817, 1376, 11, 22573, 429, 9069, 525, 304, 17194, 323, 5109, 525, 2041, 13, 24288, 2704, 4297, 374, 12440, 23126, 773, 432, 646, 387, 6707, 15676, 421, 4362, 382, 40, 1744, 429, 594, 911, 432, 13, 358, 3003, 9761, 678, 279, 5871, 13566, 279, 1196, 4588, 369, 11, 10449, 304, 264, 2797, 323, 16645, 4718, 3561, 13, 37894, 11, 419, 20027, 862, 3880, 323, 5707, 678, 279, 5390, 1995, 807, 2299, 3330, 369, 624, 151649, 271, 8420, 374, 279, 7042, 315, 279, 6722, 315, 9625, 320, 59604, 8, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 341, 262, 330, 606, 788, 330, 59604, 756, 262, 330, 69, 6655, 1269, 788, 330, 2304, 32141, 265, 756, 262, 330, 34739, 788, 341, 414, 330, 23718, 788, 330, 19, 23, 11616, 20, 16, 6, 16, 19, 13, 23, 23, 87948, 756, 414, 330, 25446, 788, 330, 17, 11616, 17, 16, 6, 24, 13, 20, 23, 6, 45, 698, 262, 456, 220, 1153, 220, 330, 11141, 788, 330, 49000, 756, 220, 330, 44441, 788, 330, 17, 11, 16, 20, 22, 11, 24, 15, 18, 756, 220, 330, 4059, 30511, 15030, 74572, 788, 330, 16, 16, 11, 18, 15, 15, 11, 15, 15, 15, 756, 220, 330, 4798, 788, 330, 16, 15, 13, 20, 20, 13136, 29456, 756, 220, 330, 1678, 28692, 788, 330, 21183, 10, 16, 25, 15, 15, 320, 43403, 7513, 4120, 12954, 532, 73594, 151643], 'meta_info': {'id': '2eee50a51c714b0f89e0bdbdac4d63b1', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 685, 'completion_tokens': 847, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 9.007576651871204, 'response_sent_to_client_ts': 1777024218.2724044}}</strong>



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
    [2026-04-24 09:50:28] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.30s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.47s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.44s/it]


    2026-04-24 09:50:37,177 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 09:50:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:57,  3.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:57,  3.11s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:24,  1.51s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:24,  1.51s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:50,  1.10it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:50,  1.10it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.20it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.20it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.84it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.84it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.58it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.58it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.36it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.36it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.24it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.24it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.24it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.92it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.92it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.92it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.46it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.46it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.46it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.98it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.98it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.98it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.91it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.91it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.91it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.91it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.28it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.28it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.28it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.28it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.28it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:00, 29.64it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:00, 29.64it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:00, 29.64it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:00, 29.64it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.64it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.64it/s]

    Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:00, 29.64it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:00, 29.64it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 39.54it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 39.54it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 39.54it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 39.54it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 39.54it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 39.54it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 39.54it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 39.54it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 47.60it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 47.60it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 47.60it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 47.60it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 47.60it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 47.60it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 47.60it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 47.60it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 53.37it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 53.37it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 53.37it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 53.37it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 53.37it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 53.37it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 53.37it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 53.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=104.52 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=104.52 GB):   2%|▏         | 1/58 [00:00<00:16,  3.50it/s]Capturing num tokens (num_tokens=7680 avail_mem=104.48 GB):   2%|▏         | 1/58 [00:00<00:16,  3.50it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=104.48 GB):   3%|▎         | 2/58 [00:00<00:15,  3.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=104.48 GB):   3%|▎         | 2/58 [00:00<00:15,  3.60it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=104.48 GB):   5%|▌         | 3/58 [00:00<00:14,  3.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=104.47 GB):   5%|▌         | 3/58 [00:00<00:14,  3.76it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=104.47 GB):   7%|▋         | 4/58 [00:01<00:14,  3.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=103.89 GB):   7%|▋         | 4/58 [00:01<00:14,  3.84it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=103.89 GB):   9%|▊         | 5/58 [00:01<00:12,  4.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=103.89 GB):   9%|▊         | 5/58 [00:01<00:12,  4.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=103.89 GB):  10%|█         | 6/58 [00:01<00:11,  4.57it/s]Capturing num tokens (num_tokens=5120 avail_mem=103.89 GB):  10%|█         | 6/58 [00:01<00:11,  4.57it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=103.89 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=103.87 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=103.87 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=103.87 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.41it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=103.87 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=103.87 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=103.87 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=103.84 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.36it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=103.84 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=103.84 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=103.84 GB):  21%|██        | 12/58 [00:02<00:06,  7.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=103.84 GB):  21%|██        | 12/58 [00:02<00:06,  7.51it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=103.84 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=103.82 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=103.82 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=103.82 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.68it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=103.73 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=103.73 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=103.17 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=103.17 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.09it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=103.17 GB):  31%|███       | 18/58 [00:02<00:03, 10.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=103.16 GB):  31%|███       | 18/58 [00:02<00:03, 10.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=103.16 GB):  31%|███       | 18/58 [00:02<00:03, 10.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=103.16 GB):  31%|███       | 18/58 [00:02<00:03, 10.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=103.16 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.15it/s]Capturing num tokens (num_tokens=960 avail_mem=103.16 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.15it/s] Capturing num tokens (num_tokens=896 avail_mem=103.16 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.15it/s]

    Capturing num tokens (num_tokens=832 avail_mem=103.15 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.15it/s]Capturing num tokens (num_tokens=832 avail_mem=103.15 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.09it/s]Capturing num tokens (num_tokens=768 avail_mem=103.15 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.09it/s]Capturing num tokens (num_tokens=704 avail_mem=103.14 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.09it/s]Capturing num tokens (num_tokens=640 avail_mem=103.14 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.09it/s]Capturing num tokens (num_tokens=640 avail_mem=103.14 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.07it/s]Capturing num tokens (num_tokens=576 avail_mem=103.14 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.07it/s]Capturing num tokens (num_tokens=512 avail_mem=103.13 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.07it/s]

    Capturing num tokens (num_tokens=480 avail_mem=103.13 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.07it/s]Capturing num tokens (num_tokens=448 avail_mem=103.13 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.07it/s]Capturing num tokens (num_tokens=448 avail_mem=103.13 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.79it/s]Capturing num tokens (num_tokens=416 avail_mem=103.12 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.79it/s]Capturing num tokens (num_tokens=384 avail_mem=103.12 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.79it/s]Capturing num tokens (num_tokens=352 avail_mem=103.12 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.79it/s]Capturing num tokens (num_tokens=320 avail_mem=103.11 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.79it/s]Capturing num tokens (num_tokens=320 avail_mem=103.11 GB):  60%|██████    | 35/58 [00:03<00:00, 26.88it/s]Capturing num tokens (num_tokens=288 avail_mem=103.11 GB):  60%|██████    | 35/58 [00:03<00:00, 26.88it/s]

    Capturing num tokens (num_tokens=256 avail_mem=103.10 GB):  60%|██████    | 35/58 [00:03<00:00, 26.88it/s]Capturing num tokens (num_tokens=240 avail_mem=103.10 GB):  60%|██████    | 35/58 [00:03<00:00, 26.88it/s]Capturing num tokens (num_tokens=224 avail_mem=103.10 GB):  60%|██████    | 35/58 [00:03<00:00, 26.88it/s]Capturing num tokens (num_tokens=224 avail_mem=103.10 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.41it/s]Capturing num tokens (num_tokens=208 avail_mem=103.09 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.41it/s]Capturing num tokens (num_tokens=192 avail_mem=103.09 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.41it/s]Capturing num tokens (num_tokens=176 avail_mem=103.09 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.41it/s]Capturing num tokens (num_tokens=160 avail_mem=103.08 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.41it/s]Capturing num tokens (num_tokens=160 avail_mem=103.08 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.34it/s]Capturing num tokens (num_tokens=144 avail_mem=103.08 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.34it/s]

    Capturing num tokens (num_tokens=128 avail_mem=103.09 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.34it/s]Capturing num tokens (num_tokens=112 avail_mem=103.03 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.34it/s]Capturing num tokens (num_tokens=96 avail_mem=103.02 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.34it/s] Capturing num tokens (num_tokens=96 avail_mem=103.02 GB):  81%|████████  | 47/58 [00:03<00:00, 31.99it/s]Capturing num tokens (num_tokens=80 avail_mem=103.02 GB):  81%|████████  | 47/58 [00:03<00:00, 31.99it/s]Capturing num tokens (num_tokens=64 avail_mem=103.01 GB):  81%|████████  | 47/58 [00:03<00:00, 31.99it/s]Capturing num tokens (num_tokens=48 avail_mem=103.01 GB):  81%|████████  | 47/58 [00:03<00:00, 31.99it/s]Capturing num tokens (num_tokens=32 avail_mem=103.01 GB):  81%|████████  | 47/58 [00:03<00:00, 31.99it/s]

    Capturing num tokens (num_tokens=32 avail_mem=103.01 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.27it/s]Capturing num tokens (num_tokens=28 avail_mem=103.01 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.27it/s]Capturing num tokens (num_tokens=24 avail_mem=103.00 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.27it/s]Capturing num tokens (num_tokens=20 avail_mem=103.00 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.27it/s]Capturing num tokens (num_tokens=16 avail_mem=102.99 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.27it/s]Capturing num tokens (num_tokens=16 avail_mem=102.99 GB):  95%|█████████▍| 55/58 [00:03<00:00, 33.90it/s]Capturing num tokens (num_tokens=12 avail_mem=102.99 GB):  95%|█████████▍| 55/58 [00:03<00:00, 33.90it/s]Capturing num tokens (num_tokens=8 avail_mem=102.99 GB):  95%|█████████▍| 55/58 [00:04<00:00, 33.90it/s] Capturing num tokens (num_tokens=4 avail_mem=102.98 GB):  95%|█████████▍| 55/58 [00:04<00:00, 33.90it/s]Capturing num tokens (num_tokens=4 avail_mem=102.98 GB): 100%|██████████| 58/58 [00:04<00:00, 14.29it/s]


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
    Generated text: Rome is the capital of Italy
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
    
    Generated text: Okay, so I need to figure out how to respond to this user's query. They asked for the information and population of the capital of France in JSON format. Let me start by breaking down what they're asking for. 
    
    First, the capital of France is Paris. That part is straightforward. But the user wants the population as well. I should check the current population of Paris. I know that Paris is a major city, so its population is likely around 2 million. However, I remember hearing that the population has been growing over the years. I think it's around 2.17 million, but I'm not entirely sure. Maybe I should verify that to make sure I provide the most accurate information.
    
    Now, the user wants this information in JSON format. JSON is a data interchange format that's easy for humans to read and write, and machines to parse and generate. So, I need to structure the data accordingly. Typically, in JSON, you use key-value pairs within curly braces. So, the keys could be "capital" and "population", and the corresponding values would be the city name and the population number.
    
    I should also consider the format. The user mentioned "JSON format," so I should present the data without any additional text or explanation unless necessary. Just the JSON object would suffice unless the user specifies otherwise.
    
    Wait, the user didn't specify whether they want the population as a number or perhaps with some additional info like growth rate or exact figure. Since they only asked for the population, I'll stick to providing just the number. I should make sure the number is accurate. Let me think: as of recent estimates, Paris has a population around 2.17 million. I think that's the latest figure available. 
    
    Is there any other information they might need? They only asked for the capital and its population, so I don't need to include anything else. I should also ensure that the JSON is properly formatted, with correct syntax to avoid any parsing errors. 
    
    I should also consider if the user might be using this data for something specific, like a project or a report. Therefore, providing an accurate and recent population figure is crucial. Maybe I can double-check the current population from a reliable source to confirm. 
    
    Another thought: should I mention that the population figure is approximate? Sometimes population numbers can fluctuate, so it might be good to include a note or a disclaimer. However, the user didn't ask for that, so I'll stick to just providing the data as requested.
    
    Putting it all together, the JSON object will have two key-value pairs: one for the capital and one for the population. The values will be the appropriate strings or numbers. 
    
    So, the JSON should look like this:
    
    {
      "capital": "Paris",
      "population": 2170000
    }
    
    Wait, 2170000 is 2,170,000, which is 2.17 million. That matches the figure I thought of. I think this should satisfy the user's request accurately and succinctly. 
    
    I should make sure there are no typos or formatting issues. Each key is in double quotes, and the string values are enclosed in double quotes. The entire structure is within curly braces. 
    
    Alright, I think I've covered all the bases here. The user asked for the capital and population in JSON, and I provided that. I double-checked the population figure to ensure accuracy and presented it in the correct format. I didn't add any extra information, keeping it as straightforward as possible.
    </think>
    
    ```json
    {
      "capital": "Paris",
      "population": 2170000
    }
    ```



```python
llm.shutdown()
```
