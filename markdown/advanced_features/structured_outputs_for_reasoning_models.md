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
    [2026-04-26 08:03:43] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-26 08:03:44] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-26 08:03:45] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-26 08:03:46] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-26 08:03:53] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-26 08:03:54] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False
    [2026-04-26 08:03:54] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:00<00:00,  1.05it/s]

    Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.04it/s]Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.04it/s]


    2026-04-26 08:04:01,144 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 08:04:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:05,  5.36s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:05,  5.36s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:10,  2.32s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:10,  2.32s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:14,  1.35s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:14,  1.35s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:32,  1.61it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:32,  1.61it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:24,  2.16it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:24,  2.16it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.83it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.83it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.58it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.58it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.45it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.45it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.45it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.17it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.17it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.17it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.61it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.61it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:05,  7.61it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  8.99it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  8.99it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  8.99it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.70it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.70it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.70it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 10.70it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 13.70it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 13.70it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 13.70it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 13.70it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 13.70it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.25it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.25it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.25it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.25it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.25it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.25it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.32it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.32it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.32it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.32it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.32it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.32it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.32it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:07<00:01, 26.32it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.64it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.64it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.64it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.64it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.64it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.64it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.64it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.64it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.64it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 46.36it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 46.36it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 46.36it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 46.36it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 46.36it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 46.36it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 46.36it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 46.36it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:07<00:00, 52.19it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:07<00:00, 52.19it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:07<00:00, 52.19it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:07<00:00, 52.19it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:07<00:00, 52.19it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:07<00:00, 52.19it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:07<00:00, 52.19it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:07<00:00, 52.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=108.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=108.74 GB):   2%|▏         | 1/58 [00:00<00:16,  3.54it/s]Capturing num tokens (num_tokens=7680 avail_mem=108.71 GB):   2%|▏         | 1/58 [00:00<00:16,  3.54it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=108.71 GB):   3%|▎         | 2/58 [00:00<00:15,  3.67it/s]Capturing num tokens (num_tokens=7168 avail_mem=108.70 GB):   3%|▎         | 2/58 [00:00<00:15,  3.67it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=108.70 GB):   5%|▌         | 3/58 [00:00<00:14,  3.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=108.70 GB):   5%|▌         | 3/58 [00:00<00:14,  3.88it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=108.70 GB):   7%|▋         | 4/58 [00:01<00:12,  4.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=108.70 GB):   7%|▋         | 4/58 [00:01<00:12,  4.17it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=108.70 GB):   9%|▊         | 5/58 [00:01<00:12,  4.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=108.70 GB):   9%|▊         | 5/58 [00:01<00:12,  4.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=108.70 GB):  10%|█         | 6/58 [00:01<00:10,  4.80it/s]Capturing num tokens (num_tokens=5120 avail_mem=108.70 GB):  10%|█         | 6/58 [00:01<00:10,  4.80it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=108.70 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.18it/s]Capturing num tokens (num_tokens=4608 avail_mem=108.70 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.18it/s]Capturing num tokens (num_tokens=4608 avail_mem=108.70 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=108.70 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.68it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=108.70 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=108.70 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=108.70 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.65it/s]Capturing num tokens (num_tokens=3584 avail_mem=108.70 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.65it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=108.70 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=108.70 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=108.70 GB):  21%|██        | 12/58 [00:02<00:05,  7.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=108.70 GB):  21%|██        | 12/58 [00:02<00:05,  7.81it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=108.70 GB):  21%|██        | 12/58 [00:02<00:05,  7.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=108.70 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.91it/s]Capturing num tokens (num_tokens=2560 avail_mem=108.69 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=108.69 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.91it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=108.69 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=108.69 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.26it/s]Capturing num tokens (num_tokens=1792 avail_mem=108.69 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.26it/s]Capturing num tokens (num_tokens=1792 avail_mem=108.69 GB):  31%|███       | 18/58 [00:02<00:03, 11.87it/s]Capturing num tokens (num_tokens=1536 avail_mem=108.68 GB):  31%|███       | 18/58 [00:02<00:03, 11.87it/s]Capturing num tokens (num_tokens=1280 avail_mem=108.68 GB):  31%|███       | 18/58 [00:02<00:03, 11.87it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=108.67 GB):  31%|███       | 18/58 [00:02<00:03, 11.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=108.67 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.86it/s]Capturing num tokens (num_tokens=960 avail_mem=108.67 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.86it/s] Capturing num tokens (num_tokens=896 avail_mem=108.66 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.86it/s]Capturing num tokens (num_tokens=832 avail_mem=108.66 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.86it/s]Capturing num tokens (num_tokens=832 avail_mem=108.66 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.89it/s]Capturing num tokens (num_tokens=768 avail_mem=108.65 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.89it/s]Capturing num tokens (num_tokens=704 avail_mem=108.65 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.89it/s]

    Capturing num tokens (num_tokens=640 avail_mem=108.65 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.89it/s]Capturing num tokens (num_tokens=640 avail_mem=108.65 GB):  47%|████▋     | 27/58 [00:02<00:01, 20.79it/s]Capturing num tokens (num_tokens=576 avail_mem=108.64 GB):  47%|████▋     | 27/58 [00:02<00:01, 20.79it/s]Capturing num tokens (num_tokens=512 avail_mem=108.64 GB):  47%|████▋     | 27/58 [00:02<00:01, 20.79it/s]Capturing num tokens (num_tokens=480 avail_mem=108.64 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.79it/s]Capturing num tokens (num_tokens=448 avail_mem=108.64 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.79it/s]Capturing num tokens (num_tokens=448 avail_mem=108.64 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.57it/s]Capturing num tokens (num_tokens=416 avail_mem=108.63 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.57it/s]Capturing num tokens (num_tokens=384 avail_mem=108.63 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.57it/s]

    Capturing num tokens (num_tokens=352 avail_mem=108.62 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.57it/s]Capturing num tokens (num_tokens=320 avail_mem=108.62 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.57it/s]Capturing num tokens (num_tokens=320 avail_mem=108.62 GB):  60%|██████    | 35/58 [00:03<00:00, 27.71it/s]Capturing num tokens (num_tokens=288 avail_mem=108.62 GB):  60%|██████    | 35/58 [00:03<00:00, 27.71it/s]Capturing num tokens (num_tokens=256 avail_mem=108.62 GB):  60%|██████    | 35/58 [00:03<00:00, 27.71it/s]Capturing num tokens (num_tokens=240 avail_mem=108.61 GB):  60%|██████    | 35/58 [00:03<00:00, 27.71it/s]Capturing num tokens (num_tokens=224 avail_mem=108.61 GB):  60%|██████    | 35/58 [00:03<00:00, 27.71it/s]Capturing num tokens (num_tokens=224 avail_mem=108.61 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.35it/s]Capturing num tokens (num_tokens=208 avail_mem=108.61 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.35it/s]Capturing num tokens (num_tokens=192 avail_mem=108.60 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.35it/s]

    Capturing num tokens (num_tokens=176 avail_mem=108.60 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.35it/s]Capturing num tokens (num_tokens=160 avail_mem=108.60 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.35it/s]Capturing num tokens (num_tokens=160 avail_mem=108.60 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.31it/s]Capturing num tokens (num_tokens=144 avail_mem=108.59 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.31it/s]Capturing num tokens (num_tokens=128 avail_mem=108.59 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.31it/s]Capturing num tokens (num_tokens=112 avail_mem=108.59 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.31it/s]Capturing num tokens (num_tokens=96 avail_mem=108.58 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.31it/s] Capturing num tokens (num_tokens=96 avail_mem=108.58 GB):  81%|████████  | 47/58 [00:03<00:00, 30.98it/s]Capturing num tokens (num_tokens=80 avail_mem=108.58 GB):  81%|████████  | 47/58 [00:03<00:00, 30.98it/s]

    Capturing num tokens (num_tokens=64 avail_mem=108.58 GB):  81%|████████  | 47/58 [00:03<00:00, 30.98it/s]Capturing num tokens (num_tokens=48 avail_mem=108.57 GB):  81%|████████  | 47/58 [00:03<00:00, 30.98it/s]Capturing num tokens (num_tokens=32 avail_mem=108.57 GB):  81%|████████  | 47/58 [00:03<00:00, 30.98it/s]Capturing num tokens (num_tokens=32 avail_mem=108.57 GB):  88%|████████▊ | 51/58 [00:03<00:00, 30.38it/s]Capturing num tokens (num_tokens=28 avail_mem=108.55 GB):  88%|████████▊ | 51/58 [00:03<00:00, 30.38it/s]Capturing num tokens (num_tokens=24 avail_mem=108.55 GB):  88%|████████▊ | 51/58 [00:03<00:00, 30.38it/s]

    Capturing num tokens (num_tokens=20 avail_mem=108.54 GB):  88%|████████▊ | 51/58 [00:03<00:00, 30.38it/s]Capturing num tokens (num_tokens=16 avail_mem=108.54 GB):  88%|████████▊ | 51/58 [00:03<00:00, 30.38it/s]Capturing num tokens (num_tokens=16 avail_mem=108.54 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.75it/s]Capturing num tokens (num_tokens=12 avail_mem=108.53 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.75it/s]Capturing num tokens (num_tokens=8 avail_mem=108.51 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.75it/s] Capturing num tokens (num_tokens=4 avail_mem=108.03 GB):  95%|█████████▍| 55/58 [00:04<00:00, 23.75it/s]

    Capturing num tokens (num_tokens=4 avail_mem=108.03 GB): 100%|██████████| 58/58 [00:04<00:00, 23.92it/s]Capturing num tokens (num_tokens=4 avail_mem=108.03 GB): 100%|██████████| 58/58 [00:04<00:00, 14.29it/s]


    [2026-04-26 08:04:15] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants to know the current date and time, along with the weather. Let me break this down.<br><br>First, I need to figure out which functions to use. The user mentioned two specific functions: 'get_current_weather' and 'get_current_date'. Both are available, so I'll use those.<br><br>Starting with the date and time. The function 'get_current_date' requires a timezone parameter. Since the user is in New York, I should use 'America/New_York' as the timezone. That should give me the exact current date and time in that location.<br><br>Next, for the weather. The function 'get_current_weather' needs a city, state, and unit. The user is in New York, so the city is 'New York'. The state abbreviation for New York is 'NY'. They probably want the temperature in Fahrenheit, so I'll set the unit to 'fahrenheit'.<br><br>I should structure the response by calling each function separately. First, the date and time, then the weather. Each function call will be on its own line with the appropriate parameters.<br><br>I also need to remember to include the source links in the response, as per the instructions. So, I'll add a note about the weather data coming from an example source and the date from another example source.<br><br>Putting it all together, I'll format each function call correctly, using the JSON structure provided. I'll make sure to specify each parameter clearly so the user understands how the information is retrieved.<br><br>Finally, I'll check that I'm only using the required parameters and that each function is called correctly without any extra information. That should cover everything the user asked for.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nWait, maybe I should check the latest statistics. I recall that in recent years, Paris has seen a slight increase. Let me think... I believe the population is approximately 2,150,000 as of 2023. That seems about right, but I\'m not 100% certain. I should make sure to present this information accurately.\n\nNext, I need to structure this into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" for context. The city is Paris, the population is 2,150,000, and the country is France.\n\nI should also consider if the user might need more details, like the exact year of the population figure. Including that could be helpful, so I\'ll add "year": 2023. That way, the user knows the data is up to date.\n\nPutting it all together, the JSON should look clean and well-structured. I\'ll make sure the syntax is correct, with proper commas and quotation marks. No markdown, just plain JSON.\n\nI think that\'s all. The user probably just needs the information quickly, so keeping it concise is key. I\'ll present the JSON without any extra fluff.\n</think>{\n\n"name": "Paris",\n"population": 21500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 14190, 11, 7196, 358, 1265, 1779, 279, 5535, 13142, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 8112, 5263, 13, 6771, 752, 1744, 1112, 358, 4411, 279, 7042, 374, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 3654, 13, 358, 1265, 1281, 2704, 311, 3042, 419, 1995, 29257, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 369, 2266, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 4734, 1042, 315, 279, 7042, 7071, 13, 55121, 429, 1410, 387, 10950, 11, 773, 358, 3278, 912, 330, 3157, 788, 220, 17, 15, 17, 18, 13, 2938, 1616, 11, 279, 1196, 8788, 279, 821, 374, 705, 311, 2400, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 4240, 323, 1632, 12, 51143, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 13, 2308, 50494, 11, 1101, 14396, 4718, 382, 40, 1744, 429, 594, 678, 13, 576, 1196, 4658, 1101, 3880, 279, 1995, 6157, 11, 773, 10282, 432, 63594, 374, 1376, 13, 358, 3278, 3042, 279, 4718, 2041, 894, 4960, 1320, 1362, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '2927634014304e0991731000c5c1302c', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 367, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.462876051664352, 'response_sent_to_client_ts': 1777190702.0828319}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should verify that.\n\nWait, maybe I should check some reliable sources to confirm the population. I recall that the population figures can vary depending on the source and the year. For example, the 2020 census might have a slightly different number than the 2021 estimate. I think the population was around 2,165,000 in 2021, but I\'m not 100% certain. I should make sure to use the most accurate and up-to-date information.\n\nAlso, the user wants the information in JSON format. JSON is a data interchange format, so I\'ll need to structure the data accordingly. I should include the city name, population, and maybe the year of the data. It\'s important to present the information clearly and accurately, so I\'ll double-check the numbers to avoid any mistakes.\n\nI should also consider if there are any other relevant details the user might find useful, like the area of the city or some key facts about it. But since the user specifically asked for population, I\'ll focus on that. Maybe adding a note about the population figure being approximate would be helpful, just in case.\n\nPutting it all together, I\'ll structure the JSON with the city name, population, and the year. I\'ll make sure the syntax is correct, using quotation marks and commas appropriately. I\'ll also keep the language clear and straightforward so that the user can easily understand the information.\n\nFinally, I\'ll review the JSON to ensure there are no errors and that the data is accurate. This way, the user gets a reliable and well-formatted response to their query.\n</think>{"name": "Paris", "population": 2165000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 14190, 11, 7196, 358, 1265, 1779, 1045, 14720, 8173, 311, 7683, 279, 7042, 13, 358, 19091, 429, 279, 7042, 12396, 646, 13289, 11649, 389, 279, 2530, 323, 279, 1042, 13, 1752, 3110, 11, 279, 220, 17, 15, 17, 15, 43602, 2578, 614, 264, 10078, 2155, 1372, 1091, 279, 220, 17, 15, 17, 16, 16045, 13, 358, 1744, 279, 7042, 572, 2163, 220, 17, 11, 16, 21, 20, 11, 15, 15, 15, 304, 220, 17, 15, 17, 16, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 3654, 13, 358, 1265, 1281, 2704, 311, 990, 279, 1429, 13382, 323, 705, 4686, 18413, 1995, 382, 13394, 11, 279, 1196, 6801, 279, 1995, 304, 4718, 3561, 13, 4718, 374, 264, 821, 51263, 3561, 11, 773, 358, 3278, 1184, 311, 5944, 279, 821, 27079, 13, 358, 1265, 2924, 279, 3283, 829, 11, 7042, 11, 323, 7196, 279, 1042, 315, 279, 821, 13, 1084, 594, 2989, 311, 3042, 279, 1995, 9355, 323, 29257, 11, 773, 358, 3278, 1990, 15934, 279, 5109, 311, 5648, 894, 20643, 382, 40, 1265, 1083, 2908, 421, 1052, 525, 894, 1008, 9760, 3565, 279, 1196, 2578, 1477, 5390, 11, 1075, 279, 3082, 315, 279, 3283, 476, 1045, 1376, 13064, 911, 432, 13, 1988, 2474, 279, 1196, 11689, 4588, 369, 7042, 11, 358, 3278, 5244, 389, 429, 13, 10696, 7842, 264, 5185, 911, 279, 7042, 7071, 1660, 44868, 1035, 387, 10950, 11, 1101, 304, 1142, 382, 97904, 432, 678, 3786, 11, 358, 3278, 5944, 279, 4718, 448, 279, 3283, 829, 11, 7042, 11, 323, 279, 1042, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 1667, 54231, 15423, 323, 76602, 34901, 13, 358, 3278, 1083, 2506, 279, 4128, 2797, 323, 30339, 773, 429, 279, 1196, 646, 6707, 3535, 279, 1995, 382, 23949, 11, 358, 3278, 3395, 279, 4718, 311, 5978, 1052, 525, 902, 5975, 323, 429, 279, 821, 374, 13382, 13, 1096, 1616, 11, 279, 1196, 5221, 264, 14720, 323, 1632, 8460, 12127, 2033, 311, 862, 3239, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 21, 20, 15, 15, 15, 92, 151643], 'meta_info': {'id': '0b8f9777137d4f2cb58484722a5b9b1d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 428, 'completion_tokens': 447, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.3530547991395, 'response_sent_to_client_ts': 1777190706.44795}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'a353e86d75bc4fd1aaa2f6b18d4b47e2', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.12018812354654074, 'response_sent_to_client_ts': 1777190706.595669}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '98cb4a5d2dde472c978cd0d087c12c21', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.12009833101183176, 'response_sent_to_client_ts': 1777190706.5956817}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '8571a6bee764493a8400757163b25a08', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11993600055575371, 'response_sent_to_client_ts': 1777190706.5956862}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'd33eb52c366c44a4aa35b8760970a433', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 20.780328059569, 'response_sent_to_client_ts': 1777190727.384122}}


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


<strong style='color: #00008B;'>{'text': 'Alright, I need to figure out how to respond to the user\'s query. They asked for the information and population of the capital of France in JSON format. So, first, I should identify what the capital of France is. I know that Paris is the capital. Next, I need the population. The latest data I have is from 2023, and the population is over 3.5 million. \n\nI\'ll structure this in JSON. The user wants a JSON object, so I\'ll create a key for Paris and a nested key for population, then include the value. I should make sure the formatting is correct with proper commas and quotes. \n\nWait, should I mention any other details like area or longitude? The user asked specifically for information and population, so maybe focusing on population as the key point. But just in case, including area and longitude might make the information more complete. \n\nAlternatively, the user might just want the population number. Since the query specifies both "information" and "population," it\'s safer to include more relevant data. So I\'ll add a key for population with the number, and perhaps a note about the approximate figure. \n\nI should also handle the population number correctly. Since the number is over three million, it\'s better to write it with commas for readability in JSON, so 3,573,274. That makes it easier to read and understand in the format. \n\nIs there anything else I should consider? Maybe the data source, but the user didn\'t ask for that, so I can omit it. Just providing the JSON with the necessary fields should suffice. \n\nSo, putting it all together, the JSON object will have a "city" key pointing to "Paris," a "population" key with the approximate number, and maybe a "notes" key explaining the figure\'s approximation. Alternatively, it can have "info" and "population" as keys with the respective details. \n\nI think the example I provided covers the necessary information in the correct JSON format. I\'ll go with the second structure I thought of, placing population as a separate key within a "population" object for clarity. That should meet the user\'s requirements effectively.\n</think>\n\nHere is the information and population of the capital of France in JSON format:\n\n```json\n{\n  "city": "Paris",\n  "population": {\n    "value": 3573274,\n    "note": "The population figure is approximate and may vary slightly depending on the source."\n  }\n}\n```', 'output_ids': [71486, 11, 358, 1184, 311, 7071, 700, 1246, 311, 5889, 311, 279, 1196, 594, 3239, 13, 2379, 4588, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 2055, 11, 1156, 11, 358, 1265, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 13, 9295, 11, 358, 1184, 279, 7042, 13, 576, 5535, 821, 358, 614, 374, 504, 220, 17, 15, 17, 18, 11, 323, 279, 7042, 374, 916, 220, 18, 13, 20, 3526, 13, 4710, 40, 3278, 5944, 419, 304, 4718, 13, 576, 1196, 6801, 264, 4718, 1633, 11, 773, 358, 3278, 1855, 264, 1376, 369, 12095, 323, 264, 24034, 1376, 369, 7042, 11, 1221, 2924, 279, 897, 13, 358, 1265, 1281, 2704, 279, 36566, 374, 4396, 448, 6169, 76602, 323, 17194, 13, 4710, 14190, 11, 1265, 358, 6286, 894, 1008, 3565, 1075, 3082, 476, 20515, 30, 576, 1196, 4588, 11689, 369, 1995, 323, 7042, 11, 773, 7196, 21080, 389, 7042, 438, 279, 1376, 1459, 13, 1988, 1101, 304, 1142, 11, 2670, 3082, 323, 20515, 2578, 1281, 279, 1995, 803, 4583, 13, 4710, 92014, 11, 279, 1196, 2578, 1101, 1366, 279, 7042, 1372, 13, 8704, 279, 3239, 29102, 2176, 330, 25069, 1, 323, 330, 44441, 1335, 432, 594, 29449, 311, 2924, 803, 9760, 821, 13, 2055, 358, 3278, 912, 264, 1376, 369, 7042, 448, 279, 1372, 11, 323, 8365, 264, 5185, 911, 279, 44868, 7071, 13, 4710, 40, 1265, 1083, 3705, 279, 7042, 1372, 12440, 13, 8704, 279, 1372, 374, 916, 2326, 3526, 11, 432, 594, 2664, 311, 3270, 432, 448, 76602, 369, 91494, 304, 4718, 11, 773, 220, 18, 11, 20, 22, 18, 11, 17, 22, 19, 13, 2938, 3643, 432, 8661, 311, 1349, 323, 3535, 304, 279, 3561, 13, 4710, 3872, 1052, 4113, 770, 358, 1265, 2908, 30, 10696, 279, 821, 2530, 11, 714, 279, 1196, 3207, 944, 2548, 369, 429, 11, 773, 358, 646, 51044, 432, 13, 4599, 8241, 279, 4718, 448, 279, 5871, 5043, 1265, 76156, 13, 4710, 4416, 11, 10687, 432, 678, 3786, 11, 279, 4718, 1633, 686, 614, 264, 330, 8926, 1, 1376, 21633, 311, 330, 59604, 1335, 264, 330, 44441, 1, 1376, 448, 279, 44868, 1372, 11, 323, 7196, 264, 330, 18286, 1, 1376, 25021, 279, 7071, 594, 56204, 13, 38478, 11, 432, 646, 614, 330, 2733, 1, 323, 330, 44441, 1, 438, 6894, 448, 279, 19511, 3565, 13, 4710, 40, 1744, 279, 3110, 358, 3897, 14521, 279, 5871, 1995, 304, 279, 4396, 4718, 3561, 13, 358, 3278, 728, 448, 279, 2086, 5944, 358, 3381, 315, 11, 24047, 7042, 438, 264, 8651, 1376, 2878, 264, 330, 44441, 1, 1633, 369, 31273, 13, 2938, 1265, 3367, 279, 1196, 594, 8502, 13444, 624, 151649, 271, 8420, 374, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 8926, 788, 330, 59604, 756, 220, 330, 44441, 788, 341, 262, 330, 957, 788, 220, 18, 20, 22, 18, 17, 22, 19, 345, 262, 330, 9974, 788, 330, 785, 7042, 7071, 374, 44868, 323, 1231, 13289, 10078, 11649, 389, 279, 2530, 10040, 220, 456, 532, 73594, 151643], 'meta_info': {'id': '9190d9056413426e94127a51c4263f5f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 450, 'completion_tokens': 519, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.288927207700908, 'response_sent_to_client_ts': 1777190732.688659}}</strong>



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
    [2026-04-26 08:05:42] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:00<00:00,  1.05it/s]

    Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.04it/s]Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.04it/s]


    2026-04-26 08:05:50,290 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 08:05:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:05,  5.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:05,  5.37s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:10,  2.33s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:10,  2.33s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:14,  1.35s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:14,  1.35s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:33,  1.59it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:24,  2.14it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:24,  2.14it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.77it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.77it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.51it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.51it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.37it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.37it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.37it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.05it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.05it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.05it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.50it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.50it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:05,  7.50it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  8.86it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  8.86it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  8.86it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.54it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.54it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.54it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 10.54it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 13.48it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 13.48it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 13.48it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 13.48it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 13.48it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 18.29it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 18.29it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 18.29it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 18.29it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 18.29it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 18.29it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 18.29it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:07<00:01, 26.99it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:07<00:00, 37.12it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:07<00:00, 37.12it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:07<00:00, 37.12it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:07<00:00, 37.12it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:07<00:00, 37.12it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:07<00:00, 37.12it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:07<00:00, 37.12it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:07<00:00, 37.12it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:07<00:00, 37.12it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:07<00:00, 46.64it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:07<00:00, 46.64it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:07<00:00, 46.64it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:07<00:00, 46.64it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:07<00:00, 46.64it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:07<00:00, 46.64it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:07<00:00, 46.64it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:07<00:00, 46.64it/s]

    Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:07<00:00, 46.64it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:07<00:00, 53.28it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:07<00:00, 53.28it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:07<00:00, 53.28it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:07<00:00, 53.28it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:08<00:00, 53.28it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:08<00:00, 53.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=90.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=90.24 GB):   2%|▏         | 1/58 [00:00<00:16,  3.43it/s]Capturing num tokens (num_tokens=7680 avail_mem=90.21 GB):   2%|▏         | 1/58 [00:00<00:16,  3.43it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=90.21 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=90.21 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=90.21 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=90.21 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=90.21 GB):   7%|▋         | 4/58 [00:01<00:13,  4.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=90.21 GB):   7%|▋         | 4/58 [00:01<00:13,  4.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=90.21 GB):   9%|▊         | 5/58 [00:01<00:12,  4.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=90.21 GB):   9%|▊         | 5/58 [00:01<00:12,  4.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=90.21 GB):  10%|█         | 6/58 [00:01<00:11,  4.71it/s]Capturing num tokens (num_tokens=5120 avail_mem=90.20 GB):  10%|█         | 6/58 [00:01<00:11,  4.71it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=90.20 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=90.21 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=90.21 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.55it/s]Capturing num tokens (num_tokens=4096 avail_mem=88.85 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.55it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=88.85 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=88.78 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=88.78 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.23it/s]Capturing num tokens (num_tokens=3584 avail_mem=85.80 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.23it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=85.80 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=83.01 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=83.01 GB):  21%|██        | 12/58 [00:02<00:06,  7.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=83.01 GB):  21%|██        | 12/58 [00:02<00:06,  7.29it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=83.01 GB):  21%|██        | 12/58 [00:02<00:06,  7.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=83.01 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.61it/s]Capturing num tokens (num_tokens=2560 avail_mem=83.01 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=83.00 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.61it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=83.00 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=83.00 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=83.00 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=83.00 GB):  31%|███       | 18/58 [00:02<00:03, 11.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=82.99 GB):  31%|███       | 18/58 [00:02<00:03, 11.66it/s]Capturing num tokens (num_tokens=1280 avail_mem=82.99 GB):  31%|███       | 18/58 [00:02<00:03, 11.66it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=82.98 GB):  31%|███       | 18/58 [00:02<00:03, 11.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=82.98 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s]Capturing num tokens (num_tokens=960 avail_mem=82.98 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s] Capturing num tokens (num_tokens=896 avail_mem=82.97 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s]Capturing num tokens (num_tokens=832 avail_mem=82.97 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s]Capturing num tokens (num_tokens=832 avail_mem=82.97 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.74it/s]Capturing num tokens (num_tokens=768 avail_mem=82.97 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.74it/s]Capturing num tokens (num_tokens=704 avail_mem=82.96 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.74it/s]

    Capturing num tokens (num_tokens=640 avail_mem=82.96 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.74it/s]Capturing num tokens (num_tokens=576 avail_mem=82.96 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.74it/s]Capturing num tokens (num_tokens=576 avail_mem=82.96 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.51it/s]Capturing num tokens (num_tokens=512 avail_mem=82.95 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.51it/s]Capturing num tokens (num_tokens=480 avail_mem=82.95 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.51it/s]Capturing num tokens (num_tokens=448 avail_mem=82.95 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.51it/s]Capturing num tokens (num_tokens=416 avail_mem=82.94 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.51it/s]Capturing num tokens (num_tokens=416 avail_mem=82.94 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.17it/s]Capturing num tokens (num_tokens=384 avail_mem=82.94 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.17it/s]

    Capturing num tokens (num_tokens=352 avail_mem=82.93 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.17it/s]Capturing num tokens (num_tokens=320 avail_mem=82.93 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.17it/s]Capturing num tokens (num_tokens=288 avail_mem=82.94 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.17it/s]Capturing num tokens (num_tokens=288 avail_mem=82.94 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.22it/s]Capturing num tokens (num_tokens=256 avail_mem=82.93 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.22it/s]Capturing num tokens (num_tokens=240 avail_mem=82.93 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.22it/s]Capturing num tokens (num_tokens=224 avail_mem=82.92 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.22it/s]Capturing num tokens (num_tokens=208 avail_mem=82.92 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.22it/s]Capturing num tokens (num_tokens=208 avail_mem=82.92 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.69it/s]Capturing num tokens (num_tokens=192 avail_mem=82.92 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.69it/s]

    Capturing num tokens (num_tokens=176 avail_mem=82.91 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.69it/s]Capturing num tokens (num_tokens=160 avail_mem=82.91 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.69it/s]Capturing num tokens (num_tokens=144 avail_mem=82.90 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.69it/s]Capturing num tokens (num_tokens=144 avail_mem=82.90 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.48it/s]Capturing num tokens (num_tokens=128 avail_mem=82.91 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.48it/s]Capturing num tokens (num_tokens=112 avail_mem=82.90 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.48it/s]Capturing num tokens (num_tokens=96 avail_mem=82.90 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.48it/s] Capturing num tokens (num_tokens=80 avail_mem=82.89 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.48it/s]Capturing num tokens (num_tokens=80 avail_mem=82.89 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.26it/s]Capturing num tokens (num_tokens=64 avail_mem=82.89 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.26it/s]

    Capturing num tokens (num_tokens=48 avail_mem=82.89 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.26it/s]Capturing num tokens (num_tokens=32 avail_mem=82.88 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.26it/s]Capturing num tokens (num_tokens=28 avail_mem=82.88 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.26it/s]Capturing num tokens (num_tokens=28 avail_mem=82.88 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.42it/s]Capturing num tokens (num_tokens=24 avail_mem=82.88 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.42it/s]Capturing num tokens (num_tokens=20 avail_mem=82.87 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.42it/s]Capturing num tokens (num_tokens=16 avail_mem=82.87 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.42it/s]Capturing num tokens (num_tokens=12 avail_mem=82.87 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.42it/s]Capturing num tokens (num_tokens=12 avail_mem=82.87 GB):  97%|█████████▋| 56/58 [00:03<00:00, 35.10it/s]Capturing num tokens (num_tokens=8 avail_mem=82.86 GB):  97%|█████████▋| 56/58 [00:03<00:00, 35.10it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=82.86 GB):  97%|█████████▋| 56/58 [00:03<00:00, 35.10it/s]Capturing num tokens (num_tokens=4 avail_mem=82.86 GB): 100%|██████████| 58/58 [00:03<00:00, 14.88it/s]


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
    
    Generated text: Alright, the user is asking for the information and population of the capital of France in JSON format. First, I need to identify the capital of France, which is Paris. Then, I should gather the current population data. As of 2023, Paris has a population around 12 million. I should note that population figures can change over time, so it's good to mention that it's an approximate figure. 
    
    I also want to make sure the JSON structure is correct. It should include "country" as "France", "capital" as "Paris", and "population" as the number. I'll format it properly, using commas correctly and ensuring the keys are in quotes. Since the user requested JSON, I need to avoid any markdown formatting beyond the JSON itself.
    
    I should keep the response concise and clear, making sure it's easy to understand. Maybe add a brief explanation that the population is approximate. That way, the user knows there might be slight variations depending on the source or the year. 
    
    Let me double-check the population numbers to ensure accuracy. Yes, Paris is the largest city in France and around 12 million makes sense. I think that's all. Time to put it all together in the JSON format as the user requested.
    </think>
    
    ```json
    {
      "country": "France",
      "capital": "Paris",
      "population": "Approximately 12 million (2023)"
    }
    ```



```python
llm.shutdown()
```
