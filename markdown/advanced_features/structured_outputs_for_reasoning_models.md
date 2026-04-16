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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-16 03:08:25] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-16 03:08:31] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-16 03:08:32] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [2026-04-16 03:08:32] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.27s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.40s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.38s/it]


    2026-04-16 03:08:40,230 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 03:08:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:56,  3.09s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:56,  3.09s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:30,  1.61s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:30,  1.61s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:54,  1.01it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:54,  1.01it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:37,  1.44it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:37,  1.44it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:27,  1.93it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:27,  1.93it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:20,  2.49it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:20,  2.49it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:16,  3.09it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:16,  3.09it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:13,  3.71it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:13,  3.71it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:11,  4.43it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:11,  4.43it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:09,  5.22it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:09,  5.22it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:07,  5.96it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:07,  5.96it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:06,  6.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:06,  6.71it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:06,  6.71it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:05,  8.24it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:05,  8.24it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:05,  8.24it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:04,  9.81it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:04,  9.81it/s]

    Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:04,  9.81it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 11.60it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 11.60it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 11.60it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 11.60it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 15.00it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 15.00it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 15.00it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 15.00it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 15.00it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 15.00it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:06<00:01, 22.13it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:06<00:01, 22.13it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:06<00:01, 22.13it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:06<00:01, 22.13it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:06<00:01, 22.13it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:06<00:01, 22.13it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:06<00:00, 28.84it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:06<00:00, 28.84it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:06<00:00, 28.84it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:06<00:00, 28.84it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:06<00:00, 28.84it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:06<00:00, 28.84it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:06<00:00, 28.84it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:00, 36.82it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:00, 36.82it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:00, 36.82it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:00, 36.82it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:00, 36.82it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:06<00:00, 36.82it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:06<00:00, 36.82it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:06<00:00, 36.82it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 44.12it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 44.12it/s]

    Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 44.12it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 44.12it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:06<00:00, 44.12it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:06<00:00, 44.12it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:06<00:00, 44.12it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:06<00:00, 44.12it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 49.30it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 49.30it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 49.30it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 49.30it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 49.30it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 49.30it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 49.30it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 49.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=101.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=101.20 GB):   2%|▏         | 1/58 [00:00<00:17,  3.25it/s]Capturing num tokens (num_tokens=7680 avail_mem=101.16 GB):   2%|▏         | 1/58 [00:00<00:17,  3.25it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=101.16 GB):   3%|▎         | 2/58 [00:00<00:16,  3.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=101.16 GB):   3%|▎         | 2/58 [00:00<00:16,  3.47it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=101.16 GB):   5%|▌         | 3/58 [00:00<00:14,  3.71it/s]Capturing num tokens (num_tokens=6656 avail_mem=101.17 GB):   5%|▌         | 3/58 [00:00<00:14,  3.71it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=101.17 GB):   7%|▋         | 4/58 [00:01<00:13,  4.01it/s]Capturing num tokens (num_tokens=6144 avail_mem=101.17 GB):   7%|▋         | 4/58 [00:01<00:13,  4.01it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=101.17 GB):   9%|▊         | 5/58 [00:01<00:12,  4.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=101.18 GB):   9%|▊         | 5/58 [00:01<00:12,  4.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=101.18 GB):  10%|█         | 6/58 [00:01<00:11,  4.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=101.18 GB):  10%|█         | 6/58 [00:01<00:11,  4.66it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=101.18 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=101.18 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=101.18 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=101.19 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.53it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=101.19 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=101.19 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=101.19 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=101.19 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.49it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=101.19 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=101.19 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=101.19 GB):  21%|██        | 12/58 [00:02<00:06,  7.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=101.19 GB):  21%|██        | 12/58 [00:02<00:06,  7.58it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=101.19 GB):  21%|██        | 12/58 [00:02<00:06,  7.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=101.19 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.72it/s]Capturing num tokens (num_tokens=2560 avail_mem=101.19 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.72it/s]Capturing num tokens (num_tokens=2560 avail_mem=101.19 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=101.19 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.84it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=101.18 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.84it/s]Capturing num tokens (num_tokens=2048 avail_mem=101.18 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=100.19 GB):  29%|██▉       | 17/58 [00:02<00:04,  9.01it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=100.19 GB):  31%|███       | 18/58 [00:02<00:05,  7.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=100.14 GB):  31%|███       | 18/58 [00:02<00:05,  7.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=100.14 GB):  33%|███▎      | 19/58 [00:03<00:05,  7.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=100.15 GB):  33%|███▎      | 19/58 [00:03<00:05,  7.24it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=100.15 GB):  34%|███▍      | 20/58 [00:03<00:05,  7.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=100.15 GB):  34%|███▍      | 20/58 [00:03<00:05,  7.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=100.15 GB):  36%|███▌      | 21/58 [00:03<00:05,  7.36it/s]Capturing num tokens (num_tokens=960 avail_mem=100.14 GB):  36%|███▌      | 21/58 [00:03<00:05,  7.36it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=100.14 GB):  38%|███▊      | 22/58 [00:03<00:04,  7.32it/s]Capturing num tokens (num_tokens=896 avail_mem=100.14 GB):  38%|███▊      | 22/58 [00:03<00:04,  7.32it/s]Capturing num tokens (num_tokens=896 avail_mem=100.14 GB):  40%|███▉      | 23/58 [00:03<00:04,  7.40it/s]Capturing num tokens (num_tokens=832 avail_mem=100.14 GB):  40%|███▉      | 23/58 [00:03<00:04,  7.40it/s]

    Capturing num tokens (num_tokens=832 avail_mem=100.14 GB):  41%|████▏     | 24/58 [00:03<00:04,  7.67it/s]Capturing num tokens (num_tokens=768 avail_mem=100.13 GB):  41%|████▏     | 24/58 [00:03<00:04,  7.67it/s]Capturing num tokens (num_tokens=768 avail_mem=100.13 GB):  43%|████▎     | 25/58 [00:03<00:04,  7.80it/s]Capturing num tokens (num_tokens=704 avail_mem=100.13 GB):  43%|████▎     | 25/58 [00:03<00:04,  7.80it/s]

    Capturing num tokens (num_tokens=704 avail_mem=100.13 GB):  45%|████▍     | 26/58 [00:03<00:03,  8.12it/s]Capturing num tokens (num_tokens=640 avail_mem=100.12 GB):  45%|████▍     | 26/58 [00:03<00:03,  8.12it/s]Capturing num tokens (num_tokens=640 avail_mem=100.12 GB):  47%|████▋     | 27/58 [00:04<00:03,  8.23it/s]Capturing num tokens (num_tokens=576 avail_mem=100.12 GB):  47%|████▋     | 27/58 [00:04<00:03,  8.23it/s]

    Capturing num tokens (num_tokens=576 avail_mem=100.12 GB):  48%|████▊     | 28/58 [00:04<00:04,  6.95it/s]Capturing num tokens (num_tokens=512 avail_mem=100.11 GB):  48%|████▊     | 28/58 [00:04<00:04,  6.95it/s]

    Capturing num tokens (num_tokens=512 avail_mem=100.11 GB):  50%|█████     | 29/58 [00:04<00:06,  4.16it/s]Capturing num tokens (num_tokens=480 avail_mem=100.11 GB):  50%|█████     | 29/58 [00:04<00:06,  4.16it/s]Capturing num tokens (num_tokens=480 avail_mem=100.11 GB):  52%|█████▏    | 30/58 [00:04<00:05,  4.96it/s]Capturing num tokens (num_tokens=448 avail_mem=100.11 GB):  52%|█████▏    | 30/58 [00:04<00:05,  4.96it/s]

    Capturing num tokens (num_tokens=448 avail_mem=100.11 GB):  53%|█████▎    | 31/58 [00:05<00:04,  5.52it/s]Capturing num tokens (num_tokens=416 avail_mem=100.11 GB):  53%|█████▎    | 31/58 [00:05<00:04,  5.52it/s]

    Capturing num tokens (num_tokens=416 avail_mem=100.11 GB):  55%|█████▌    | 32/58 [00:05<00:05,  5.14it/s]Capturing num tokens (num_tokens=384 avail_mem=100.10 GB):  55%|█████▌    | 32/58 [00:05<00:05,  5.14it/s]Capturing num tokens (num_tokens=384 avail_mem=100.10 GB):  57%|█████▋    | 33/58 [00:05<00:04,  5.20it/s]Capturing num tokens (num_tokens=352 avail_mem=100.10 GB):  57%|█████▋    | 33/58 [00:05<00:04,  5.20it/s]

    Capturing num tokens (num_tokens=352 avail_mem=100.10 GB):  59%|█████▊    | 34/58 [00:05<00:04,  5.62it/s]Capturing num tokens (num_tokens=320 avail_mem=100.09 GB):  59%|█████▊    | 34/58 [00:05<00:04,  5.62it/s]Capturing num tokens (num_tokens=320 avail_mem=100.09 GB):  60%|██████    | 35/58 [00:05<00:04,  5.69it/s]Capturing num tokens (num_tokens=288 avail_mem=100.09 GB):  60%|██████    | 35/58 [00:05<00:04,  5.69it/s]

    Capturing num tokens (num_tokens=256 avail_mem=100.09 GB):  60%|██████    | 35/58 [00:05<00:04,  5.69it/s]Capturing num tokens (num_tokens=256 avail_mem=100.09 GB):  64%|██████▍   | 37/58 [00:05<00:02,  7.40it/s]Capturing num tokens (num_tokens=240 avail_mem=100.08 GB):  64%|██████▍   | 37/58 [00:05<00:02,  7.40it/s]Capturing num tokens (num_tokens=224 avail_mem=100.08 GB):  64%|██████▍   | 37/58 [00:05<00:02,  7.40it/s]

    Capturing num tokens (num_tokens=224 avail_mem=100.08 GB):  67%|██████▋   | 39/58 [00:06<00:02,  9.49it/s]Capturing num tokens (num_tokens=208 avail_mem=100.07 GB):  67%|██████▋   | 39/58 [00:06<00:02,  9.49it/s]Capturing num tokens (num_tokens=192 avail_mem=100.07 GB):  67%|██████▋   | 39/58 [00:06<00:02,  9.49it/s]Capturing num tokens (num_tokens=192 avail_mem=100.07 GB):  71%|███████   | 41/58 [00:06<00:01, 11.71it/s]Capturing num tokens (num_tokens=176 avail_mem=100.07 GB):  71%|███████   | 41/58 [00:06<00:01, 11.71it/s]Capturing num tokens (num_tokens=160 avail_mem=100.07 GB):  71%|███████   | 41/58 [00:06<00:01, 11.71it/s]Capturing num tokens (num_tokens=144 avail_mem=100.06 GB):  71%|███████   | 41/58 [00:06<00:01, 11.71it/s]

    Capturing num tokens (num_tokens=144 avail_mem=100.06 GB):  76%|███████▌  | 44/58 [00:06<00:00, 15.15it/s]Capturing num tokens (num_tokens=128 avail_mem=100.07 GB):  76%|███████▌  | 44/58 [00:06<00:00, 15.15it/s]Capturing num tokens (num_tokens=112 avail_mem=100.06 GB):  76%|███████▌  | 44/58 [00:06<00:00, 15.15it/s]Capturing num tokens (num_tokens=96 avail_mem=100.06 GB):  76%|███████▌  | 44/58 [00:06<00:00, 15.15it/s] Capturing num tokens (num_tokens=96 avail_mem=100.06 GB):  81%|████████  | 47/58 [00:06<00:00, 18.45it/s]Capturing num tokens (num_tokens=80 avail_mem=100.05 GB):  81%|████████  | 47/58 [00:06<00:00, 18.45it/s]Capturing num tokens (num_tokens=64 avail_mem=100.05 GB):  81%|████████  | 47/58 [00:06<00:00, 18.45it/s]Capturing num tokens (num_tokens=48 avail_mem=100.05 GB):  81%|████████  | 47/58 [00:06<00:00, 18.45it/s]

    Capturing num tokens (num_tokens=48 avail_mem=100.05 GB):  86%|████████▌ | 50/58 [00:06<00:00, 21.21it/s]Capturing num tokens (num_tokens=32 avail_mem=100.04 GB):  86%|████████▌ | 50/58 [00:06<00:00, 21.21it/s]Capturing num tokens (num_tokens=28 avail_mem=100.04 GB):  86%|████████▌ | 50/58 [00:06<00:00, 21.21it/s]Capturing num tokens (num_tokens=24 avail_mem=100.04 GB):  86%|████████▌ | 50/58 [00:06<00:00, 21.21it/s]Capturing num tokens (num_tokens=20 avail_mem=100.03 GB):  86%|████████▌ | 50/58 [00:06<00:00, 21.21it/s]Capturing num tokens (num_tokens=20 avail_mem=100.03 GB):  93%|█████████▎| 54/58 [00:06<00:00, 24.81it/s]Capturing num tokens (num_tokens=16 avail_mem=100.03 GB):  93%|█████████▎| 54/58 [00:06<00:00, 24.81it/s]Capturing num tokens (num_tokens=12 avail_mem=100.01 GB):  93%|█████████▎| 54/58 [00:06<00:00, 24.81it/s]Capturing num tokens (num_tokens=8 avail_mem=100.00 GB):  93%|█████████▎| 54/58 [00:06<00:00, 24.81it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=100.00 GB):  93%|█████████▎| 54/58 [00:06<00:00, 24.81it/s]Capturing num tokens (num_tokens=4 avail_mem=100.00 GB): 100%|██████████| 58/58 [00:06<00:00, 27.54it/s]Capturing num tokens (num_tokens=4 avail_mem=100.00 GB): 100%|██████████| 58/58 [00:06<00:00,  8.64it/s]


    [2026-04-16 03:08:55] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants to know the current date and time along with the weather. Let me break down how I should approach this.<br><br>First, I need to figure out which functions to use. The user mentioned they're in New York, so I should probably get the current date and time for that location. Looking at the available functions, there's 'get_current_date' which requires a timezone parameter. Since New York is in the Eastern Timezone, I think the correct abbreviation is 'EST'. So, I'll call get_current_date with 'EST' as the timezone.<br><br>Next, for the weather, the user specifically asked about New York. I should use 'get_current_weather' here. The function needs a city and state. New York's city is 'New York' and the state is 'NY'. Also, the temperature unit isn't specified, so I'll default to Celsius.<br><br>Now, I need to structure the function calls properly. I'll start with get_current_date, include the timezone as 'EST' in a JSON object. Then, I'll call get_current_weather with city 'New York' and state 'NY'. I'll make sure each function call is on a separate line and follows the required format, including the start_tag and end_tag correctly.<br><br>I should also remember to include the sources in the response. The date and time will come from the get_current_date function, and the weather from get_current_weather. I'll format everything neatly so the user can easily read the results.<br><br><br>content: <function=get_current_date>{"timezone": "EST"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "celsius"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nWait, maybe I should check the latest statistics. I recall that in recent years, Paris has seen a slight increase. Let me think... I believe the population is approximately 2,150,000 as of 2023. That seems about right, but I\'m not 100% certain. I should make sure to present this information accurately.\n\nNext, I need to structure this into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" for context. The city is Paris, the population is 2,150,000, and the country is France.\n\nI should also consider if the user might need more details, like the exact year of the population figure. Including that could be helpful, so I\'ll add "year": 2023. That way, the user knows the data is up to date.\n\nPutting it all together, the JSON should look clean and well-structured. I\'ll make sure the syntax is correct, with proper commas and quotation marks. No markdown, just plain JSON.\n\nI think that\'s all. The user probably just needs the information quickly, so keeping it concise is key. I\'ll present the JSON without any extra fluff.\n</think>{\n\n"name": "Paris",\n"population": 21500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 14190, 11, 7196, 358, 1265, 1779, 279, 5535, 13142, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 8112, 5263, 13, 6771, 752, 1744, 1112, 358, 4411, 279, 7042, 374, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 3654, 13, 358, 1265, 1281, 2704, 311, 3042, 419, 1995, 29257, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 369, 2266, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 4734, 1042, 315, 279, 7042, 7071, 13, 55121, 429, 1410, 387, 10950, 11, 773, 358, 3278, 912, 330, 3157, 788, 220, 17, 15, 17, 18, 13, 2938, 1616, 11, 279, 1196, 8788, 279, 821, 374, 705, 311, 2400, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 4240, 323, 1632, 12, 51143, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 13, 2308, 50494, 11, 1101, 14396, 4718, 382, 40, 1744, 429, 594, 678, 13, 576, 1196, 4658, 1101, 3880, 279, 1995, 6157, 11, 773, 10282, 432, 63594, 374, 1376, 13, 358, 3278, 3042, 279, 4718, 2041, 894, 4960, 1320, 1362, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '7254cbc242af4118ad6b4c692486102b', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 367, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 21.667006109841168, 'response_sent_to_client_ts': 1776308984.8426285}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should verify that.\n\nWait, maybe I should check some reliable sources to confirm the population. I recall that the population figures can vary depending on the source and the year. For example, the 2020 census might have a slightly different number than the 2021 estimate. I think the population was around 2,165,000 in 2021, but I\'m not 100% certain. I should make sure to use the most accurate and up-to-date information.\n\nAlso, the user wants the information in JSON format. JSON is a data interchange format, so I\'ll need to structure the data accordingly. I should include the city name, population, and maybe the year of the data. It\'s important to present the information clearly and accurately, so I\'ll double-check the numbers to avoid any mistakes.\n\nI should also consider if there are any other relevant details the user might find useful, like the area of the city or some key facts about it. But since the user specifically asked for population, I\'ll focus on that. Maybe adding a note about the population figure being approximate would be helpful, just in case.\n\nPutting it all together, I\'ll structure the JSON with the city name, population, and the year. I\'ll make sure the syntax is correct, using quotation marks and commas appropriately. I\'ll also keep the language clear and straightforward so that the user can easily understand the information.\n\nFinally, I\'ll review the JSON to ensure there are no errors and that the data is accurate. This way, the user gets a reliable and well-formatted response to their query.\n</think>{"name": "Paris", "population": 2165000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 14190, 11, 7196, 358, 1265, 1779, 1045, 14720, 8173, 311, 7683, 279, 7042, 13, 358, 19091, 429, 279, 7042, 12396, 646, 13289, 11649, 389, 279, 2530, 323, 279, 1042, 13, 1752, 3110, 11, 279, 220, 17, 15, 17, 15, 43602, 2578, 614, 264, 10078, 2155, 1372, 1091, 279, 220, 17, 15, 17, 16, 16045, 13, 358, 1744, 279, 7042, 572, 2163, 220, 17, 11, 16, 21, 20, 11, 15, 15, 15, 304, 220, 17, 15, 17, 16, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 3654, 13, 358, 1265, 1281, 2704, 311, 990, 279, 1429, 13382, 323, 705, 4686, 18413, 1995, 382, 13394, 11, 279, 1196, 6801, 279, 1995, 304, 4718, 3561, 13, 4718, 374, 264, 821, 51263, 3561, 11, 773, 358, 3278, 1184, 311, 5944, 279, 821, 27079, 13, 358, 1265, 2924, 279, 3283, 829, 11, 7042, 11, 323, 7196, 279, 1042, 315, 279, 821, 13, 1084, 594, 2989, 311, 3042, 279, 1995, 9355, 323, 29257, 11, 773, 358, 3278, 1990, 15934, 279, 5109, 311, 5648, 894, 20643, 382, 40, 1265, 1083, 2908, 421, 1052, 525, 894, 1008, 9760, 3565, 279, 1196, 2578, 1477, 5390, 11, 1075, 279, 3082, 315, 279, 3283, 476, 1045, 1376, 13064, 911, 432, 13, 1988, 2474, 279, 1196, 11689, 4588, 369, 7042, 11, 358, 3278, 5244, 389, 429, 13, 10696, 7842, 264, 5185, 911, 279, 7042, 7071, 1660, 44868, 1035, 387, 10950, 11, 1101, 304, 1142, 382, 97904, 432, 678, 3786, 11, 358, 3278, 5944, 279, 4718, 448, 279, 3283, 829, 11, 7042, 11, 323, 279, 1042, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 1667, 54231, 15423, 323, 76602, 34901, 13, 358, 3278, 1083, 2506, 279, 4128, 2797, 323, 30339, 773, 429, 279, 1196, 646, 6707, 3535, 279, 1995, 382, 23949, 11, 358, 3278, 3395, 279, 4718, 311, 5978, 1052, 525, 902, 5975, 323, 429, 279, 821, 374, 13382, 13, 1096, 1616, 11, 279, 1196, 5221, 264, 14720, 323, 1632, 8460, 12127, 2033, 311, 862, 3239, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 21, 20, 15, 15, 15, 92, 151643], 'meta_info': {'id': '1f059b76f6b94135b57f0e68ecc0caeb', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 428, 'completion_tokens': 447, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.708787771873176, 'response_sent_to_client_ts': 1776308989.5641098}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'cb7981540d894b2495c0d9fceda57fd2', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14024805556982756, 'response_sent_to_client_ts': 1776308989.7497036}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '53d81a09bea74db4ab54d36cdb4fc924', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1400769893079996, 'response_sent_to_client_ts': 1776308989.749719}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '06a9e621d28943fbae668edc4ab6c434', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14001236483454704, 'response_sent_to_client_ts': 1776308989.7497237}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '5af0365bfd6d4dddb75a51e28f1d0fc7', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.77198890130967, 'response_sent_to_client_ts': 1776309009.5293112}}


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


<strong style='color: #00008B;'>{'text': 'Alright, the user asked for the information and population of the capital of France in the JSON format. The capital is Paris, obviously.\n\nI need to find the most recent population data. I remember that the population of Paris has been increasing steadily. In 2020, it was around 2.16 million people. I should verify that date to make sure it\'s accurate and up-to-date.\n\nNext, I need to include some basic info about Paris. Its administrative divisions, like the number of departments, are 20. The prefecture is Seine-et-Marne, which is an important administrative area. The metropolitan area of Paris is much larger, around 12.5 million people, which is good to note since sometimes people think of Paris as a metro area when considering its population.\n\nI should structure this information in a JSON format. I\'ll create a key-value pairs where "nom" is the name, "population" is the current estimate, "info" includes other details like departments, prefecture, and metropolitan area, and "notes" can have any relevant remarks, but in this case, it\'s straightforward.\n\nI should make sure the JSON is properly formatted with commas and brackets. No trailing commas to avoid errors. Also, the population figure should be accurate and clearly stated. I don’t want to leave any room for confusion, so providing a note that it\'s an estimate is a good idea.\n\nPutting it all together, the JSON should be clean, easy to read, and contain all the necessary information without any missing data. That should meet the user\'s request effectively.\n</think>\n\n```json\n{\n  "nom": "Paris",\n  "population": 2160000,\n  "info": {\n    "foo": {\n      "nom_de departements": 20,\n      "nom_de prefecture": "Saint-Denis",\n      "massue": "Île-de-France"\n    }\n  },\n  "notes": "Les informations sont valables pour l\'année 2023."\n}\n```', 'output_ids': [71486, 11, 279, 1196, 4588, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 279, 4718, 3561, 13, 576, 6722, 374, 12095, 11, 13900, 382, 40, 1184, 311, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7703, 41971, 13, 758, 220, 17, 15, 17, 15, 11, 432, 572, 2163, 220, 17, 13, 16, 21, 3526, 1251, 13, 358, 1265, 10146, 429, 2400, 311, 1281, 2704, 432, 594, 13382, 323, 705, 4686, 18413, 382, 5847, 11, 358, 1184, 311, 2924, 1045, 6770, 3546, 911, 12095, 13, 11445, 22707, 36501, 11, 1075, 279, 1372, 315, 25215, 11, 525, 220, 17, 15, 13, 576, 89445, 552, 374, 1345, 482, 96010, 47435, 811, 11, 892, 374, 458, 2989, 22707, 3082, 13, 576, 57406, 3082, 315, 12095, 374, 1753, 8131, 11, 2163, 220, 16, 17, 13, 20, 3526, 1251, 11, 892, 374, 1661, 311, 5185, 2474, 7025, 1251, 1744, 315, 12095, 438, 264, 33482, 3082, 979, 12831, 1181, 7042, 382, 40, 1265, 5944, 419, 1995, 304, 264, 4718, 3561, 13, 358, 3278, 1855, 264, 1376, 19083, 13530, 1380, 330, 16687, 1, 374, 279, 829, 11, 330, 44441, 1, 374, 279, 1482, 16045, 11, 330, 2733, 1, 5646, 1008, 3565, 1075, 25215, 11, 89445, 552, 11, 323, 57406, 3082, 11, 323, 330, 18286, 1, 646, 614, 894, 9760, 23325, 11, 714, 304, 419, 1142, 11, 432, 594, 30339, 382, 40, 1265, 1281, 2704, 279, 4718, 374, 10277, 23126, 448, 76602, 323, 38929, 13, 2308, 27748, 76602, 311, 5648, 5975, 13, 7281, 11, 279, 7042, 7071, 1265, 387, 13382, 323, 9355, 10982, 13, 358, 1513, 1405, 1366, 311, 5274, 894, 3054, 369, 21340, 11, 773, 8241, 264, 5185, 429, 432, 594, 458, 16045, 374, 264, 1661, 4522, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 387, 4240, 11, 4135, 311, 1349, 11, 323, 6644, 678, 279, 5871, 1995, 2041, 894, 7402, 821, 13, 2938, 1265, 3367, 279, 1196, 594, 1681, 13444, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 16687, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 21, 15, 15, 15, 15, 345, 220, 330, 2733, 788, 341, 262, 330, 7975, 788, 341, 414, 330, 16687, 2259, 11515, 3723, 788, 220, 17, 15, 345, 414, 330, 16687, 2259, 89445, 552, 788, 330, 56375, 9420, 38585, 756, 414, 330, 26331, 361, 788, 330, 71807, 273, 6810, 7276, 34106, 698, 262, 456, 220, 1153, 220, 330, 18286, 788, 330, 23711, 43727, 14789, 1044, 4788, 4914, 326, 6, 1020, 7888, 220, 17, 15, 17, 18, 10040, 532, 73594, 151643], 'meta_info': {'id': '6c69d0f8e1c145e3bc40e2ec014cb313', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 327, 'completion_tokens': 425, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.233599793165922, 'response_sent_to_client_ts': 1776309013.7732394}}</strong>



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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-16 03:10:24] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.03s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.10s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.09s/it]


    2026-04-16 03:10:32,278 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 03:10:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:54,  3.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:54,  3.07s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:23,  1.49s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:23,  1.49s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:49,  1.12it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:49,  1.12it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:23,  2.24it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:23,  2.24it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.88it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.88it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.62it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.62it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.40it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.40it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.29it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.29it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.29it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.51it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.51it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.51it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04, 10.08it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04, 10.08it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04, 10.08it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 12.03it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 12.03it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 12.03it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 12.03it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.45it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.45it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.45it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.45it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.45it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.96it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:01, 20.96it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 40.06it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 40.06it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 40.06it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 40.06it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 40.06it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 40.06it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 40.06it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 40.06it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 40.06it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 49.51it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 49.51it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 49.51it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 49.51it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 49.51it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 49.51it/s]

    Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 49.51it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 47.49it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 47.49it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 47.49it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 47.49it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 47.49it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 47.49it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 43.46it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 43.46it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 43.46it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=104.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=104.78 GB):   2%|▏         | 1/58 [00:00<00:39,  1.45it/s]Capturing num tokens (num_tokens=7680 avail_mem=104.09 GB):   2%|▏         | 1/58 [00:00<00:39,  1.45it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=104.09 GB):   3%|▎         | 2/58 [00:01<00:33,  1.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=104.09 GB):   3%|▎         | 2/58 [00:01<00:33,  1.65it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=104.09 GB):   5%|▌         | 3/58 [00:01<00:30,  1.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=104.10 GB):   5%|▌         | 3/58 [00:01<00:30,  1.79it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=104.10 GB):   7%|▋         | 4/58 [00:02<00:27,  1.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=103.12 GB):   7%|▋         | 4/58 [00:02<00:27,  1.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=103.12 GB):   9%|▊         | 5/58 [00:02<00:26,  2.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=103.13 GB):   9%|▊         | 5/58 [00:02<00:26,  2.02it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=103.13 GB):  10%|█         | 6/58 [00:03<00:23,  2.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=103.13 GB):  10%|█         | 6/58 [00:03<00:23,  2.22it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=103.13 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=102.71 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.41it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=102.71 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=102.71 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.73it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=102.71 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.07it/s]Capturing num tokens (num_tokens=3840 avail_mem=102.71 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.07it/s]Capturing num tokens (num_tokens=3840 avail_mem=102.71 GB):  17%|█▋        | 10/58 [00:04<00:13,  3.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=102.71 GB):  17%|█▋        | 10/58 [00:04<00:13,  3.51it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=102.71 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=102.70 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=102.70 GB):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=102.70 GB):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=102.70 GB):  22%|██▏       | 13/58 [00:04<00:08,  5.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=102.70 GB):  22%|██▏       | 13/58 [00:04<00:08,  5.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=102.70 GB):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=102.70 GB):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=102.70 GB):  26%|██▌       | 15/58 [00:04<00:06,  6.52it/s]Capturing num tokens (num_tokens=2304 avail_mem=102.70 GB):  26%|██▌       | 15/58 [00:04<00:06,  6.52it/s]Capturing num tokens (num_tokens=2048 avail_mem=102.70 GB):  26%|██▌       | 15/58 [00:04<00:06,  6.52it/s]Capturing num tokens (num_tokens=2048 avail_mem=102.70 GB):  29%|██▉       | 17/58 [00:04<00:04,  8.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=101.52 GB):  29%|██▉       | 17/58 [00:04<00:04,  8.27it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=101.52 GB):  31%|███       | 18/58 [00:05<00:05,  7.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=101.51 GB):  31%|███       | 18/58 [00:05<00:05,  7.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=101.51 GB):  33%|███▎      | 19/58 [00:05<00:05,  7.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=102.67 GB):  33%|███▎      | 19/58 [00:05<00:05,  7.57it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=102.67 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=102.67 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=102.67 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.05it/s]Capturing num tokens (num_tokens=960 avail_mem=101.68 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.05it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=101.68 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.80it/s]Capturing num tokens (num_tokens=896 avail_mem=101.68 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.80it/s]Capturing num tokens (num_tokens=896 avail_mem=101.68 GB):  40%|███▉      | 23/58 [00:05<00:04,  8.00it/s]Capturing num tokens (num_tokens=832 avail_mem=102.66 GB):  40%|███▉      | 23/58 [00:05<00:04,  8.00it/s]

    Capturing num tokens (num_tokens=832 avail_mem=102.66 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.12it/s]Capturing num tokens (num_tokens=768 avail_mem=101.73 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.12it/s]Capturing num tokens (num_tokens=768 avail_mem=101.73 GB):  43%|████▎     | 25/58 [00:05<00:04,  8.07it/s]Capturing num tokens (num_tokens=704 avail_mem=101.73 GB):  43%|████▎     | 25/58 [00:05<00:04,  8.07it/s]

    Capturing num tokens (num_tokens=640 avail_mem=102.65 GB):  43%|████▎     | 25/58 [00:06<00:04,  8.07it/s]Capturing num tokens (num_tokens=640 avail_mem=102.65 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.83it/s]Capturing num tokens (num_tokens=576 avail_mem=101.79 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.83it/s]

    Capturing num tokens (num_tokens=576 avail_mem=101.79 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.89it/s]Capturing num tokens (num_tokens=512 avail_mem=102.74 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.89it/s]Capturing num tokens (num_tokens=480 avail_mem=102.64 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.89it/s]Capturing num tokens (num_tokens=480 avail_mem=102.64 GB):  52%|█████▏    | 30/58 [00:06<00:02,  9.73it/s]Capturing num tokens (num_tokens=448 avail_mem=101.86 GB):  52%|█████▏    | 30/58 [00:06<00:02,  9.73it/s]

    Capturing num tokens (num_tokens=448 avail_mem=101.86 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.75it/s]Capturing num tokens (num_tokens=416 avail_mem=102.65 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.75it/s]Capturing num tokens (num_tokens=384 avail_mem=101.91 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.75it/s]Capturing num tokens (num_tokens=384 avail_mem=101.91 GB):  57%|█████▋    | 33/58 [00:06<00:02, 10.22it/s]Capturing num tokens (num_tokens=352 avail_mem=101.91 GB):  57%|█████▋    | 33/58 [00:06<00:02, 10.22it/s]

    Capturing num tokens (num_tokens=320 avail_mem=102.63 GB):  57%|█████▋    | 33/58 [00:06<00:02, 10.22it/s]Capturing num tokens (num_tokens=320 avail_mem=102.63 GB):  60%|██████    | 35/58 [00:06<00:02,  9.85it/s]Capturing num tokens (num_tokens=288 avail_mem=101.67 GB):  60%|██████    | 35/58 [00:06<00:02,  9.85it/s]

    Capturing num tokens (num_tokens=288 avail_mem=101.67 GB):  62%|██████▏   | 36/58 [00:07<00:02,  8.79it/s]Capturing num tokens (num_tokens=256 avail_mem=101.67 GB):  62%|██████▏   | 36/58 [00:07<00:02,  8.79it/s]Capturing num tokens (num_tokens=256 avail_mem=101.67 GB):  64%|██████▍   | 37/58 [00:07<00:02,  8.96it/s]Capturing num tokens (num_tokens=240 avail_mem=101.73 GB):  64%|██████▍   | 37/58 [00:07<00:02,  8.96it/s]

    Capturing num tokens (num_tokens=240 avail_mem=101.73 GB):  66%|██████▌   | 38/58 [00:07<00:02,  8.30it/s]Capturing num tokens (num_tokens=224 avail_mem=101.73 GB):  66%|██████▌   | 38/58 [00:07<00:02,  8.30it/s]Capturing num tokens (num_tokens=208 avail_mem=102.32 GB):  66%|██████▌   | 38/58 [00:07<00:02,  8.30it/s]

    Capturing num tokens (num_tokens=208 avail_mem=102.32 GB):  69%|██████▉   | 40/58 [00:07<00:02,  7.84it/s]Capturing num tokens (num_tokens=192 avail_mem=102.33 GB):  69%|██████▉   | 40/58 [00:07<00:02,  7.84it/s]Capturing num tokens (num_tokens=176 avail_mem=101.83 GB):  69%|██████▉   | 40/58 [00:07<00:02,  7.84it/s]Capturing num tokens (num_tokens=176 avail_mem=101.83 GB):  72%|███████▏  | 42/58 [00:07<00:01,  8.81it/s]Capturing num tokens (num_tokens=160 avail_mem=102.32 GB):  72%|███████▏  | 42/58 [00:07<00:01,  8.81it/s]

    Capturing num tokens (num_tokens=144 avail_mem=101.85 GB):  72%|███████▏  | 42/58 [00:07<00:01,  8.81it/s]Capturing num tokens (num_tokens=144 avail_mem=101.85 GB):  76%|███████▌  | 44/58 [00:07<00:01,  9.70it/s]Capturing num tokens (num_tokens=128 avail_mem=102.32 GB):  76%|███████▌  | 44/58 [00:07<00:01,  9.70it/s]Capturing num tokens (num_tokens=112 avail_mem=101.88 GB):  76%|███████▌  | 44/58 [00:08<00:01,  9.70it/s]

    Capturing num tokens (num_tokens=112 avail_mem=101.88 GB):  79%|███████▉  | 46/58 [00:08<00:01, 10.59it/s]Capturing num tokens (num_tokens=96 avail_mem=102.31 GB):  79%|███████▉  | 46/58 [00:08<00:01, 10.59it/s] Capturing num tokens (num_tokens=80 avail_mem=101.90 GB):  79%|███████▉  | 46/58 [00:08<00:01, 10.59it/s]Capturing num tokens (num_tokens=80 avail_mem=101.90 GB):  83%|████████▎ | 48/58 [00:08<00:00, 11.46it/s]Capturing num tokens (num_tokens=64 avail_mem=101.94 GB):  83%|████████▎ | 48/58 [00:08<00:00, 11.46it/s]Capturing num tokens (num_tokens=48 avail_mem=102.23 GB):  83%|████████▎ | 48/58 [00:08<00:00, 11.46it/s]

    Capturing num tokens (num_tokens=48 avail_mem=102.23 GB):  86%|████████▌ | 50/58 [00:08<00:00, 12.30it/s]Capturing num tokens (num_tokens=32 avail_mem=102.29 GB):  86%|████████▌ | 50/58 [00:08<00:00, 12.30it/s]Capturing num tokens (num_tokens=28 avail_mem=101.96 GB):  86%|████████▌ | 50/58 [00:08<00:00, 12.30it/s]Capturing num tokens (num_tokens=28 avail_mem=101.96 GB):  90%|████████▉ | 52/58 [00:08<00:00, 13.28it/s]Capturing num tokens (num_tokens=24 avail_mem=102.29 GB):  90%|████████▉ | 52/58 [00:08<00:00, 13.28it/s]Capturing num tokens (num_tokens=20 avail_mem=101.98 GB):  90%|████████▉ | 52/58 [00:08<00:00, 13.28it/s]

    Capturing num tokens (num_tokens=20 avail_mem=101.98 GB):  93%|█████████▎| 54/58 [00:08<00:00, 14.40it/s]Capturing num tokens (num_tokens=16 avail_mem=102.28 GB):  93%|█████████▎| 54/58 [00:08<00:00, 14.40it/s]Capturing num tokens (num_tokens=12 avail_mem=102.28 GB):  93%|█████████▎| 54/58 [00:08<00:00, 14.40it/s]Capturing num tokens (num_tokens=12 avail_mem=102.28 GB):  97%|█████████▋| 56/58 [00:08<00:00, 14.78it/s]Capturing num tokens (num_tokens=8 avail_mem=102.04 GB):  97%|█████████▋| 56/58 [00:08<00:00, 14.78it/s] Capturing num tokens (num_tokens=4 avail_mem=102.25 GB):  97%|█████████▋| 56/58 [00:08<00:00, 14.78it/s]

    Capturing num tokens (num_tokens=4 avail_mem=102.25 GB): 100%|██████████| 58/58 [00:08<00:00,  6.55it/s]


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
        "population": 138000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
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
    
    Generated text: Okay, the user just asked for the information and population of the capital of France in JSON format. So, first, I need to figure out what exactly they're looking for. The capital of France is definitely Paris. I should make sure I have the correct and up-to-date population figure. I recall that as of the latest data, the population of Paris is around 2.1 million. But wait, populations can change every year, so I should double-check that number to ensure accuracy. Maybe I should look up the current estimates to confirm. 
    
    Also, the user asked for the information and population. So, I should structure the JSON to include both the city name and its population. It would make sense to put it into a proper JSON format, which is a data structure used for storing information. So, the key points are the city name and the population figure. 
    
    I need to present this in a clear and concise way without any extra information that the user didn't ask for. Keeping it straightforward will help the user get exactly what they need quickly. I should avoid any unnecessary details or formatting issues to make sure the JSON is valid and can be used properly.
    
    Alright, putting it all together, I'll create a JSON object with the city name and population. I'll make sure the numbers are accurate and formatted correctly. That should cover everything the user is asking for.
    </think>
    
    Certainly! Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "city": "Paris",
      "population": 2162000
    }
    ```
    
    This JSON object contains the name of the capital city ("Paris") and its approximate population as of the latest available data (2,162,000).



```python
llm.shutdown()
```
