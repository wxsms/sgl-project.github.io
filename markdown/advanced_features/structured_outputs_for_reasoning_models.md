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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.21s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.22s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.22s/it]


    2026-05-13 03:59:17,016 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 03:59:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:26,  5.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:26,  5.73s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:19,  2.48s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:19,  2.48s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:19,  1.44s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:19,  1.44s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:50,  1.06it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:50,  1.06it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:25,  2.06it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:25,  2.06it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.71it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.71it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.43it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.43it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.29it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.29it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:07<00:11,  4.29it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:07,  5.98it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:07,  5.98it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:07,  5.98it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:05,  7.57it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:05,  7.57it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:05,  7.57it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.18it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.18it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.18it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.07it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.07it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.07it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.07it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.30it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.30it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.30it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.30it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.30it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.60it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.60it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.60it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.60it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.60it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.60it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]

    Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:08<00:00, 32.30it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:08<00:00, 32.30it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:08<00:00, 32.30it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:08<00:00, 32.30it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:08<00:00, 32.30it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:08<00:00, 32.30it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:08<00:00, 32.30it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:08<00:00, 38.82it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:08<00:00, 38.82it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:08<00:00, 38.82it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:08<00:00, 38.82it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:08<00:00, 38.82it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:08<00:00, 38.82it/s]

    Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:08<00:00, 38.82it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:08<00:00, 38.82it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:08<00:00, 46.52it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:08<00:00, 46.52it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:08<00:00, 46.52it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:08<00:00, 46.52it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:08<00:00, 46.52it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:08<00:00, 46.52it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:08<00:00, 46.52it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:08<00:00, 46.52it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:08<00:00, 46.52it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:08<00:00, 54.43it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:08<00:00, 54.43it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:08<00:00, 54.43it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:08<00:00, 54.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=45.99 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.95 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=45.95 GB):   3%|▎         | 2/58 [00:00<00:15,  3.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.95 GB):   3%|▎         | 2/58 [00:00<00:15,  3.57it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=45.95 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=45.95 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=45.95 GB):   7%|▋         | 4/58 [00:01<00:13,  4.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.95 GB):   7%|▋         | 4/58 [00:01<00:13,  4.02it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=45.95 GB):   9%|▊         | 5/58 [00:01<00:13,  3.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.95 GB):   9%|▊         | 5/58 [00:01<00:13,  3.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.95 GB):  10%|█         | 6/58 [00:01<00:11,  4.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.95 GB):  10%|█         | 6/58 [00:01<00:11,  4.37it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=45.95 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.95 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.95 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.95 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.34it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=45.95 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.95 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.95 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.35it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.95 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.35it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=45.95 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.91it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.95 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.91it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=45.95 GB):  21%|██        | 12/58 [00:02<00:10,  4.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.16 GB):  21%|██        | 12/58 [00:02<00:10,  4.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.16 GB):  22%|██▏       | 13/58 [00:02<00:08,  5.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.16 GB):  22%|██▏       | 13/58 [00:02<00:08,  5.08it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.16 GB):  24%|██▍       | 14/58 [00:02<00:07,  5.91it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.15 GB):  24%|██▍       | 14/58 [00:02<00:07,  5.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.15 GB):  24%|██▍       | 14/58 [00:02<00:07,  5.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.36it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=43.15 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.15 GB):  31%|███       | 18/58 [00:03<00:04,  9.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.14 GB):  31%|███       | 18/58 [00:03<00:04,  9.09it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.14 GB):  31%|███       | 18/58 [00:03<00:04,  9.09it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.14 GB):  34%|███▍      | 20/58 [00:03<00:03, 10.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.13 GB):  34%|███▍      | 20/58 [00:03<00:03, 10.80it/s]

    Capturing num tokens (num_tokens=960 avail_mem=43.13 GB):  34%|███▍      | 20/58 [00:03<00:03, 10.80it/s] Capturing num tokens (num_tokens=896 avail_mem=43.12 GB):  34%|███▍      | 20/58 [00:03<00:03, 10.80it/s]Capturing num tokens (num_tokens=896 avail_mem=43.12 GB):  40%|███▉      | 23/58 [00:03<00:02, 13.21it/s]Capturing num tokens (num_tokens=832 avail_mem=43.12 GB):  40%|███▉      | 23/58 [00:03<00:02, 13.21it/s]Capturing num tokens (num_tokens=768 avail_mem=43.11 GB):  40%|███▉      | 23/58 [00:03<00:02, 13.21it/s]Capturing num tokens (num_tokens=704 avail_mem=43.11 GB):  40%|███▉      | 23/58 [00:03<00:02, 13.21it/s]

    Capturing num tokens (num_tokens=704 avail_mem=43.11 GB):  45%|████▍     | 26/58 [00:03<00:02, 15.43it/s]Capturing num tokens (num_tokens=640 avail_mem=43.11 GB):  45%|████▍     | 26/58 [00:03<00:02, 15.43it/s]Capturing num tokens (num_tokens=576 avail_mem=43.10 GB):  45%|████▍     | 26/58 [00:03<00:02, 15.43it/s]Capturing num tokens (num_tokens=512 avail_mem=43.10 GB):  45%|████▍     | 26/58 [00:03<00:02, 15.43it/s]Capturing num tokens (num_tokens=512 avail_mem=43.10 GB):  50%|█████     | 29/58 [00:03<00:01, 17.44it/s]Capturing num tokens (num_tokens=480 avail_mem=43.10 GB):  50%|█████     | 29/58 [00:03<00:01, 17.44it/s]Capturing num tokens (num_tokens=448 avail_mem=43.09 GB):  50%|█████     | 29/58 [00:03<00:01, 17.44it/s]

    Capturing num tokens (num_tokens=416 avail_mem=43.09 GB):  50%|█████     | 29/58 [00:03<00:01, 17.44it/s]Capturing num tokens (num_tokens=416 avail_mem=43.09 GB):  55%|█████▌    | 32/58 [00:03<00:01, 18.73it/s]Capturing num tokens (num_tokens=384 avail_mem=43.09 GB):  55%|█████▌    | 32/58 [00:03<00:01, 18.73it/s]Capturing num tokens (num_tokens=352 avail_mem=43.08 GB):  55%|█████▌    | 32/58 [00:03<00:01, 18.73it/s]Capturing num tokens (num_tokens=320 avail_mem=43.08 GB):  55%|█████▌    | 32/58 [00:03<00:01, 18.73it/s]Capturing num tokens (num_tokens=320 avail_mem=43.08 GB):  60%|██████    | 35/58 [00:03<00:01, 21.25it/s]Capturing num tokens (num_tokens=288 avail_mem=43.08 GB):  60%|██████    | 35/58 [00:03<00:01, 21.25it/s]Capturing num tokens (num_tokens=256 avail_mem=43.08 GB):  60%|██████    | 35/58 [00:04<00:01, 21.25it/s]

    Capturing num tokens (num_tokens=240 avail_mem=43.07 GB):  60%|██████    | 35/58 [00:04<00:01, 21.25it/s]Capturing num tokens (num_tokens=240 avail_mem=43.07 GB):  66%|██████▌   | 38/58 [00:04<00:00, 22.82it/s]Capturing num tokens (num_tokens=224 avail_mem=43.07 GB):  66%|██████▌   | 38/58 [00:04<00:00, 22.82it/s]Capturing num tokens (num_tokens=208 avail_mem=43.07 GB):  66%|██████▌   | 38/58 [00:04<00:00, 22.82it/s]Capturing num tokens (num_tokens=192 avail_mem=43.06 GB):  66%|██████▌   | 38/58 [00:04<00:00, 22.82it/s]Capturing num tokens (num_tokens=192 avail_mem=43.06 GB):  71%|███████   | 41/58 [00:04<00:00, 23.76it/s]Capturing num tokens (num_tokens=176 avail_mem=43.06 GB):  71%|███████   | 41/58 [00:04<00:00, 23.76it/s]Capturing num tokens (num_tokens=160 avail_mem=43.06 GB):  71%|███████   | 41/58 [00:04<00:00, 23.76it/s]

    Capturing num tokens (num_tokens=144 avail_mem=43.05 GB):  71%|███████   | 41/58 [00:04<00:00, 23.76it/s]Capturing num tokens (num_tokens=144 avail_mem=43.05 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.72it/s]Capturing num tokens (num_tokens=128 avail_mem=43.05 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.72it/s]Capturing num tokens (num_tokens=112 avail_mem=43.05 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.72it/s]Capturing num tokens (num_tokens=96 avail_mem=43.05 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.72it/s] Capturing num tokens (num_tokens=80 avail_mem=43.04 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.72it/s]Capturing num tokens (num_tokens=80 avail_mem=43.04 GB):  83%|████████▎ | 48/58 [00:04<00:00, 28.05it/s]Capturing num tokens (num_tokens=64 avail_mem=43.04 GB):  83%|████████▎ | 48/58 [00:04<00:00, 28.05it/s]Capturing num tokens (num_tokens=48 avail_mem=43.03 GB):  83%|████████▎ | 48/58 [00:04<00:00, 28.05it/s]

    Capturing num tokens (num_tokens=32 avail_mem=43.03 GB):  83%|████████▎ | 48/58 [00:04<00:00, 28.05it/s]Capturing num tokens (num_tokens=28 avail_mem=43.03 GB):  83%|████████▎ | 48/58 [00:04<00:00, 28.05it/s]Capturing num tokens (num_tokens=28 avail_mem=43.03 GB):  90%|████████▉ | 52/58 [00:04<00:00, 29.31it/s]Capturing num tokens (num_tokens=24 avail_mem=43.03 GB):  90%|████████▉ | 52/58 [00:04<00:00, 29.31it/s]Capturing num tokens (num_tokens=20 avail_mem=43.02 GB):  90%|████████▉ | 52/58 [00:04<00:00, 29.31it/s]Capturing num tokens (num_tokens=16 avail_mem=43.02 GB):  90%|████████▉ | 52/58 [00:04<00:00, 29.31it/s]Capturing num tokens (num_tokens=12 avail_mem=43.01 GB):  90%|████████▉ | 52/58 [00:04<00:00, 29.31it/s]Capturing num tokens (num_tokens=12 avail_mem=43.01 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.21it/s]Capturing num tokens (num_tokens=8 avail_mem=43.01 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.21it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=43.01 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.21it/s]Capturing num tokens (num_tokens=4 avail_mem=43.01 GB): 100%|██████████| 58/58 [00:04<00:00, 12.36it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think the population has been growing over the years. I recall reading somewhere that it's over 21 million now. Maybe around 21.6 million? I'm not sure if that's the exact number or just an estimate. I should look it up to confirm. Also, I should make sure that Paris is indeed the capital and not another city like Lyon or Marseille. I'm pretty sure Paris is the official capital, but I'm not 100% certain. Maybe I can think about the most well-known city in France and that's probably Paris.<br><br>So, putting it all together, the capital is Paris, and the population is approximately 21.6 million. I should present this information in JSON format as the user requested. I need to make sure the JSON is correctly formatted with the key "capital" and "population". I should also include the population as a number, not a string, so it's 21600000. Let me double-check the population number to ensure accuracy. Yeah, I think that's correct. So the final JSON should have the correct structure with the right values.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21600000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think the population has been growing over the years. I recall reading somewhere that it's over 21 million now. Maybe around 21.6 million? I'm not sure if that's the exact number or just an estimate. I should look it up to confirm. Also, I should make sure that Paris is indeed the capital and not another city like Lyon or Marseille. I'm pretty sure Paris is the official capital, but I'm not 100% certain. Maybe I can think about the most well-known city in France and that's probably Paris.<br><br>So, putting it all together, the capital is Paris, and the population is approximately 21.6 million. I should present this information in JSON format as the user requested. I need to make sure the JSON is correctly formatted with the key "capital" and "population". I should also include the population as a number, not a string, so it's 21600000. Let me double-check the population number to ensure accuracy. Yeah, I think that's correct. So the final JSON should have the correct structure with the right values.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21600000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population. I think it's a big city, maybe around 3 million? But I'm not certain. I should probably check some reliable sources to confirm this. Maybe I can look up recent population data or news articles that mention Paris's population. I remember hearing that Paris is one of the most populous cities in the world, but I'm not sure if it's over 3 million or not. I should also consider factors like urbanization and migration that might affect the population numbers. Maybe the population has grown a bit since the last census. I'll try to recall if I've heard any recent statistics or if there are any upcoming censuses that might provide the latest data. I think the population figure is something like 3.5 million, but I'm not entirely sure. I should make sure to present this information in a clear and accurate way, perhaps referencing a recent source or official statistics to back it up.<br><br><br>content: Paris is the capital of France</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning a bit about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light" because of the famous Eiffel Tower. But is Paris the capital? I think it is, but I'm not entirely certain. <br><br>Wait, I also recall that there's another city called Lyon. Isn't Lyon the capital of a region or something? Maybe I'm mixing up the regions. I think the capital refers to the main city, so Paris might be the official capital. But I'm a bit confused because sometimes people talk about different capitals for regions or departments. For example, I think each department has a capital city, and maybe Paris is the capital of a department or something like that. <br><br>Let me try to remember. The Eiffel Tower is in Paris, and it's a symbol of France. Also, the Louvre Museum is in Paris, which is a world-renowned museum. So, if Paris is such a significant city with all these famous landmarks, it makes sense that it's the capital. But I'm still a little unsure because I think I heard somewhere that Lyon is the capital of France, but that might be incorrect. <br><br>I should probably double-check. I know that the capital is the seat of government, so maybe I can think about other capitals I know. For example, Germany's capital is Berlin, Italy's is Rome, Spain's is Madrid. So, following that pattern, France's capital should be Paris. Yeah, that seems right. I think I was confusing it with another city, maybe Lyon, but no, I'm pretty sure Paris is correct. <br><br>Also, I remember that Paris is the administrative center, where the government offices are located. So, that would make it the capital. Yeah, I'm pretty confident now that Paris is the capital of France.<br><br><br>content: Paris</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the current date and time in New York, along with the weather. Let me break this down step by step.<br><br>First, I need to figure out how to get the current date and time. The available function for that is 'get_current_date'. It requires a timezone parameter. Since the user is in New York, I'll use 'America/New_York' as the timezone. So the function call should be <function=get_current_date>{"timezone": "America/New_York"}</function>.<br><br>Next, the user also wants the weather. That means I need to use the 'get_current_weather' function. For this, I'll need the city, state, and unit. The city is New York, but I should check the state abbreviation. New York is NY, so the state is 'NY'. The unit can be Fahrenheit since the user didn't specify otherwise. So the parameters for this function will be city: "New York", state: "NY", unit: "fahrenheit".<br><br>Now, I have to call both functions one after the other. I'll start with the date and time, then move on to the weather. I'll make sure to format each function call correctly, using the specified syntax with start_tag and end_tag.<br><br>Putting it all together, I'll write the two function calls on separate lines. First, the date and time with the timezone, then the weather with the city, state, and unit. I'll include the sources as per the instructions, using the respective function names and parameters.<br><br>I should double-check that all required parameters are included and that the syntax is correct. No markdown, just plain text with the function calls in the specified format. Also, make sure to mention the sources in the response to add credibility.<br><br>Alright, I think that covers everything the user is asking for. Time to structure the response properly.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>  <br><br>Sources:  <br>- get_current_date: https://openweathermap.org/api  <br>- get_current_weather: https://openweathermap.org/api</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '664087f3d3064d25883b11c6dbd089b2', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.731567265000194, 'response_sent_to_client_ts': 1778644803.8815753}}



<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that's the starting point.<br><br>Next, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I'm not exactly sure of the current number. I think it's around 2 million, but I should double-check that. Maybe I can recall that it's approximately 2,150,000 as of recent estimates.<br><br>Now, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.<br><br>I should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they're strings, but population is a number, so it should be without quotes.<br><br>Putting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.<br><br>I wonder if the user needs more details, like the population figure's source or the exact year it was recorded. But since they only asked for the information, I'll stick to what's requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.<br><br>Also, considering the user's possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.<br><br>In summary, I'll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I'll keep it simple and straightforward since the user didn't ask for anything too complex.<br><br><br>content: {"name": "Paris", "population": 2150000}</strong>


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '7d195d8353ef4f968ab2af01df7e1f89', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 6.654046102426946, 'response_sent_to_client_ts': 1778644810.5453465}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '70c1d2f4df3a497197ee268302f8a654', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.13375292532145977, 'response_sent_to_client_ts': 1778644810.7240243}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '1d3f2b6bf7f340d393ee0c8b31f4a9a8', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.13368029287084937, 'response_sent_to_client_ts': 1778644810.7240381}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '6e5e51cfefc64350b81135062700ccc9', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1336388378404081, 'response_sent_to_client_ts': 1778644810.7240431}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '9e74d5ceeb594842a3f568542830bf8b', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 22.5454691760242, 'response_sent_to_client_ts': 1778644833.2777185}}


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


<strong style='color: #00008B;'>{'text': 'Okay, I need to figure out how to respond to this user query. They\'re asking for the information and population of the capital of France in JSON format. \n\nFirst, I should recall that the capital of France is Paris. Got it. Next, I need the population. From what I remember, Paris has a population around 2 million. But I should double-check that to make sure it\'s accurate. Maybe I can recall recent data or think about any recent news about population changes. I believe the figure is approximately 2,170,000 as of the latest estimates. \n\nNow, the user wants this in JSON format. JSON typically uses key-value pairs, so I\'ll structure it with "capital" as the key and the population as the corresponding value. I should also include the city\'s name, Paris, in the structure. \n\nI might also consider providing some related geographical information, like the administrative division code and area in square kilometers. That could add more value and make the response more comprehensive. I think Paris is in Ile-de-France and its administrative code is 92. Its area is about 105 square kilometers. Including this information could be helpful if the user is looking for more detailed data.\n\nI need to make sure that the JSON is correctly formatted with proper syntax—commas in the right places and no trailing commas. Also, the keys should be in camel case, so "capital" and "population" are fine, and "/adminDivCode" seems like a good addition for more context.\n\nPutting it all together, I\'ll structure the JSON with the capital\'s details and the population. Let me make sure the population is accurate and that the other fields are correctly placed. There\'s no need for any markdown here, just plain JSON as per the user\'s request.\n\nSo, the final JSON should include "city" with the name, "capital" indicating it\'s the capital, "population" with the number, "administrativeDivisionCode" as 92, and "area" in square kilometers. I think that covers all the bases the user is asking for. I\'ll present this in a clear, concise manner without any extra fluff.\n</think>\n\n```json\n{\n  "city": "Paris",\n  "capital": true,\n  "population": 2170000,\n  "administrativeDivisionCode": 92,\n  "area": 105\n}\n```', 'output_ids': [32313, 11, 358, 1184, 311, 7071, 700, 1246, 311, 5889, 311, 419, 1196, 3239, 13, 2379, 2299, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 4710, 5338, 11, 358, 1265, 19091, 429, 279, 6722, 315, 9625, 374, 12095, 13, 24528, 432, 13, 9295, 11, 358, 1184, 279, 7042, 13, 5542, 1128, 358, 6099, 11, 12095, 702, 264, 7042, 2163, 220, 17, 3526, 13, 1988, 358, 1265, 1990, 15934, 429, 311, 1281, 2704, 432, 594, 13382, 13, 10696, 358, 646, 19091, 3213, 821, 476, 1744, 911, 894, 3213, 3669, 911, 7042, 4344, 13, 358, 4411, 279, 7071, 374, 13187, 220, 17, 11, 16, 22, 15, 11, 15, 15, 15, 438, 315, 279, 5535, 17530, 13, 4710, 7039, 11, 279, 1196, 6801, 419, 304, 4718, 3561, 13, 4718, 11136, 5711, 1376, 19083, 13530, 11, 773, 358, 3278, 5944, 432, 448, 330, 65063, 1, 438, 279, 1376, 323, 279, 7042, 438, 279, 12159, 897, 13, 358, 1265, 1083, 2924, 279, 3283, 594, 829, 11, 12095, 11, 304, 279, 5944, 13, 4710, 40, 2578, 1083, 2908, 8241, 1045, 5435, 52901, 1995, 11, 1075, 279, 22707, 12804, 2038, 323, 3082, 304, 9334, 40568, 13, 2938, 1410, 912, 803, 897, 323, 1281, 279, 2033, 803, 15817, 13, 358, 1744, 12095, 374, 304, 358, 273, 6810, 7276, 34106, 323, 1181, 22707, 2038, 374, 220, 24, 17, 13, 11445, 3082, 374, 911, 220, 16, 15, 20, 9334, 40568, 13, 55121, 419, 1995, 1410, 387, 10950, 421, 279, 1196, 374, 3330, 369, 803, 11682, 821, 382, 40, 1184, 311, 1281, 2704, 429, 279, 4718, 374, 12440, 23126, 448, 6169, 19482, 2293, 3621, 300, 304, 279, 1290, 7482, 323, 902, 27748, 76602, 13, 7281, 11, 279, 6894, 1265, 387, 304, 49152, 1142, 11, 773, 330, 65063, 1, 323, 330, 44441, 1, 525, 6915, 11, 323, 330, 17402, 12509, 2078, 1, 4977, 1075, 264, 1661, 5256, 369, 803, 2266, 382, 97904, 432, 678, 3786, 11, 358, 3278, 5944, 279, 4718, 448, 279, 6722, 594, 3565, 323, 279, 7042, 13, 6771, 752, 1281, 2704, 279, 7042, 374, 13382, 323, 429, 279, 1008, 5043, 525, 12440, 9099, 13, 2619, 594, 902, 1184, 369, 894, 50494, 1588, 11, 1101, 14396, 4718, 438, 817, 279, 1196, 594, 1681, 382, 4416, 11, 279, 1590, 4718, 1265, 2924, 330, 8926, 1, 448, 279, 829, 11, 330, 65063, 1, 18860, 432, 594, 279, 6722, 11, 330, 44441, 1, 448, 279, 1372, 11, 330, 68849, 1388, 51237, 2078, 1, 438, 220, 24, 17, 11, 323, 330, 4798, 1, 304, 9334, 40568, 13, 358, 1744, 429, 14521, 678, 279, 23092, 279, 1196, 374, 10161, 369, 13, 358, 3278, 3042, 419, 304, 264, 2797, 11, 63594, 11566, 2041, 894, 4960, 1320, 1362, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 8926, 788, 330, 59604, 756, 220, 330, 65063, 788, 830, 345, 220, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 345, 220, 330, 68849, 1388, 51237, 2078, 788, 220, 24, 17, 345, 220, 330, 4798, 788, 220, 16, 15, 20, 198, 532, 73594, 151643], 'meta_info': {'id': 'a5c0cafb42f94f10ba5107d75be871b7', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 450, 'completion_tokens': 504, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.695253715850413, 'response_sent_to_client_ts': 1778644837.981766}}</strong>



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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.12s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.06s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.07s/it]


    2026-05-13 04:00:54,782 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 04:00:54] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:02,  5.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:02,  5.31s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:09,  2.31s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:09,  2.31s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:14,  1.35s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:14,  1.35s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:33,  1.61it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:33,  1.61it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:24,  2.16it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:24,  2.16it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.80it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.80it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.56it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.56it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.41it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.41it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.41it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:08,  5.68it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:08,  5.68it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:08,  5.56it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:08,  5.56it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:07,  5.65it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:07,  5.65it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:07,  5.65it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:05,  7.65it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:05,  7.65it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:05,  7.65it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:04,  9.76it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:04,  9.76it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:04,  9.76it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:04,  9.76it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 13.32it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 13.32it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 13.32it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 13.32it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 13.32it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 18.82it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 18.82it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 18.82it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 18.82it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 18.82it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 18.82it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.16it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.16it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.16it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.16it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.16it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.16it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.16it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 34.34it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 34.34it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 34.34it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 34.34it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 34.34it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 34.34it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 34.34it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 34.34it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:07<00:00, 34.34it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 44.61it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 44.61it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:07<00:00, 44.61it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:08<00:00, 44.61it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:08<00:00, 44.61it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:08<00:00, 44.61it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:08<00:00, 44.61it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:08<00:00, 44.61it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:08<00:00, 44.61it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:08<00:00, 52.30it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:08<00:00, 52.30it/s]

    Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:08<00:00, 52.30it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:08<00:00, 52.30it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:08<00:00, 52.30it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:08<00:00, 52.30it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:08<00:00, 52.30it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:08<00:00, 52.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=44.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=44.18 GB):   2%|▏         | 1/58 [00:00<00:16,  3.38it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.15 GB):   2%|▏         | 1/58 [00:00<00:16,  3.38it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.15 GB):   3%|▎         | 2/58 [00:00<00:15,  3.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=44.15 GB):   3%|▎         | 2/58 [00:00<00:15,  3.55it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=44.15 GB):   5%|▌         | 3/58 [00:00<00:14,  3.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.15 GB):   5%|▌         | 3/58 [00:00<00:14,  3.78it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=44.15 GB):   7%|▋         | 4/58 [00:01<00:13,  4.05it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.15 GB):   7%|▋         | 4/58 [00:01<00:13,  4.05it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.15 GB):   9%|▊         | 5/58 [00:01<00:15,  3.43it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.15 GB):   9%|▊         | 5/58 [00:01<00:15,  3.43it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=44.15 GB):  10%|█         | 6/58 [00:01<00:16,  3.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.15 GB):  10%|█         | 6/58 [00:01<00:16,  3.06it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=44.15 GB):  12%|█▏        | 7/58 [00:02<00:17,  2.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.15 GB):  12%|█▏        | 7/58 [00:02<00:17,  2.93it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=44.15 GB):  14%|█▍        | 8/58 [00:02<00:16,  2.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.15 GB):  14%|█▍        | 8/58 [00:02<00:16,  2.96it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=44.15 GB):  16%|█▌        | 9/58 [00:02<00:16,  3.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.15 GB):  16%|█▌        | 9/58 [00:02<00:16,  3.04it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=44.15 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=44.15 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.14it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=44.15 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.15 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.28it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=44.15 GB):  21%|██        | 12/58 [00:03<00:13,  3.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=44.14 GB):  21%|██        | 12/58 [00:03<00:13,  3.29it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=44.14 GB):  22%|██▏       | 13/58 [00:03<00:12,  3.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=44.14 GB):  22%|██▏       | 13/58 [00:03<00:12,  3.48it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=44.14 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.76it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.14 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.76it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=44.14 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.14 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.14 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.13 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.43it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=44.13 GB):  29%|██▉       | 17/58 [00:04<00:08,  4.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.13 GB):  29%|██▉       | 17/58 [00:04<00:08,  4.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.13 GB):  31%|███       | 18/58 [00:04<00:07,  5.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.13 GB):  31%|███       | 18/58 [00:04<00:07,  5.31it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=44.13 GB):  33%|███▎      | 19/58 [00:04<00:06,  5.99it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.13 GB):  33%|███▎      | 19/58 [00:04<00:06,  5.99it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.13 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.12 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.71it/s]Capturing num tokens (num_tokens=960 avail_mem=44.11 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.71it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=44.11 GB):  38%|███▊      | 22/58 [00:05<00:04,  8.39it/s]Capturing num tokens (num_tokens=896 avail_mem=44.11 GB):  38%|███▊      | 22/58 [00:05<00:04,  8.39it/s]Capturing num tokens (num_tokens=832 avail_mem=44.11 GB):  38%|███▊      | 22/58 [00:05<00:04,  8.39it/s]Capturing num tokens (num_tokens=832 avail_mem=44.11 GB):  41%|████▏     | 24/58 [00:05<00:03,  9.93it/s]Capturing num tokens (num_tokens=768 avail_mem=44.10 GB):  41%|████▏     | 24/58 [00:05<00:03,  9.93it/s]

    Capturing num tokens (num_tokens=704 avail_mem=44.10 GB):  41%|████▏     | 24/58 [00:05<00:03,  9.93it/s]Capturing num tokens (num_tokens=704 avail_mem=44.10 GB):  45%|████▍     | 26/58 [00:05<00:02, 11.82it/s]Capturing num tokens (num_tokens=640 avail_mem=44.09 GB):  45%|████▍     | 26/58 [00:05<00:02, 11.82it/s]Capturing num tokens (num_tokens=576 avail_mem=44.09 GB):  45%|████▍     | 26/58 [00:05<00:02, 11.82it/s]Capturing num tokens (num_tokens=576 avail_mem=44.09 GB):  48%|████▊     | 28/58 [00:05<00:02, 13.41it/s]Capturing num tokens (num_tokens=512 avail_mem=44.08 GB):  48%|████▊     | 28/58 [00:05<00:02, 13.41it/s]

    Capturing num tokens (num_tokens=480 avail_mem=44.08 GB):  48%|████▊     | 28/58 [00:05<00:02, 13.41it/s]Capturing num tokens (num_tokens=448 avail_mem=44.08 GB):  48%|████▊     | 28/58 [00:05<00:02, 13.41it/s]Capturing num tokens (num_tokens=448 avail_mem=44.08 GB):  53%|█████▎    | 31/58 [00:05<00:01, 15.38it/s]Capturing num tokens (num_tokens=416 avail_mem=44.07 GB):  53%|█████▎    | 31/58 [00:05<00:01, 15.38it/s]Capturing num tokens (num_tokens=384 avail_mem=44.07 GB):  53%|█████▎    | 31/58 [00:05<00:01, 15.38it/s]Capturing num tokens (num_tokens=352 avail_mem=44.06 GB):  53%|█████▎    | 31/58 [00:05<00:01, 15.38it/s]

    Capturing num tokens (num_tokens=352 avail_mem=44.06 GB):  59%|█████▊    | 34/58 [00:05<00:01, 18.60it/s]Capturing num tokens (num_tokens=320 avail_mem=44.06 GB):  59%|█████▊    | 34/58 [00:05<00:01, 18.60it/s]Capturing num tokens (num_tokens=288 avail_mem=44.07 GB):  59%|█████▊    | 34/58 [00:05<00:01, 18.60it/s]Capturing num tokens (num_tokens=256 avail_mem=44.06 GB):  59%|█████▊    | 34/58 [00:05<00:01, 18.60it/s]Capturing num tokens (num_tokens=240 avail_mem=44.06 GB):  59%|█████▊    | 34/58 [00:05<00:01, 18.60it/s]Capturing num tokens (num_tokens=240 avail_mem=44.06 GB):  66%|██████▌   | 38/58 [00:05<00:00, 21.81it/s]Capturing num tokens (num_tokens=224 avail_mem=44.05 GB):  66%|██████▌   | 38/58 [00:05<00:00, 21.81it/s]Capturing num tokens (num_tokens=208 avail_mem=44.05 GB):  66%|██████▌   | 38/58 [00:06<00:00, 21.81it/s]

    Capturing num tokens (num_tokens=192 avail_mem=44.05 GB):  66%|██████▌   | 38/58 [00:06<00:00, 21.81it/s]Capturing num tokens (num_tokens=192 avail_mem=44.05 GB):  71%|███████   | 41/58 [00:06<00:00, 21.49it/s]Capturing num tokens (num_tokens=176 avail_mem=44.04 GB):  71%|███████   | 41/58 [00:06<00:00, 21.49it/s]Capturing num tokens (num_tokens=160 avail_mem=44.04 GB):  71%|███████   | 41/58 [00:06<00:00, 21.49it/s]Capturing num tokens (num_tokens=144 avail_mem=44.03 GB):  71%|███████   | 41/58 [00:06<00:00, 21.49it/s]Capturing num tokens (num_tokens=144 avail_mem=44.03 GB):  76%|███████▌  | 44/58 [00:06<00:00, 22.44it/s]Capturing num tokens (num_tokens=128 avail_mem=44.04 GB):  76%|███████▌  | 44/58 [00:06<00:00, 22.44it/s]Capturing num tokens (num_tokens=112 avail_mem=44.03 GB):  76%|███████▌  | 44/58 [00:06<00:00, 22.44it/s]

    Capturing num tokens (num_tokens=96 avail_mem=44.03 GB):  76%|███████▌  | 44/58 [00:06<00:00, 22.44it/s] Capturing num tokens (num_tokens=96 avail_mem=44.03 GB):  81%|████████  | 47/58 [00:06<00:00, 21.94it/s]Capturing num tokens (num_tokens=80 avail_mem=42.89 GB):  81%|████████  | 47/58 [00:06<00:00, 21.94it/s]Capturing num tokens (num_tokens=64 avail_mem=42.89 GB):  81%|████████  | 47/58 [00:06<00:00, 21.94it/s]

    Capturing num tokens (num_tokens=48 avail_mem=44.27 GB):  81%|████████  | 47/58 [00:06<00:00, 21.94it/s]Capturing num tokens (num_tokens=48 avail_mem=44.27 GB):  86%|████████▌ | 50/58 [00:06<00:00, 17.58it/s]Capturing num tokens (num_tokens=32 avail_mem=43.98 GB):  86%|████████▌ | 50/58 [00:06<00:00, 17.58it/s]Capturing num tokens (num_tokens=28 avail_mem=42.99 GB):  86%|████████▌ | 50/58 [00:06<00:00, 17.58it/s]

    Capturing num tokens (num_tokens=28 avail_mem=42.99 GB):  90%|████████▉ | 52/58 [00:06<00:00, 14.16it/s]Capturing num tokens (num_tokens=24 avail_mem=42.99 GB):  90%|████████▉ | 52/58 [00:06<00:00, 14.16it/s]Capturing num tokens (num_tokens=20 avail_mem=43.97 GB):  90%|████████▉ | 52/58 [00:06<00:00, 14.16it/s]Capturing num tokens (num_tokens=20 avail_mem=43.97 GB):  93%|█████████▎| 54/58 [00:07<00:00, 13.52it/s]Capturing num tokens (num_tokens=16 avail_mem=43.05 GB):  93%|█████████▎| 54/58 [00:07<00:00, 13.52it/s]

    Capturing num tokens (num_tokens=12 avail_mem=43.04 GB):  93%|█████████▎| 54/58 [00:07<00:00, 13.52it/s]Capturing num tokens (num_tokens=12 avail_mem=43.04 GB):  97%|█████████▋| 56/58 [00:07<00:00, 12.42it/s]Capturing num tokens (num_tokens=8 avail_mem=43.04 GB):  97%|█████████▋| 56/58 [00:07<00:00, 12.42it/s] Capturing num tokens (num_tokens=4 avail_mem=43.96 GB):  97%|█████████▋| 56/58 [00:07<00:00, 12.42it/s]

    Capturing num tokens (num_tokens=4 avail_mem=43.96 GB): 100%|██████████| 58/58 [00:07<00:00, 12.23it/s]Capturing num tokens (num_tokens=4 avail_mem=43.96 GB): 100%|██████████| 58/58 [00:07<00:00,  7.81it/s]


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
    
    Generated text: Okay, so I need to figure out the capital of France and its population. Hmm, I know that France is a country in Europe, and I think its capital is Paris. Yeah, Paris is the most famous city there and the one everyone refers to as the capital. Now, about the population. I remember reading somewhere that Paris is one of the largest cities in the world. I think the population is over 30 million, maybe 32 or 33 million? But I'm not sure if it's exactly 33. I should double-check that. Wait, sometimes the population counts can vary depending on the source and the year. Maybe it's around 33.7 million now? I think I've heard that figure before. So, putting it all together, the JSON format would have "capital" as "Paris" and "population" as 33700000, which is 33,700,000. Let me make sure that's correct. Yeah, I think that's right. So, the JSON would look like {"capital": "Paris", "population": 33700000}.
    </think>
    
    ```json
    {
      "capital": "Paris",
      "population": 33700000
    }
    ```



```python
llm.shutdown()
```
