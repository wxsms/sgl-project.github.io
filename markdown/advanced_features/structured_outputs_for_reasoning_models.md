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


    [2026-05-12 03:41:18] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.34s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.25s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.26s/it]


    2026-05-12 03:41:23,932 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 03:41:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:05,  2.23s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:05,  2.23s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:11,  1.30s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:11,  1.30s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.20it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.20it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.87it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.87it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.61it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.61it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.47it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.47it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.47it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.15it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.15it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.15it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.29it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.29it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.29it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.17it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.17it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.17it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.17it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.05it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.05it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.05it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.05it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.05it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.48it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.48it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.48it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.48it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.48it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.48it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 19.48it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.21it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.21it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.21it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.21it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.21it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.21it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.21it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 35.49it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 35.49it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 35.49it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 35.49it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 35.49it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 35.49it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 35.49it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 35.49it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 35.49it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 45.04it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 45.04it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 45.04it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 45.04it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 45.04it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 45.04it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 45.04it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 45.04it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:07<00:00, 49.91it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:07<00:00, 49.91it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:07<00:00, 49.91it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:07<00:00, 49.91it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:07<00:00, 49.91it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:07<00:00, 49.91it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:07<00:00, 49.91it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:07<00:00, 49.91it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=41.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=41.43 GB):   2%|▏         | 1/58 [00:00<00:17,  3.31it/s]Capturing num tokens (num_tokens=7680 avail_mem=41.39 GB):   2%|▏         | 1/58 [00:00<00:17,  3.31it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=41.39 GB):   3%|▎         | 2/58 [00:00<00:16,  3.48it/s]Capturing num tokens (num_tokens=7168 avail_mem=41.39 GB):   3%|▎         | 2/58 [00:00<00:16,  3.48it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=41.39 GB):   5%|▌         | 3/58 [00:00<00:14,  3.70it/s]Capturing num tokens (num_tokens=6656 avail_mem=41.39 GB):   5%|▌         | 3/58 [00:00<00:14,  3.70it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=41.39 GB):   7%|▋         | 4/58 [00:01<00:16,  3.22it/s]Capturing num tokens (num_tokens=6144 avail_mem=41.39 GB):   7%|▋         | 4/58 [00:01<00:16,  3.22it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=41.39 GB):   9%|▊         | 5/58 [00:01<00:14,  3.64it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.39 GB):   9%|▊         | 5/58 [00:01<00:14,  3.64it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.39 GB):  10%|█         | 6/58 [00:01<00:12,  4.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.39 GB):  10%|█         | 6/58 [00:01<00:12,  4.14it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=41.39 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.39 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.39 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.16it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.39 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.16it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=41.39 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.39 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.39 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.39 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.25it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=41.39 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.39 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.39 GB):  21%|██        | 12/58 [00:02<00:06,  7.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.32 GB):  21%|██        | 12/58 [00:02<00:06,  7.46it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=41.32 GB):  21%|██        | 12/58 [00:02<00:06,  7.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.32 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.32 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.32 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.66it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=41.32 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.31 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.31 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.31 GB):  31%|███       | 18/58 [00:02<00:03, 11.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.31 GB):  31%|███       | 18/58 [00:02<00:03, 11.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.31 GB):  31%|███       | 18/58 [00:02<00:03, 11.57it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=41.31 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.29 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.49it/s]Capturing num tokens (num_tokens=960 avail_mem=41.29 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.49it/s] Capturing num tokens (num_tokens=896 avail_mem=41.29 GB):  34%|███▍      | 20/58 [00:03<00:02, 13.49it/s]Capturing num tokens (num_tokens=896 avail_mem=41.29 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.73it/s]Capturing num tokens (num_tokens=832 avail_mem=41.28 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.73it/s]Capturing num tokens (num_tokens=768 avail_mem=41.28 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.73it/s]Capturing num tokens (num_tokens=704 avail_mem=41.27 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.73it/s]

    Capturing num tokens (num_tokens=704 avail_mem=41.27 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.58it/s]Capturing num tokens (num_tokens=640 avail_mem=41.27 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.58it/s]Capturing num tokens (num_tokens=576 avail_mem=41.27 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.58it/s]Capturing num tokens (num_tokens=512 avail_mem=41.26 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.58it/s]Capturing num tokens (num_tokens=480 avail_mem=41.26 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.58it/s]Capturing num tokens (num_tokens=480 avail_mem=41.26 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Capturing num tokens (num_tokens=448 avail_mem=41.26 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Capturing num tokens (num_tokens=416 avail_mem=41.25 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Capturing num tokens (num_tokens=384 avail_mem=41.25 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]

    Capturing num tokens (num_tokens=352 avail_mem=41.24 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Capturing num tokens (num_tokens=352 avail_mem=41.24 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.10it/s]Capturing num tokens (num_tokens=320 avail_mem=41.24 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.10it/s]Capturing num tokens (num_tokens=288 avail_mem=41.24 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.10it/s]Capturing num tokens (num_tokens=256 avail_mem=41.24 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.10it/s]Capturing num tokens (num_tokens=240 avail_mem=41.23 GB):  59%|█████▊    | 34/58 [00:03<00:00, 26.10it/s]Capturing num tokens (num_tokens=240 avail_mem=41.23 GB):  66%|██████▌   | 38/58 [00:03<00:00, 29.15it/s]Capturing num tokens (num_tokens=224 avail_mem=41.23 GB):  66%|██████▌   | 38/58 [00:03<00:00, 29.15it/s]

    Capturing num tokens (num_tokens=208 avail_mem=41.23 GB):  66%|██████▌   | 38/58 [00:03<00:00, 29.15it/s]Capturing num tokens (num_tokens=192 avail_mem=41.23 GB):  66%|██████▌   | 38/58 [00:03<00:00, 29.15it/s]Capturing num tokens (num_tokens=192 avail_mem=41.23 GB):  71%|███████   | 41/58 [00:03<00:00, 21.55it/s]Capturing num tokens (num_tokens=176 avail_mem=41.22 GB):  71%|███████   | 41/58 [00:03<00:00, 21.55it/s]Capturing num tokens (num_tokens=160 avail_mem=41.22 GB):  71%|███████   | 41/58 [00:03<00:00, 21.55it/s]Capturing num tokens (num_tokens=144 avail_mem=41.21 GB):  71%|███████   | 41/58 [00:03<00:00, 21.55it/s]Capturing num tokens (num_tokens=128 avail_mem=41.22 GB):  71%|███████   | 41/58 [00:03<00:00, 21.55it/s]Capturing num tokens (num_tokens=128 avail_mem=41.22 GB):  78%|███████▊  | 45/58 [00:03<00:00, 25.27it/s]Capturing num tokens (num_tokens=112 avail_mem=41.21 GB):  78%|███████▊  | 45/58 [00:03<00:00, 25.27it/s]Capturing num tokens (num_tokens=96 avail_mem=41.21 GB):  78%|███████▊  | 45/58 [00:03<00:00, 25.27it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=41.18 GB):  78%|███████▊  | 45/58 [00:03<00:00, 25.27it/s]Capturing num tokens (num_tokens=64 avail_mem=41.18 GB):  78%|███████▊  | 45/58 [00:03<00:00, 25.27it/s]Capturing num tokens (num_tokens=64 avail_mem=41.18 GB):  84%|████████▍ | 49/58 [00:04<00:00, 26.87it/s]Capturing num tokens (num_tokens=48 avail_mem=41.18 GB):  84%|████████▍ | 49/58 [00:04<00:00, 26.87it/s]Capturing num tokens (num_tokens=32 avail_mem=41.17 GB):  84%|████████▍ | 49/58 [00:04<00:00, 26.87it/s]Capturing num tokens (num_tokens=28 avail_mem=41.17 GB):  84%|████████▍ | 49/58 [00:04<00:00, 26.87it/s]Capturing num tokens (num_tokens=28 avail_mem=41.17 GB):  90%|████████▉ | 52/58 [00:04<00:00, 27.52it/s]Capturing num tokens (num_tokens=24 avail_mem=41.17 GB):  90%|████████▉ | 52/58 [00:04<00:00, 27.52it/s]

    Capturing num tokens (num_tokens=20 avail_mem=41.16 GB):  90%|████████▉ | 52/58 [00:04<00:00, 27.52it/s]Capturing num tokens (num_tokens=16 avail_mem=41.16 GB):  90%|████████▉ | 52/58 [00:04<00:00, 27.52it/s]Capturing num tokens (num_tokens=16 avail_mem=41.16 GB):  95%|█████████▍| 55/58 [00:04<00:00, 27.31it/s]Capturing num tokens (num_tokens=12 avail_mem=41.16 GB):  95%|█████████▍| 55/58 [00:04<00:00, 27.31it/s]Capturing num tokens (num_tokens=8 avail_mem=41.15 GB):  95%|█████████▍| 55/58 [00:04<00:00, 27.31it/s] Capturing num tokens (num_tokens=4 avail_mem=41.15 GB):  95%|█████████▍| 55/58 [00:04<00:00, 27.31it/s]Capturing num tokens (num_tokens=4 avail_mem=41.15 GB): 100%|██████████| 58/58 [00:04<00:00, 13.46it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think the population has been growing over the years. I recall that in recent years, Paris has been attracting a lot of immigrants and tourists, so the population might be higher than I initially thought. Maybe it's over 21 million now? I'm not sure if it's exactly 21.3 million or something else. I should look up the latest data to confirm.<br><br>I also wonder if the population figure is accurate for the current year. Population numbers can change yearly due to births, deaths, and migration. So, it's important to get the most recent statistics to provide an accurate answer. I think the best way is to reference a reliable source, like a government website or a recent publication, to get the exact figure.<br><br>After checking, I find that as of 2023, the population of Paris is approximately 21.3 million people. That seems to align with what I thought. So, I can confidently say that the capital of France is Paris, and its population is about 21.3 million.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21300000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants the current date and time along with the weather. Let me figure out how to approach this.<br><br>First, I need to determine which functions to use. The user mentioned two possibilities: get_current_weather and get_current_date. Both are available, so I can use either or both.<br><br>Starting with the date and time, I should use get_current_date. The function requires a timezone parameter. Since the user is in New York, I can specify the timezone as 'America/New_York'. I don't need any other parameters for this function, so the parameters object will just have timezone set to 'America/New_York'.<br><br>Next, for the weather, I'll use get_current_weather. This function needs city, state, and unit. The city is New York, but I should check if the state abbreviation is 'NY' because sometimes it's 'NYC' but I think the function expects the full state. So, city is 'New York', state is 'NY', and unit is probably Fahrenheit since that's the default, but the user didn't specify. So I'll include unit: 'fahrenheit' in the parameters.<br><br>I should structure the response by first calling get_current_date with the timezone, then get_current_weather with the necessary parameters. I'll make sure each function call is on its own line with the correct start and end tags, and include the parameters as JSON objects.<br><br>I also need to remember to add sources when using external information, but since I'm using built-in functions, there's no external data to cite. So I'll just format the response correctly as per the instructions, making sure each function call is properly formatted with the parameters included.<br><br>Putting it all together, I'll write the two function calls one after the other, each on their own line, using the correct syntax. That should cover the user's request accurately.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '859214a12c1e474498f2033a9cae4308', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 6.022315366193652, 'response_sent_to_client_ts': 1778557331.0216904}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '29b96796c3c040bd898b02820bd23f63', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.8648886419832706, 'response_sent_to_client_ts': 1778557334.8978598}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'd79b70b74f084daaa7da2e01c57fbe87', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09427502937614918, 'response_sent_to_client_ts': 1778557335.018362}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'cd5f318ca3bc4c1595ea184a2a020ece', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09420514106750488, 'response_sent_to_client_ts': 1778557335.018378}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '8ddfcc9858e54ca9814bf17194afa7ba', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09415309131145477, 'response_sent_to_client_ts': 1778557335.0184014}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '2b576d9152d045448e2414e9c3425fe0', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 20.947628993541002, 'response_sent_to_client_ts': 1778557355.97312}}


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


<strong style='color: #00008B;'>{'text': 'Alright, let\'s break down what the user is asking for. They want me to provide information about the capital of France, specifically its population, in JSON format. \n\nFirst, I need to identify the capital of France. That\'s straightforward—I know it\'s Paris. So, the key pieces of information here are the name of the city and its population.\n\nNow, I should consider the population figure. From what I remember, Paris has a sizable population, but I don\'t have the exact number memorized. I should probably look that up to ensure accuracy. I\'ll check a reliable source or maybe even calculate the approximate number to present it correctly.\n\nI should structure the JSON with a key-value pair. The key would be "capital," and the value would be another object containing "name" and "population." This makes it clear and easy to parse.\n\nAlso, I need to ensure the population number is correct because misinformation can be a problem. I\'ll confirm that the population is around 2.17 million as of 2023, but I\'ll mention it\'s subject to change due to factors like migration and natural growth to cover any uncertainties.\n\nFinally, I\'ll phrase it in a friendly way, offering further assistance if they need more details. That should cover everything the user is asking for.\n</think>\n\nHere is the information about the capital of France in JSON format:\n\n```json\n{\n  "capital": {\n    "name": "Paris",\n    "population": 2170000\n  }\n}\n```\n\nThis JSON object contains the name of the capital, "Paris," and its population, which is approximately 2,170,000 as of recent estimates. Let me know if you need more details!', 'output_ids': [71486, 11, 1077, 594, 1438, 1495, 1128, 279, 1196, 374, 10161, 369, 13, 2379, 1366, 752, 311, 3410, 1995, 911, 279, 6722, 315, 9625, 11, 11689, 1181, 7042, 11, 304, 4718, 3561, 13, 4710, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 2938, 594, 30339, 50657, 1414, 432, 594, 12095, 13, 2055, 11, 279, 1376, 9666, 315, 1995, 1588, 525, 279, 829, 315, 279, 3283, 323, 1181, 7042, 382, 7039, 11, 358, 1265, 2908, 279, 7042, 7071, 13, 5542, 1128, 358, 6099, 11, 12095, 702, 264, 78098, 7042, 11, 714, 358, 1513, 944, 614, 279, 4734, 1372, 16035, 1506, 13, 358, 1265, 4658, 1401, 429, 705, 311, 5978, 13403, 13, 358, 3278, 1779, 264, 14720, 2530, 476, 7196, 1496, 11047, 279, 44868, 1372, 311, 3042, 432, 12440, 382, 40, 1265, 5944, 279, 4718, 448, 264, 1376, 19083, 6716, 13, 576, 1376, 1035, 387, 330, 65063, 1335, 323, 279, 897, 1035, 387, 2441, 1633, 8482, 330, 606, 1, 323, 330, 44441, 1189, 1096, 3643, 432, 2797, 323, 4135, 311, 4715, 382, 13394, 11, 358, 1184, 311, 5978, 279, 7042, 1372, 374, 4396, 1576, 74059, 646, 387, 264, 3491, 13, 358, 3278, 7683, 429, 279, 7042, 374, 2163, 220, 17, 13, 16, 22, 3526, 438, 315, 220, 17, 15, 17, 18, 11, 714, 358, 3278, 6286, 432, 594, 3832, 311, 2297, 4152, 311, 9363, 1075, 11906, 323, 5810, 6513, 311, 3421, 894, 68767, 382, 23949, 11, 358, 3278, 17133, 432, 304, 264, 11657, 1616, 11, 10004, 4623, 12994, 421, 807, 1184, 803, 3565, 13, 2938, 1265, 3421, 4297, 279, 1196, 374, 10161, 369, 624, 151649, 271, 8420, 374, 279, 1995, 911, 279, 6722, 315, 9625, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 341, 262, 330, 606, 788, 330, 59604, 756, 262, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 198, 220, 456, 532, 13874, 19324, 1986, 4718, 1633, 5610, 279, 829, 315, 279, 6722, 11, 330, 59604, 1335, 323, 1181, 7042, 11, 892, 374, 13187, 220, 17, 11, 16, 22, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 13, 6771, 752, 1414, 421, 498, 1184, 803, 3565, 0, 151643], 'meta_info': {'id': '271c83361aac4e1bbef5920a59e7a039', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 266, 'completion_tokens': 359, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.0525597259402275, 'response_sent_to_client_ts': 1778557359.033636}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.28s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.29s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.29s/it]


    2026-05-12 03:42:54,475 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 03:42:54] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:07,  2.28s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:07,  2.28s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:13,  1.33s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:13,  1.33s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.14it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.14it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.62it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.84it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.84it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.58it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.58it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.13it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.13it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.13it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.32it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.32it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.32it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.25it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.25it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.25it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.25it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.99it/s]

    Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.99it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 35.24it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 35.24it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 35.24it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 35.24it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 35.24it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 35.24it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 35.24it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 35.24it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:07<00:00, 35.24it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 45.65it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 45.65it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:07<00:00, 45.65it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:07<00:00, 45.65it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:07<00:00, 45.65it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:07<00:00, 45.65it/s]

    Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:07<00:00, 45.65it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:07<00:00, 48.90it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:07<00:00, 48.90it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:07<00:00, 48.90it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:07<00:00, 48.90it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:07<00:00, 48.90it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:07<00:00, 48.90it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:07<00:00, 48.90it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:07<00:00, 48.90it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:07<00:00, 48.90it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:07<00:00, 57.43it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:07<00:00, 57.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=61.68 GB):   2%|▏         | 1/58 [00:00<00:16,  3.39it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.64 GB):   2%|▏         | 1/58 [00:00<00:16,  3.39it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=61.64 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.64 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=61.64 GB):   5%|▌         | 3/58 [00:00<00:15,  3.50it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.64 GB):   5%|▌         | 3/58 [00:00<00:15,  3.50it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=61.64 GB):   7%|▋         | 4/58 [00:01<00:13,  3.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.64 GB):   7%|▋         | 4/58 [00:01<00:13,  3.88it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.64 GB):   9%|▊         | 5/58 [00:01<00:12,  4.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.64 GB):   9%|▊         | 5/58 [00:01<00:12,  4.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.64 GB):  10%|█         | 6/58 [00:01<00:11,  4.62it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.64 GB):  10%|█         | 6/58 [00:01<00:11,  4.62it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=61.64 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.64 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.64 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.64 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.52it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=61.64 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.64 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.64 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.64 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.50it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=61.64 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.95it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.64 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.95it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.64 GB):  21%|██        | 12/58 [00:02<00:06,  7.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.63 GB):  21%|██        | 12/58 [00:02<00:06,  7.58it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=61.63 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.18it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.63 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.63 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.63 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.63 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.57it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=61.62 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.62 GB):  29%|██▉       | 17/58 [00:02<00:03, 11.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.62 GB):  29%|██▉       | 17/58 [00:02<00:03, 11.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.62 GB):  29%|██▉       | 17/58 [00:02<00:03, 11.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.62 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.62 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.75it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=61.61 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.75it/s]Capturing num tokens (num_tokens=960 avail_mem=61.60 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.75it/s] Capturing num tokens (num_tokens=960 avail_mem=61.60 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.94it/s]Capturing num tokens (num_tokens=896 avail_mem=61.60 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.94it/s]Capturing num tokens (num_tokens=832 avail_mem=61.60 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.94it/s]Capturing num tokens (num_tokens=768 avail_mem=61.59 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.94it/s]Capturing num tokens (num_tokens=768 avail_mem=61.59 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Capturing num tokens (num_tokens=704 avail_mem=61.59 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]

    Capturing num tokens (num_tokens=640 avail_mem=61.59 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Capturing num tokens (num_tokens=576 avail_mem=61.58 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.70it/s]Capturing num tokens (num_tokens=576 avail_mem=61.58 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.39it/s]Capturing num tokens (num_tokens=512 avail_mem=61.58 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.39it/s]Capturing num tokens (num_tokens=480 avail_mem=61.57 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.39it/s]Capturing num tokens (num_tokens=448 avail_mem=61.57 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.39it/s]Capturing num tokens (num_tokens=416 avail_mem=61.57 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.39it/s]Capturing num tokens (num_tokens=416 avail_mem=61.57 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]Capturing num tokens (num_tokens=384 avail_mem=61.56 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]

    Capturing num tokens (num_tokens=352 avail_mem=61.56 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]Capturing num tokens (num_tokens=320 avail_mem=61.55 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]Capturing num tokens (num_tokens=288 avail_mem=61.56 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]Capturing num tokens (num_tokens=288 avail_mem=61.56 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.48it/s]Capturing num tokens (num_tokens=256 avail_mem=61.55 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.48it/s]Capturing num tokens (num_tokens=240 avail_mem=61.55 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.48it/s]Capturing num tokens (num_tokens=224 avail_mem=61.55 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.48it/s]Capturing num tokens (num_tokens=208 avail_mem=61.54 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.48it/s]Capturing num tokens (num_tokens=208 avail_mem=61.54 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.12it/s]Capturing num tokens (num_tokens=192 avail_mem=61.54 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.12it/s]

    Capturing num tokens (num_tokens=176 avail_mem=61.54 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.12it/s]Capturing num tokens (num_tokens=160 avail_mem=61.53 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.12it/s]Capturing num tokens (num_tokens=144 avail_mem=61.53 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.12it/s]Capturing num tokens (num_tokens=144 avail_mem=61.53 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.16it/s]Capturing num tokens (num_tokens=128 avail_mem=61.53 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.16it/s]Capturing num tokens (num_tokens=112 avail_mem=61.53 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.16it/s]Capturing num tokens (num_tokens=96 avail_mem=61.52 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.16it/s] Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.16it/s]Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.82it/s]

    Capturing num tokens (num_tokens=64 avail_mem=61.50 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.82it/s]Capturing num tokens (num_tokens=48 avail_mem=61.49 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.82it/s]Capturing num tokens (num_tokens=32 avail_mem=61.49 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.82it/s]Capturing num tokens (num_tokens=28 avail_mem=61.49 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.82it/s]Capturing num tokens (num_tokens=28 avail_mem=61.49 GB):  90%|████████▉ | 52/58 [00:03<00:00, 27.84it/s]Capturing num tokens (num_tokens=24 avail_mem=61.48 GB):  90%|████████▉ | 52/58 [00:03<00:00, 27.84it/s]Capturing num tokens (num_tokens=20 avail_mem=60.99 GB):  90%|████████▉ | 52/58 [00:03<00:00, 27.84it/s]

    Capturing num tokens (num_tokens=16 avail_mem=60.98 GB):  90%|████████▉ | 52/58 [00:03<00:00, 27.84it/s]Capturing num tokens (num_tokens=12 avail_mem=60.82 GB):  90%|████████▉ | 52/58 [00:03<00:00, 27.84it/s]Capturing num tokens (num_tokens=12 avail_mem=60.82 GB):  97%|█████████▋| 56/58 [00:03<00:00, 29.78it/s]Capturing num tokens (num_tokens=8 avail_mem=60.82 GB):  97%|█████████▋| 56/58 [00:03<00:00, 29.78it/s] Capturing num tokens (num_tokens=4 avail_mem=60.81 GB):  97%|█████████▋| 56/58 [00:04<00:00, 29.78it/s]Capturing num tokens (num_tokens=4 avail_mem=60.81 GB): 100%|██████████| 58/58 [00:04<00:00, 14.36it/s]


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
    
    Generated text: Alright, the user is asking for the information and population of the capital of France in JSON format. So, first, I need to recall what the capital of France is. Let me think... Oh yeah, it's Paris.
    
    Next, I should gather the population data. I remember that Paris is the most populous city in France, but the exact number can change each year. I think it's around 2 million people. Wait, is it exactly 2,175,000? I should double-check that. Oh, no, I think the latest figure is approximately 2,175,000 as of 2023. I should make sure to include that number in the JSON.
    
    Now, structuring this into JSON. The user wants a JSON format, so I'll create an object with a key like "capital". Under that, I'll have another object containing "name" and "population". That makes sense because it's a nested structure, which is common for such data.
    
    I should also consider the formatting. The population number should be an integer since it's a count, so I'll write it without any decimal places. The keys should be in lowercase to follow standard JSON conventions unless specified otherwise.
    
    Putting it all together, the JSON should look clean and clear. I'll make sure there are commas in the right places and that the structure is properly nested. No trailing commas to avoid syntax errors.
    
    Finally, I'll present the JSON without any additional text, just the data, as the user requested. That should fulfill their query effectively.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "capital": {
        "name": "Paris",
        "population": 2175000
      }
    }
    ```



```python
llm.shutdown()
```
