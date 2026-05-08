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


    [2026-05-08 03:15:31] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.57s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.54s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.55s/it]


    2026-05-08 03:15:37,982 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 03:15:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:53,  5.14s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:53,  5.14s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:05,  2.24s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:05,  2.24s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:11,  1.31s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:11,  1.31s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.64it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.20it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.20it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.87it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.87it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.61it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.61it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.47it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.47it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.47it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.16it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.16it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.16it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.74it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.74it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.74it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.35it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.35it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.35it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.29it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.29it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.29it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.29it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.61it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.61it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.61it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.61it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.61it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.97it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.97it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.97it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.97it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.97it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.97it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.97it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.40it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.40it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.40it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.40it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.40it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.40it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.40it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.40it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.40it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 46.75it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 46.75it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 46.75it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 46.75it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 46.75it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 46.75it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 46.75it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 46.75it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 46.75it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 55.38it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 55.38it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 55.38it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 55.38it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 55.38it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 55.38it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 55.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=61.68 GB):   2%|▏         | 1/58 [00:00<00:16,  3.39it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.65 GB):   2%|▏         | 1/58 [00:00<00:16,  3.39it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=61.65 GB):   3%|▎         | 2/58 [00:00<00:16,  3.43it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.62 GB):   3%|▎         | 2/58 [00:00<00:16,  3.43it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=61.62 GB):   5%|▌         | 3/58 [00:00<00:15,  3.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.97 GB):   5%|▌         | 3/58 [00:00<00:15,  3.57it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=60.97 GB):   7%|▋         | 4/58 [00:01<00:13,  3.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.97 GB):   7%|▋         | 4/58 [00:01<00:13,  3.93it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=60.97 GB):   9%|▊         | 5/58 [00:01<00:12,  4.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.97 GB):   9%|▊         | 5/58 [00:01<00:12,  4.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.97 GB):  10%|█         | 6/58 [00:01<00:11,  4.64it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.97 GB):  10%|█         | 6/58 [00:01<00:11,  4.64it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=60.97 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.97 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.97 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.97 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.54it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=60.97 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.97 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.97 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.97 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.52it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=60.97 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.97 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.97 GB):  21%|██        | 12/58 [00:02<00:05,  7.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=60.96 GB):  21%|██        | 12/58 [00:02<00:05,  7.68it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=60.96 GB):  21%|██        | 12/58 [00:02<00:05,  7.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.96 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.96 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.86it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=60.96 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.94 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.94 GB):  28%|██▊       | 16/58 [00:02<00:04,  8.84it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.94 GB):  28%|██▊       | 16/58 [00:02<00:04,  8.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.92 GB):  28%|██▊       | 16/58 [00:02<00:04,  8.84it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=60.92 GB):  31%|███       | 18/58 [00:02<00:03, 10.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.28 GB):  31%|███       | 18/58 [00:02<00:03, 10.59it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.28 GB):  31%|███       | 18/58 [00:02<00:03, 10.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.27 GB):  31%|███       | 18/58 [00:02<00:03, 10.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.27 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.16it/s]Capturing num tokens (num_tokens=960 avail_mem=60.26 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.16it/s] Capturing num tokens (num_tokens=896 avail_mem=60.26 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.16it/s]

    Capturing num tokens (num_tokens=832 avail_mem=60.26 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.16it/s]Capturing num tokens (num_tokens=832 avail_mem=60.26 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.44it/s]Capturing num tokens (num_tokens=768 avail_mem=60.25 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.44it/s]Capturing num tokens (num_tokens=704 avail_mem=60.25 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.44it/s]Capturing num tokens (num_tokens=640 avail_mem=60.24 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.44it/s]Capturing num tokens (num_tokens=576 avail_mem=60.24 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.44it/s]Capturing num tokens (num_tokens=576 avail_mem=60.24 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.36it/s]Capturing num tokens (num_tokens=512 avail_mem=60.23 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.36it/s]Capturing num tokens (num_tokens=480 avail_mem=60.23 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.36it/s]

    Capturing num tokens (num_tokens=448 avail_mem=59.17 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.36it/s]Capturing num tokens (num_tokens=416 avail_mem=59.17 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.36it/s]Capturing num tokens (num_tokens=416 avail_mem=59.17 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Capturing num tokens (num_tokens=384 avail_mem=59.17 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Capturing num tokens (num_tokens=352 avail_mem=59.00 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Capturing num tokens (num_tokens=320 avail_mem=45.19 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Capturing num tokens (num_tokens=288 avail_mem=45.20 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.65it/s]Capturing num tokens (num_tokens=288 avail_mem=45.20 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.66it/s]Capturing num tokens (num_tokens=256 avail_mem=45.19 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.66it/s]

    Capturing num tokens (num_tokens=240 avail_mem=45.19 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.66it/s]Capturing num tokens (num_tokens=224 avail_mem=45.19 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.66it/s]Capturing num tokens (num_tokens=208 avail_mem=45.18 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.66it/s]Capturing num tokens (num_tokens=208 avail_mem=45.18 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.76it/s]Capturing num tokens (num_tokens=192 avail_mem=45.18 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.76it/s]Capturing num tokens (num_tokens=176 avail_mem=45.18 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.76it/s]Capturing num tokens (num_tokens=160 avail_mem=45.17 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.76it/s]Capturing num tokens (num_tokens=144 avail_mem=45.17 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.76it/s]Capturing num tokens (num_tokens=144 avail_mem=45.17 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.15it/s]Capturing num tokens (num_tokens=128 avail_mem=45.17 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.15it/s]

    Capturing num tokens (num_tokens=112 avail_mem=45.17 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.15it/s]Capturing num tokens (num_tokens=96 avail_mem=45.16 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.15it/s] Capturing num tokens (num_tokens=80 avail_mem=45.16 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.15it/s]Capturing num tokens (num_tokens=64 avail_mem=45.15 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.15it/s]Capturing num tokens (num_tokens=64 avail_mem=45.15 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.61it/s]Capturing num tokens (num_tokens=48 avail_mem=45.15 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.61it/s]Capturing num tokens (num_tokens=32 avail_mem=45.15 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.61it/s]Capturing num tokens (num_tokens=28 avail_mem=45.15 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.61it/s]Capturing num tokens (num_tokens=24 avail_mem=45.14 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.61it/s]Capturing num tokens (num_tokens=20 avail_mem=45.14 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.61it/s]

    Capturing num tokens (num_tokens=20 avail_mem=45.14 GB):  93%|█████████▎| 54/58 [00:03<00:00, 37.43it/s]Capturing num tokens (num_tokens=16 avail_mem=45.14 GB):  93%|█████████▎| 54/58 [00:03<00:00, 37.43it/s]Capturing num tokens (num_tokens=12 avail_mem=45.13 GB):  93%|█████████▎| 54/58 [00:03<00:00, 37.43it/s]Capturing num tokens (num_tokens=8 avail_mem=45.13 GB):  93%|█████████▎| 54/58 [00:03<00:00, 37.43it/s] Capturing num tokens (num_tokens=4 avail_mem=45.13 GB):  93%|█████████▎| 54/58 [00:03<00:00, 37.43it/s]Capturing num tokens (num_tokens=4 avail_mem=45.13 GB): 100%|██████████| 58/58 [00:03<00:00, 14.71it/s]


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


<strong style='color: #00008B;'>reasoing_content: Alright, so the user is asking for the current date and time in New York and the weather there. Let me break down how I should approach this.<br><br>First, I need to figure out which functions to use. The user mentioned New York, so I should use the functions 'get_current_date' and 'get_current_weather'. <br><br>Starting with the date, I'll call 'get_current_date' with the timezone parameter set to 'America/New_York'. That should give me the current date and time in that location.<br><br>Next, for the weather, I'll call 'get_current_weather' with the city as 'New York' and the unit as 'fahrenheit' since the user didn't specify, but Fahrenheit is commonly used in the US. <br><br>I should make sure to structure each function call correctly, using the required parameters and formatting the JSON objects properly. Also, I'll need to include the sources for each function in the response, as per the instructions.<br><br>Putting it all together, I'll first reply with the date, then the weather, each on their own line with the appropriate function calls and sources.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br>Source: get_current_date function call  <br><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>  <br>Source: get_current_weather function call</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'de2567f8c8c0442cb1ecb67967e0827e', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.842149183154106, 'response_sent_to_client_ts': 1778210180.4844294}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': 'a342e423f08e4f6baedc0e63a9e6b993', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 16.447797088883817, 'response_sent_to_client_ts': 1778210196.9409993}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '7383382d0629450297a0e6d46b2041eb', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09531306568533182, 'response_sent_to_client_ts': 1778210197.0633373}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '8949a62e985c41d8b04f938970c0ab44', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09525469178333879, 'response_sent_to_client_ts': 1778210197.063351}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'bcddd429a6a1425ca3bd6743a749c84d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09521445399150252, 'response_sent_to_client_ts': 1778210197.0633552}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'de11bbeb93324b28b7d32acc29fd31f3', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.2234311029315, 'response_sent_to_client_ts': 1778210216.295746}}


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


<strong style='color: #00008B;'>{'text': 'Alright, so I need to figure out the capital of France and its population. Hmm, let me start by recalling some basic facts. I know that Paris is the capital of France, that much I\'m certain about. Now, regarding the population, I\'m a bit fuzzy. I remember it\'s a large city, but I\'m not sure of the exact number.\n\nI think it\'s around 20 million people. Maybe a bit more? I recall hearing that Paris is one of the most populous cities in Europe, so perhaps it\'s in the mid-20s. Wait, no, I think it\'s the most populous in France, but I\'m not exactly sure of the figure. Maybe I can think of related information. France\'s total population is roughly 40 million, right? So if Paris is the capital, it would make sense that it\'s a significant portion of that, maybe two-thirds or something. So two-thirds of 40 million would be about 26.6 million. That seems plausible.\n\nBut I should double-check. I remember reading somewhere that Paris has grown a lot over the years. But I don\'t have exact numbers on hand. Maybe I can think of other major cities in France, like Lyon or Marseille, but they are much smaller. So Paris being in the mid-20 million range is reasonable.\n\nI think I\'ll go with 21 million as an approximate figure for Paris’s population. That\'s an average number I\'ve come across in various sources. It\'s a bit on the lower side, but considering some projections, I suppose it\'s a manageable figure for a general-purpose JSON response.\n\nSo, putting it all together, the JSON should have a key for the capital and another for the population, each with their respective values. I need to make sure the syntax is correct, with keys enclosed in quotes and the overall structure wrapped in square brackets for an array. \n\nLet me write that down: \n\n{\n  "capital": "Paris",\n  "population": 21000000\n}\n\nBut wait, in JSON, the population should be a number, not a string, so yes, 21000000 without quotes. And the keys are in lowercase letters as required. \n\nI don\'t think I need to include units here because population is just a number, and "Paris" is straightforward as the capital. So this should be the correct JSON format for the answer. I hope I\'m right because sometimes population numbers can vary based on the source and the year. But for now, 21 million seems like a good approximation.\n</think>\n\nThe JSON representation of the capital and population of France is as follows:\n\n```json\n{\n  "capital": "Paris",\n  "population": 21000000\n}\n```', 'output_ids': [71486, 11, 773, 358, 1184, 311, 7071, 700, 279, 6722, 315, 9625, 323, 1181, 7042, 13, 88190, 11, 1077, 752, 1191, 553, 88646, 1045, 6770, 13064, 13, 358, 1414, 429, 12095, 374, 279, 6722, 315, 9625, 11, 429, 1753, 358, 2776, 3654, 911, 13, 4695, 11, 8826, 279, 7042, 11, 358, 2776, 264, 2699, 52733, 13, 358, 6099, 432, 594, 264, 3460, 3283, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 382, 40, 1744, 432, 594, 2163, 220, 17, 15, 3526, 1251, 13, 10696, 264, 2699, 803, 30, 358, 19091, 10778, 429, 12095, 374, 825, 315, 279, 1429, 94451, 9720, 304, 4505, 11, 773, 8365, 432, 594, 304, 279, 5099, 12, 17, 15, 82, 13, 13824, 11, 902, 11, 358, 1744, 432, 594, 279, 1429, 94451, 304, 9625, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 7071, 13, 10696, 358, 646, 1744, 315, 5435, 1995, 13, 9625, 594, 2790, 7042, 374, 17267, 220, 19, 15, 3526, 11, 1290, 30, 2055, 421, 12095, 374, 279, 6722, 11, 432, 1035, 1281, 5530, 429, 432, 594, 264, 5089, 13348, 315, 429, 11, 7196, 1378, 44626, 476, 2494, 13, 2055, 1378, 44626, 315, 220, 19, 15, 3526, 1035, 387, 911, 220, 17, 21, 13, 21, 3526, 13, 2938, 4977, 49334, 382, 3983, 358, 1265, 1990, 15934, 13, 358, 6099, 5290, 14696, 429, 12095, 702, 14700, 264, 2696, 916, 279, 1635, 13, 1988, 358, 1513, 944, 614, 4734, 5109, 389, 1424, 13, 10696, 358, 646, 1744, 315, 1008, 3598, 9720, 304, 9625, 11, 1075, 55201, 476, 87693, 11, 714, 807, 525, 1753, 9155, 13, 2055, 12095, 1660, 304, 279, 5099, 12, 17, 15, 3526, 2088, 374, 13276, 382, 40, 1744, 358, 3278, 728, 448, 220, 17, 16, 3526, 438, 458, 44868, 7071, 369, 12095, 748, 7042, 13, 2938, 594, 458, 5461, 1372, 358, 3003, 2525, 3941, 304, 5257, 8173, 13, 1084, 594, 264, 2699, 389, 279, 4722, 3108, 11, 714, 12831, 1045, 40479, 11, 358, 22477, 432, 594, 264, 70028, 7071, 369, 264, 4586, 58238, 4718, 2033, 382, 4416, 11, 10687, 432, 678, 3786, 11, 279, 4718, 1265, 614, 264, 1376, 369, 279, 6722, 323, 2441, 369, 279, 7042, 11, 1817, 448, 862, 19511, 2750, 13, 358, 1184, 311, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6894, 43810, 304, 17194, 323, 279, 8084, 5944, 19472, 304, 9334, 38929, 369, 458, 1334, 13, 4710, 10061, 752, 3270, 429, 1495, 25, 4710, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 15, 15, 15, 15, 15, 15, 198, 630, 3983, 3783, 11, 304, 4718, 11, 279, 7042, 1265, 387, 264, 1372, 11, 537, 264, 914, 11, 773, 9834, 11, 220, 17, 16, 15, 15, 15, 15, 15, 15, 2041, 17194, 13, 1597, 279, 6894, 525, 304, 42047, 11931, 438, 2567, 13, 4710, 40, 1513, 944, 1744, 358, 1184, 311, 2924, 8153, 1588, 1576, 7042, 374, 1101, 264, 1372, 11, 323, 330, 59604, 1, 374, 30339, 438, 279, 6722, 13, 2055, 419, 1265, 387, 279, 4396, 4718, 3561, 369, 279, 4226, 13, 358, 3900, 358, 2776, 1290, 1576, 7025, 7042, 5109, 646, 13289, 3118, 389, 279, 2530, 323, 279, 1042, 13, 1988, 369, 1431, 11, 220, 17, 16, 3526, 4977, 1075, 264, 1661, 56204, 624, 151649, 271, 785, 4718, 13042, 315, 279, 6722, 323, 7042, 315, 9625, 374, 438, 11017, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 15, 15, 15, 15, 15, 15, 198, 532, 73594, 151643], 'meta_info': {'id': 'b5e024e756bd4b5385616e469564c0a5', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 538, 'completion_tokens': 581, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 9.686974979937077, 'response_sent_to_client_ts': 1778210225.9919095}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.53s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.55s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.55s/it]


    2026-05-08 03:17:21,704 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 03:17:21] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:00,  5.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:00,  5.28s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:09,  2.30s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:09,  2.30s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:13,  1.34s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:13,  1.34s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:32,  1.61it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:32,  1.61it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.83it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.83it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.57it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.57it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.42it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.42it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.42it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.12it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.12it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.12it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.68it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.68it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.68it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.28it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.28it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.28it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.17it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.17it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.17it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.17it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.38it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.38it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.38it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.38it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.38it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.71it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.71it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.71it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.71it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.71it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.71it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.88it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.88it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.88it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.88it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.88it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.88it/s]

    Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.88it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 34.73it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 34.73it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 34.73it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 34.73it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 34.73it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 34.73it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 34.73it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 34.73it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:07<00:00, 43.51it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:07<00:00, 43.51it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:07<00:00, 43.51it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:07<00:00, 43.51it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:07<00:00, 43.51it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:07<00:00, 43.51it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:07<00:00, 43.51it/s]

    Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:07<00:00, 43.51it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:07<00:00, 49.64it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:07<00:00, 49.64it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:07<00:00, 49.64it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:07<00:00, 49.64it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:07<00:00, 49.64it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:07<00:00, 49.64it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:07<00:00, 49.64it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:07<00:00, 49.64it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:07<00:00, 49.64it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:07<00:00, 49.64it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00, 59.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=61.68 GB):   2%|▏         | 1/58 [00:00<00:16,  3.42it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.64 GB):   2%|▏         | 1/58 [00:00<00:16,  3.42it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=61.64 GB):   3%|▎         | 2/58 [00:00<00:15,  3.59it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.64 GB):   3%|▎         | 2/58 [00:00<00:15,  3.59it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=61.64 GB):   5%|▌         | 3/58 [00:00<00:14,  3.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.64 GB):   5%|▌         | 3/58 [00:00<00:14,  3.82it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=61.64 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.64 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.64 GB):   9%|▊         | 5/58 [00:01<00:12,  4.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.64 GB):   9%|▊         | 5/58 [00:01<00:12,  4.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.64 GB):  10%|█         | 6/58 [00:01<00:10,  4.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.64 GB):  10%|█         | 6/58 [00:01<00:10,  4.76it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=61.64 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.64 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.64 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.64 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.47it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=61.64 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.64 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.64 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.31it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.64 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.31it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=61.64 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.64 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.64 GB):  21%|██        | 12/58 [00:02<00:07,  6.43it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.63 GB):  21%|██        | 12/58 [00:02<00:07,  6.43it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=61.63 GB):  21%|██        | 12/58 [00:02<00:07,  6.43it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.63 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.87it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.63 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.87it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.63 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.87it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=61.63 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.29it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.62 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.62 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.62 GB):  31%|███       | 18/58 [00:02<00:03, 10.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.62 GB):  31%|███       | 18/58 [00:02<00:03, 10.95it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.62 GB):  31%|███       | 18/58 [00:02<00:03, 10.95it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=61.61 GB):  31%|███       | 18/58 [00:02<00:03, 10.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.61 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.93it/s]Capturing num tokens (num_tokens=960 avail_mem=61.60 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.93it/s] Capturing num tokens (num_tokens=896 avail_mem=61.60 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.93it/s]Capturing num tokens (num_tokens=832 avail_mem=61.60 GB):  36%|███▌      | 21/58 [00:03<00:02, 13.93it/s]Capturing num tokens (num_tokens=832 avail_mem=61.60 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.56it/s]Capturing num tokens (num_tokens=768 avail_mem=61.59 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.56it/s]

    Capturing num tokens (num_tokens=704 avail_mem=61.59 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.56it/s]Capturing num tokens (num_tokens=640 avail_mem=61.59 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.56it/s]Capturing num tokens (num_tokens=640 avail_mem=61.59 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.45it/s]Capturing num tokens (num_tokens=576 avail_mem=61.58 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.45it/s]Capturing num tokens (num_tokens=512 avail_mem=61.58 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.45it/s]Capturing num tokens (num_tokens=480 avail_mem=61.57 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.45it/s]Capturing num tokens (num_tokens=448 avail_mem=61.57 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.45it/s]Capturing num tokens (num_tokens=448 avail_mem=61.57 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.77it/s]Capturing num tokens (num_tokens=416 avail_mem=61.57 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.77it/s]

    Capturing num tokens (num_tokens=384 avail_mem=61.56 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.77it/s]Capturing num tokens (num_tokens=352 avail_mem=61.56 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.77it/s]Capturing num tokens (num_tokens=320 avail_mem=61.55 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.77it/s]Capturing num tokens (num_tokens=320 avail_mem=61.55 GB):  60%|██████    | 35/58 [00:03<00:00, 25.97it/s]Capturing num tokens (num_tokens=288 avail_mem=61.56 GB):  60%|██████    | 35/58 [00:03<00:00, 25.97it/s]Capturing num tokens (num_tokens=256 avail_mem=61.55 GB):  60%|██████    | 35/58 [00:03<00:00, 25.97it/s]Capturing num tokens (num_tokens=240 avail_mem=61.55 GB):  60%|██████    | 35/58 [00:03<00:00, 25.97it/s]Capturing num tokens (num_tokens=224 avail_mem=61.55 GB):  60%|██████    | 35/58 [00:03<00:00, 25.97it/s]Capturing num tokens (num_tokens=224 avail_mem=61.55 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.68it/s]Capturing num tokens (num_tokens=208 avail_mem=61.54 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.68it/s]

    Capturing num tokens (num_tokens=192 avail_mem=61.54 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.68it/s]Capturing num tokens (num_tokens=176 avail_mem=61.54 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.68it/s]Capturing num tokens (num_tokens=160 avail_mem=61.53 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.68it/s]Capturing num tokens (num_tokens=160 avail_mem=61.53 GB):  74%|███████▍  | 43/58 [00:03<00:00, 29.43it/s]Capturing num tokens (num_tokens=144 avail_mem=61.53 GB):  74%|███████▍  | 43/58 [00:03<00:00, 29.43it/s]Capturing num tokens (num_tokens=128 avail_mem=61.53 GB):  74%|███████▍  | 43/58 [00:03<00:00, 29.43it/s]Capturing num tokens (num_tokens=112 avail_mem=61.53 GB):  74%|███████▍  | 43/58 [00:03<00:00, 29.43it/s]

    Capturing num tokens (num_tokens=112 avail_mem=61.53 GB):  79%|███████▉  | 46/58 [00:03<00:00, 28.13it/s]Capturing num tokens (num_tokens=96 avail_mem=61.52 GB):  79%|███████▉  | 46/58 [00:03<00:00, 28.13it/s] Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  79%|███████▉  | 46/58 [00:03<00:00, 28.13it/s]Capturing num tokens (num_tokens=64 avail_mem=61.51 GB):  79%|███████▉  | 46/58 [00:03<00:00, 28.13it/s]Capturing num tokens (num_tokens=48 avail_mem=61.51 GB):  79%|███████▉  | 46/58 [00:03<00:00, 28.13it/s]Capturing num tokens (num_tokens=48 avail_mem=61.51 GB):  86%|████████▌ | 50/58 [00:03<00:00, 30.78it/s]Capturing num tokens (num_tokens=32 avail_mem=61.51 GB):  86%|████████▌ | 50/58 [00:03<00:00, 30.78it/s]Capturing num tokens (num_tokens=28 avail_mem=61.51 GB):  86%|████████▌ | 50/58 [00:03<00:00, 30.78it/s]Capturing num tokens (num_tokens=24 avail_mem=61.50 GB):  86%|████████▌ | 50/58 [00:03<00:00, 30.78it/s]Capturing num tokens (num_tokens=20 avail_mem=61.50 GB):  86%|████████▌ | 50/58 [00:03<00:00, 30.78it/s]

    Capturing num tokens (num_tokens=20 avail_mem=61.50 GB):  93%|█████████▎| 54/58 [00:03<00:00, 32.53it/s]Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  93%|█████████▎| 54/58 [00:03<00:00, 32.53it/s]Capturing num tokens (num_tokens=12 avail_mem=61.49 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.53it/s]Capturing num tokens (num_tokens=8 avail_mem=61.49 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.53it/s] Capturing num tokens (num_tokens=4 avail_mem=61.49 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.53it/s]Capturing num tokens (num_tokens=4 avail_mem=61.49 GB): 100%|██████████| 58/58 [00:04<00:00, 32.02it/s]Capturing num tokens (num_tokens=4 avail_mem=61.49 GB): 100%|██████████| 58/58 [00:04<00:00, 14.10it/s]


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
    Generated text: Berlin is the capital of Germany
    ===============================
    Prompt: Give me the information of the capital of Germany.
    Generated text: Paris is the capital of France
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
    Generated text: England



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
    
    Generated text: Okay, the user asked for the information and population of the capital of France in JSON format. The capital is Paris. I should provide the population as of the most recent data available. I remember the population was around 2 million in 2020. I'll structure this in JSON with appropriate fields. Maybe include city name, country, country code, population, and population density. That seems comprehensive and useful for the user.
    </think>
    
    ```json
    {
      "capital": {
        "city": "Paris",
        "country": "France",
        "country_code": "FR",
        "population": 2192000,
        "population_density": 1100
      }
    }
    ```



```python
llm.shutdown()
```
