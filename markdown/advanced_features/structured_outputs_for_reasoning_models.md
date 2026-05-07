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


    [2026-05-07 18:06:47] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.21s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.23s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.23s/it]


    2026-05-07 18:06:52,915 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 18:06:52] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:45,  5.02s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:45,  5.02s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:02,  2.19s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:02,  2.19s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:10,  1.28s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:10,  1.28s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:31,  1.67it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:23,  2.19it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:23,  2.19it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.79it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.79it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.49it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.49it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.34it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.34it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.34it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.04it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.04it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.04it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.62it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.62it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.62it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.23it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.23it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.23it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.17it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.17it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.17it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.17it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.53it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.53it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.53it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.53it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.53it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.08it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.88it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.88it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.88it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.88it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.88it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.88it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.88it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.37it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.37it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.37it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.37it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.37it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.37it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.37it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.37it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.37it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 46.56it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 46.56it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 46.56it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 46.56it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 46.56it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 46.56it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 46.56it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 46.56it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 46.56it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 54.75it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 54.75it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 54.75it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 54.75it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 54.75it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 54.75it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 54.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=20.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=20.08 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=20.05 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=20.05 GB):   3%|▎         | 2/58 [00:00<00:15,  3.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=20.05 GB):   3%|▎         | 2/58 [00:00<00:15,  3.52it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=20.05 GB):   5%|▌         | 3/58 [00:00<00:14,  3.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=20.05 GB):   5%|▌         | 3/58 [00:00<00:14,  3.79it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=20.05 GB):   7%|▋         | 4/58 [00:01<00:13,  4.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=20.05 GB):   7%|▋         | 4/58 [00:01<00:13,  4.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=20.05 GB):   9%|▊         | 5/58 [00:01<00:12,  4.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=20.05 GB):   9%|▊         | 5/58 [00:01<00:12,  4.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=20.05 GB):  10%|█         | 6/58 [00:01<00:10,  4.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=20.05 GB):  10%|█         | 6/58 [00:01<00:10,  4.77it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=20.05 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=20.05 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=20.05 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=20.05 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.69it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=20.05 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=20.05 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=20.05 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=20.05 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.69it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=20.05 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=20.04 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=20.04 GB):  21%|██        | 12/58 [00:02<00:05,  7.82it/s]Capturing num tokens (num_tokens=3072 avail_mem=20.04 GB):  21%|██        | 12/58 [00:02<00:05,  7.82it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=20.04 GB):  21%|██        | 12/58 [00:02<00:05,  7.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=20.04 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=20.04 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=20.04 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.95it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=20.04 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=20.03 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=20.03 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=20.03 GB):  31%|███       | 18/58 [00:02<00:03, 11.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=20.03 GB):  31%|███       | 18/58 [00:02<00:03, 11.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=20.03 GB):  31%|███       | 18/58 [00:02<00:03, 11.81it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=20.01 GB):  31%|███       | 18/58 [00:02<00:03, 11.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=20.01 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s]Capturing num tokens (num_tokens=960 avail_mem=20.01 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s] Capturing num tokens (num_tokens=896 avail_mem=20.01 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s]Capturing num tokens (num_tokens=832 avail_mem=20.00 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.70it/s]Capturing num tokens (num_tokens=832 avail_mem=20.00 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.67it/s]Capturing num tokens (num_tokens=768 avail_mem=20.00 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.67it/s]Capturing num tokens (num_tokens=704 avail_mem=19.99 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.67it/s]

    Capturing num tokens (num_tokens=640 avail_mem=19.99 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.67it/s]Capturing num tokens (num_tokens=640 avail_mem=19.99 GB):  47%|████▋     | 27/58 [00:02<00:01, 20.65it/s]Capturing num tokens (num_tokens=576 avail_mem=19.99 GB):  47%|████▋     | 27/58 [00:02<00:01, 20.65it/s]Capturing num tokens (num_tokens=512 avail_mem=19.98 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.65it/s]Capturing num tokens (num_tokens=480 avail_mem=19.98 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.65it/s]Capturing num tokens (num_tokens=448 avail_mem=19.98 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.65it/s]Capturing num tokens (num_tokens=448 avail_mem=19.98 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.87it/s]Capturing num tokens (num_tokens=416 avail_mem=19.98 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.87it/s]Capturing num tokens (num_tokens=384 avail_mem=19.97 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.87it/s]

    Capturing num tokens (num_tokens=352 avail_mem=19.97 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.87it/s]Capturing num tokens (num_tokens=352 avail_mem=19.97 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.33it/s]Capturing num tokens (num_tokens=320 avail_mem=19.96 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.33it/s]Capturing num tokens (num_tokens=288 avail_mem=19.97 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.33it/s]Capturing num tokens (num_tokens=256 avail_mem=19.96 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.33it/s]Capturing num tokens (num_tokens=240 avail_mem=19.96 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.33it/s]Capturing num tokens (num_tokens=240 avail_mem=19.96 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.49it/s]Capturing num tokens (num_tokens=224 avail_mem=19.96 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.49it/s]Capturing num tokens (num_tokens=208 avail_mem=19.95 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.49it/s]Capturing num tokens (num_tokens=192 avail_mem=19.95 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.49it/s]

    Capturing num tokens (num_tokens=176 avail_mem=19.94 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.49it/s]Capturing num tokens (num_tokens=176 avail_mem=19.94 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.24it/s]Capturing num tokens (num_tokens=160 avail_mem=19.94 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.24it/s]Capturing num tokens (num_tokens=144 avail_mem=19.94 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.24it/s]Capturing num tokens (num_tokens=128 avail_mem=19.94 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.24it/s]Capturing num tokens (num_tokens=112 avail_mem=19.94 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.24it/s]Capturing num tokens (num_tokens=112 avail_mem=19.94 GB):  79%|███████▉  | 46/58 [00:03<00:00, 33.29it/s]Capturing num tokens (num_tokens=96 avail_mem=19.93 GB):  79%|███████▉  | 46/58 [00:03<00:00, 33.29it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=18.99 GB):  79%|███████▉  | 46/58 [00:03<00:00, 33.29it/s]Capturing num tokens (num_tokens=64 avail_mem=18.98 GB):  79%|███████▉  | 46/58 [00:03<00:00, 33.29it/s]

    Capturing num tokens (num_tokens=48 avail_mem=18.98 GB):  79%|███████▉  | 46/58 [00:03<00:00, 33.29it/s]Capturing num tokens (num_tokens=48 avail_mem=18.98 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.45it/s]Capturing num tokens (num_tokens=32 avail_mem=18.98 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.45it/s]Capturing num tokens (num_tokens=28 avail_mem=18.98 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.45it/s]Capturing num tokens (num_tokens=24 avail_mem=18.97 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.45it/s]Capturing num tokens (num_tokens=20 avail_mem=18.97 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.45it/s]Capturing num tokens (num_tokens=16 avail_mem=18.97 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.45it/s]Capturing num tokens (num_tokens=16 avail_mem=18.97 GB):  95%|█████████▍| 55/58 [00:03<00:00, 25.80it/s]Capturing num tokens (num_tokens=12 avail_mem=18.96 GB):  95%|█████████▍| 55/58 [00:03<00:00, 25.80it/s]Capturing num tokens (num_tokens=8 avail_mem=18.96 GB):  95%|█████████▍| 55/58 [00:04<00:00, 25.80it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=18.96 GB):  95%|█████████▍| 55/58 [00:04<00:00, 25.80it/s]Capturing num tokens (num_tokens=4 avail_mem=18.96 GB): 100%|██████████| 58/58 [00:04<00:00, 14.28it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the current date and time in New York and the weather there. I need to figure out how to get this using the functions provided.<br><br>First, I remember that there's a function called get_current_date which requires a timezone parameter. The user is in New York, so the timezone would be 'America/New_York'. I should call this function with that parameter.<br><br>Next, for the weather, I need to use get_current_weather. The city is New York, the state is NY, and since the user didn't specify, I'll default to Celsius. So I'll structure the function call with these parameters.<br><br>I should make sure to format each function call correctly, using the required parameters and enclosing them in the specified JSON format. Also, I need to include the source links for each function in the response to show where the information is coming from.<br><br>Putting it all together, I'll send two separate function calls: one for the date and time, and another for the weather, each with their respective parameters. That way, the user gets both pieces of information they asked for.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function><br><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "celsius"}</function><br><br>Source: <br>- [get_current_date](https:///docs.example.com/python-functions#current-date)<br>- [get_current_weather](https://docs.example.com/python-functions#current-weather)</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'd48fc15ce2d0486582dcf726c127216d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.110017051920295, 'response_sent_to_client_ts': 1778177253.0716546}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'b78284863c8a457894c3d96815eb2116', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.384525207104161, 'response_sent_to_client_ts': 1778177257.4649086}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'e09bd9619b6e48f0a61cfa78d7205150', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.098677878966555, 'response_sent_to_client_ts': 1778177257.5881996}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '4aa55d09636447eeaef1dac3063115a6', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09837914491072297, 'response_sent_to_client_ts': 1778177257.5882146}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'c7ae84f7f4ea44018f1b143e592b34af', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09825733886100352, 'response_sent_to_client_ts': 1778177257.5882208}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '93074917f1d7490186378712165d0adf', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 17.056345706107095, 'response_sent_to_client_ts': 1778177274.651385}}


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


<strong style='color: #00008B;'>{'text': 'Okay, the user just asked for the population and information about the capital of France, specifically in JSON format. Let me break this down. They want the information in a structured format, probably for easy access or integration into another system.\n\nFirst, I should identify the capital\'s name, which is Paris. Then, I need to find its population figure. I remember it\'s in the hundreds of millions, maybe around 21 million people. But I should double-check that to make sure it\'s accurate. Also, Paris is known for a lot of famous landmarks. Maybe mentioning places like the Eiffel Tower, the Louvre, and Napoleon\'s tomb could add some context.\n\nI should structure this information neatly in JSON. The keys could be "name", "population", and a few other relevant details. Maybe include the location within France, the main attractions, and perhaps who the mayor is, even if it\'s an approximate name like Charlie M pillow cause I can\'t remember the exact name right now.\n\nWait, the exact population might change, so I should present it as an approximate value. Also, I remember the current mayor as Charles Peur不幸的是，我也记混了一些信息，所以需要确保准确无误。嗯，所以正确的 mayor name应该是 Charlesneider?\n</think>\n\nCertainly! Here is the information about the capital of France, Paris, presented in JSON format:\n\n```json\n{\n  "name": "Paris",\n  "population": 21500000,\n  "facts": {\n    "area": "City",\n    "coordinates": {\n      "latitude": "48.8566",\n      "longitude": "-2.3522"\n    },\n    "famous Landmarks": [\n      "Eiffel Tower",\n      "Louvre Museum",\n      "Napoleon\'s Tomb (Cemphet)"\n    ],\n    " mayor": "Charles-bedroom Charlesneider"\n  }\n}\n```\n\nLet me know if you need further details!', 'output_ids': [32313, 11, 279, 1196, 1101, 4588, 369, 279, 7042, 323, 1995, 911, 279, 6722, 315, 9625, 11, 11689, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 2379, 1366, 279, 1995, 304, 264, 32930, 3561, 11, 4658, 369, 4135, 2615, 476, 17590, 1119, 2441, 1849, 382, 5338, 11, 358, 1265, 10542, 279, 6722, 594, 829, 11, 892, 374, 12095, 13, 5005, 11, 358, 1184, 311, 1477, 1181, 7042, 7071, 13, 358, 6099, 432, 594, 304, 279, 11499, 315, 11728, 11, 7196, 2163, 220, 17, 16, 3526, 1251, 13, 1988, 358, 1265, 1990, 15934, 429, 311, 1281, 2704, 432, 594, 13382, 13, 7281, 11, 12095, 374, 3881, 369, 264, 2696, 315, 11245, 59924, 13, 10696, 44291, 7482, 1075, 279, 468, 3092, 301, 21938, 11, 279, 9729, 48506, 11, 323, 69427, 594, 43080, 1410, 912, 1045, 2266, 382, 40, 1265, 5944, 419, 1995, 62166, 304, 4718, 13, 576, 6894, 1410, 387, 330, 606, 497, 330, 44441, 497, 323, 264, 2421, 1008, 9760, 3565, 13, 10696, 2924, 279, 3728, 2878, 9625, 11, 279, 1887, 38491, 11, 323, 8365, 879, 279, 16923, 374, 11, 1496, 421, 432, 594, 458, 44868, 829, 1075, 24927, 386, 42325, 5240, 358, 646, 944, 6099, 279, 4734, 829, 1290, 1431, 382, 14190, 11, 279, 4734, 7042, 2578, 2297, 11, 773, 358, 1265, 3042, 432, 438, 458, 44868, 897, 13, 7281, 11, 358, 6099, 279, 1482, 16923, 438, 14927, 5139, 324, 107364, 100146, 3837, 104284, 40814, 100235, 105593, 27369, 3837, 99999, 85106, 103944, 102188, 42192, 29056, 1773, 106287, 3837, 99999, 105045, 16923, 829, 104583, 14927, 47611, 5267, 151649, 271, 95456, 0, 5692, 374, 279, 1995, 911, 279, 6722, 315, 9625, 11, 12095, 11, 10449, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 15, 345, 220, 330, 68053, 788, 341, 262, 330, 4798, 788, 330, 12730, 756, 262, 330, 34739, 788, 341, 414, 330, 23718, 788, 330, 19, 23, 13, 23, 20, 21, 21, 756, 414, 330, 25446, 788, 6523, 17, 13, 18, 20, 17, 17, 698, 262, 1153, 262, 330, 69, 22517, 11426, 15544, 788, 2278, 414, 330, 36, 3092, 301, 21938, 756, 414, 330, 92806, 48506, 16328, 756, 414, 330, 45, 391, 59103, 594, 56172, 320, 34, 336, 21951, 12954, 262, 3211, 262, 330, 16923, 788, 330, 53467, 66867, 14927, 47611, 698, 220, 456, 532, 13874, 19324, 10061, 752, 1414, 421, 498, 1184, 4623, 3565, 0, 151643], 'meta_info': {'id': 'effb74e97fa547fab04105b9e7c93205', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 259, 'completion_tokens': 408, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.036201623035595, 'response_sent_to_client_ts': 1778177279.6960163}}</strong>



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


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.27s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.29s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.28s/it]


    2026-05-07 18:08:15,364 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 18:08:15] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:04,  2.23s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:04,  2.23s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:11,  1.30s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:11,  1.30s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.22it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.22it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.89it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.89it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.64it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.64it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.21it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.21it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.21it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.79it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.79it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.79it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.42it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.42it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.42it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.39it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.39it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.39it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.39it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.71it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.71it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.71it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.71it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.71it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 20.23it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 20.23it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 20.23it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 20.23it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.23it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.23it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.23it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 29.24it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 29.24it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 29.24it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 29.24it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 29.24it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 29.24it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 29.24it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.84it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.84it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.84it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.84it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.84it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.84it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.84it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.84it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.84it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 47.15it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 47.15it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 47.15it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 47.15it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 47.15it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 47.15it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 47.15it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 47.15it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 47.15it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 55.81it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 55.81it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 55.81it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 55.81it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 55.81it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 55.81it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 55.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=42.03 GB):   2%|▏         | 1/58 [00:00<00:16,  3.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.00 GB):   2%|▏         | 1/58 [00:00<00:16,  3.47it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.00 GB):   3%|▎         | 2/58 [00:00<00:15,  3.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.00 GB):   3%|▎         | 2/58 [00:00<00:15,  3.64it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.00 GB):   5%|▌         | 3/58 [00:00<00:14,  3.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.00 GB):   5%|▌         | 3/58 [00:00<00:14,  3.87it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.00 GB):   7%|▋         | 4/58 [00:01<00:12,  4.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.00 GB):   7%|▋         | 4/58 [00:01<00:12,  4.16it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.00 GB):   9%|▊         | 5/58 [00:01<00:11,  4.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.00 GB):   9%|▊         | 5/58 [00:01<00:11,  4.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.00 GB):  10%|█         | 6/58 [00:01<00:11,  4.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.94 GB):  10%|█         | 6/58 [00:01<00:11,  4.69it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=41.94 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.89 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.89 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.16it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.85 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.16it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=41.85 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.84 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.52it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=41.84 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.34 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.12it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=41.34 GB):  19%|█▉        | 11/58 [00:02<00:09,  4.87it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.34 GB):  19%|█▉        | 11/58 [00:02<00:09,  4.87it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=41.34 GB):  21%|██        | 12/58 [00:02<00:09,  4.86it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.33 GB):  21%|██        | 12/58 [00:02<00:09,  4.86it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=41.33 GB):  22%|██▏       | 13/58 [00:02<00:09,  4.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.33 GB):  22%|██▏       | 13/58 [00:02<00:09,  4.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.33 GB):  24%|██▍       | 14/58 [00:02<00:07,  5.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.33 GB):  24%|██▍       | 14/58 [00:02<00:07,  5.53it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=41.33 GB):  24%|██▍       | 14/58 [00:02<00:07,  5.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.33 GB):  28%|██▊       | 16/58 [00:03<00:06,  6.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.63 GB):  28%|██▊       | 16/58 [00:03<00:06,  6.28it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=61.63 GB):  28%|██▊       | 16/58 [00:03<00:06,  6.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.63 GB):  31%|███       | 18/58 [00:03<00:04,  8.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.62 GB):  31%|███       | 18/58 [00:03<00:04,  8.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.62 GB):  31%|███       | 18/58 [00:03<00:04,  8.20it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=61.62 GB):  34%|███▍      | 20/58 [00:03<00:03,  9.60it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.61 GB):  34%|███▍      | 20/58 [00:03<00:03,  9.60it/s]Capturing num tokens (num_tokens=960 avail_mem=61.60 GB):  34%|███▍      | 20/58 [00:03<00:03,  9.60it/s] Capturing num tokens (num_tokens=960 avail_mem=61.60 GB):  38%|███▊      | 22/58 [00:03<00:03, 11.42it/s]Capturing num tokens (num_tokens=896 avail_mem=61.60 GB):  38%|███▊      | 22/58 [00:03<00:03, 11.42it/s]Capturing num tokens (num_tokens=832 avail_mem=61.60 GB):  38%|███▊      | 22/58 [00:03<00:03, 11.42it/s]

    Capturing num tokens (num_tokens=832 avail_mem=61.60 GB):  41%|████▏     | 24/58 [00:03<00:02, 13.03it/s]Capturing num tokens (num_tokens=768 avail_mem=61.59 GB):  41%|████▏     | 24/58 [00:03<00:02, 13.03it/s]Capturing num tokens (num_tokens=704 avail_mem=61.59 GB):  41%|████▏     | 24/58 [00:03<00:02, 13.03it/s]Capturing num tokens (num_tokens=640 avail_mem=61.59 GB):  41%|████▏     | 24/58 [00:03<00:02, 13.03it/s]Capturing num tokens (num_tokens=640 avail_mem=61.59 GB):  47%|████▋     | 27/58 [00:03<00:02, 14.87it/s]Capturing num tokens (num_tokens=576 avail_mem=61.58 GB):  47%|████▋     | 27/58 [00:03<00:02, 14.87it/s]

    Capturing num tokens (num_tokens=512 avail_mem=61.58 GB):  47%|████▋     | 27/58 [00:03<00:02, 14.87it/s]Capturing num tokens (num_tokens=480 avail_mem=61.58 GB):  47%|████▋     | 27/58 [00:03<00:02, 14.87it/s]Capturing num tokens (num_tokens=480 avail_mem=61.58 GB):  52%|█████▏    | 30/58 [00:03<00:01, 17.51it/s]Capturing num tokens (num_tokens=448 avail_mem=61.57 GB):  52%|█████▏    | 30/58 [00:03<00:01, 17.51it/s]Capturing num tokens (num_tokens=416 avail_mem=61.57 GB):  52%|█████▏    | 30/58 [00:04<00:01, 17.51it/s]Capturing num tokens (num_tokens=384 avail_mem=61.57 GB):  52%|█████▏    | 30/58 [00:04<00:01, 17.51it/s]Capturing num tokens (num_tokens=384 avail_mem=61.57 GB):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]Capturing num tokens (num_tokens=352 avail_mem=61.56 GB):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]

    Capturing num tokens (num_tokens=320 avail_mem=61.56 GB):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]Capturing num tokens (num_tokens=288 avail_mem=61.56 GB):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]Capturing num tokens (num_tokens=256 avail_mem=61.56 GB):  57%|█████▋    | 33/58 [00:04<00:01, 19.56it/s]Capturing num tokens (num_tokens=256 avail_mem=61.56 GB):  64%|██████▍   | 37/58 [00:04<00:00, 23.80it/s]Capturing num tokens (num_tokens=240 avail_mem=61.55 GB):  64%|██████▍   | 37/58 [00:04<00:00, 23.80it/s]Capturing num tokens (num_tokens=224 avail_mem=61.55 GB):  64%|██████▍   | 37/58 [00:04<00:00, 23.80it/s]Capturing num tokens (num_tokens=208 avail_mem=61.54 GB):  64%|██████▍   | 37/58 [00:04<00:00, 23.80it/s]Capturing num tokens (num_tokens=192 avail_mem=61.54 GB):  64%|██████▍   | 37/58 [00:04<00:00, 23.80it/s]Capturing num tokens (num_tokens=192 avail_mem=61.54 GB):  71%|███████   | 41/58 [00:04<00:00, 27.40it/s]Capturing num tokens (num_tokens=176 avail_mem=61.54 GB):  71%|███████   | 41/58 [00:04<00:00, 27.40it/s]

    Capturing num tokens (num_tokens=160 avail_mem=61.54 GB):  71%|███████   | 41/58 [00:04<00:00, 27.40it/s]Capturing num tokens (num_tokens=144 avail_mem=61.53 GB):  71%|███████   | 41/58 [00:04<00:00, 27.40it/s]Capturing num tokens (num_tokens=128 avail_mem=61.53 GB):  71%|███████   | 41/58 [00:04<00:00, 27.40it/s]Capturing num tokens (num_tokens=128 avail_mem=61.53 GB):  78%|███████▊  | 45/58 [00:04<00:00, 30.14it/s]Capturing num tokens (num_tokens=112 avail_mem=61.53 GB):  78%|███████▊  | 45/58 [00:04<00:00, 30.14it/s]Capturing num tokens (num_tokens=96 avail_mem=61.52 GB):  78%|███████▊  | 45/58 [00:04<00:00, 30.14it/s] Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  78%|███████▊  | 45/58 [00:04<00:00, 30.14it/s]Capturing num tokens (num_tokens=64 avail_mem=61.52 GB):  78%|███████▊  | 45/58 [00:04<00:00, 30.14it/s]Capturing num tokens (num_tokens=64 avail_mem=61.52 GB):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Capturing num tokens (num_tokens=48 avail_mem=61.51 GB):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]

    Capturing num tokens (num_tokens=32 avail_mem=61.51 GB):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Capturing num tokens (num_tokens=28 avail_mem=61.51 GB):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Capturing num tokens (num_tokens=24 avail_mem=61.51 GB):  84%|████████▍ | 49/58 [00:04<00:00, 32.31it/s]Capturing num tokens (num_tokens=24 avail_mem=61.51 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.16it/s]Capturing num tokens (num_tokens=20 avail_mem=61.50 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.16it/s]Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.16it/s]Capturing num tokens (num_tokens=12 avail_mem=61.49 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.16it/s]Capturing num tokens (num_tokens=8 avail_mem=61.49 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.16it/s] Capturing num tokens (num_tokens=8 avail_mem=61.49 GB):  98%|█████████▊| 57/58 [00:04<00:00, 35.58it/s]Capturing num tokens (num_tokens=4 avail_mem=61.49 GB):  98%|█████████▊| 57/58 [00:04<00:00, 35.58it/s]

    Capturing num tokens (num_tokens=4 avail_mem=61.49 GB): 100%|██████████| 58/58 [00:04<00:00, 12.26it/s]


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
    
    Generated text: Okay, so I need to figure out how to respond to the user's request. They asked for the information and population of the capital of France in JSON format. First, I should confirm who the capital of France is. I know that Paris is the capital, so that's where we're focusing.
    
    Next, I need to gather accurate information about Paris's population. I'm not sure of the exact current number, but I remember it's around 2 million people. I should check if that's the most recent estimate or if it's from a specific year. Maybe I can recall that Paris is one of the largest cities in the world, so 2 million sounds right.
    
    Now, how to present this information in JSON. JSON requires key-value pairs, so I'll need to structure it with "City": "Paris" and "Population": 2000000. I should make sure to include both the name and the population number. 
    
    Wait, should I mention the units? Probably not necessary since the user just asked for population, so 2000000 is fine. Also, the user might need this for a quick data extraction, so accuracy is key. I don't want to provide outdated information.
    
    I should also consider if the user needs more details, like area or establishment year, but since they specifically asked for population, I'll stick to that. Keeping it simple and clear is probably best.
    
    So, putting it all together, the JSON structure would be:
    
    {
      "City": "Paris",
      "Population": 2000000
    }
    
    I think that covers what the user is asking for. I don't see any other data needed, so this should be sufficient.
    </think>
    
    ```json
    {
      "City": "Paris",
      "Population": 2000000
    }
    ```



```python
llm.shutdown()
```
