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


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:02<00:02,  2.76s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:05<00:00,  2.56s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:05<00:00,  2.59s/it]


    2026-05-17 06:14:31,002 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 06:14:31] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:01,  5.29s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:01,  5.29s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:25,  2.59s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:25,  2.59s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:33,  1.71s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:33,  1.71s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:08,  1.27s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:08,  1.27s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:52,  1.00it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:52,  1.00it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:08<00:43,  1.20it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:08<00:43,  1.20it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:35,  1.43it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:35,  1.43it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:09<00:30,  1.65it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:09<00:30,  1.65it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:25,  1.90it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:25,  1.90it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:22,  2.18it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:22,  2.18it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:10<00:19,  2.42it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:10<00:19,  2.42it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:10<00:16,  2.72it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:10<00:16,  2.72it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:10<00:14,  3.02it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:10<00:14,  3.02it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:10<00:13,  3.30it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:10<00:13,  3.30it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:11<00:11,  3.70it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:11<00:11,  3.70it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:11<00:10,  4.11it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:11<00:10,  4.11it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:11<00:08,  4.59it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:11<00:08,  4.59it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:11<00:07,  5.16it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:11<00:07,  5.16it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:11<00:06,  5.86it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:11<00:06,  5.86it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:11<00:06,  5.86it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:11<00:04,  7.48it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:11<00:04,  7.48it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:11<00:04,  7.48it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:11<00:03,  9.11it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:11<00:03,  9.11it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:11<00:03,  9.11it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:12<00:03, 10.66it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:12<00:03, 10.66it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:12<00:03, 10.66it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:12<00:02, 12.25it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:12<00:02, 12.25it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:12<00:02, 12.25it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:12<00:02, 12.25it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:12<00:01, 14.93it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:12<00:01, 14.93it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:12<00:01, 14.93it/s]

    Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:12<00:01, 15.97it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:12<00:01, 15.97it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:12<00:01, 15.97it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:12<00:01, 15.97it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:12<00:01, 18.80it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:12<00:01, 18.80it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:12<00:01, 18.80it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:12<00:01, 18.80it/s]

    Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:12<00:01, 18.80it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:12<00:00, 22.99it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:12<00:00, 22.99it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:12<00:00, 22.99it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:12<00:00, 22.99it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:12<00:00, 24.63it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:12<00:00, 24.63it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:12<00:00, 24.63it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:12<00:00, 24.63it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:12<00:00, 24.63it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:12<00:00, 27.30it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:12<00:00, 27.30it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:12<00:00, 27.30it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:12<00:00, 27.30it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:12<00:00, 27.30it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:13<00:00, 28.18it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:13<00:00, 28.18it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:13<00:00, 28.18it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:13<00:00, 28.18it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:13<00:00, 28.18it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:13<00:00, 31.15it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:13<00:00, 31.15it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:13<00:00, 31.15it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:13<00:00, 31.15it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:13<00:00, 31.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:13<00:00,  4.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.70 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.70 GB):   3%|▎         | 2/58 [00:01<00:49,  1.13it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   3%|▎         | 2/58 [00:01<00:49,  1.13it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   5%|▌         | 3/58 [00:02<00:45,  1.21it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.82 GB):   5%|▌         | 3/58 [00:02<00:45,  1.21it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.82 GB):   7%|▋         | 4/58 [00:03<00:41,  1.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.90 GB):   7%|▋         | 4/58 [00:03<00:41,  1.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.90 GB):   9%|▊         | 5/58 [00:03<00:38,  1.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.96 GB):   9%|▊         | 5/58 [00:03<00:38,  1.39it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.96 GB):  10%|█         | 6/58 [00:04<00:34,  1.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=26.02 GB):  10%|█         | 6/58 [00:04<00:34,  1.51it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=26.02 GB):  12%|█▏        | 7/58 [00:04<00:31,  1.63it/s]Capturing num tokens (num_tokens=4608 avail_mem=26.09 GB):  12%|█▏        | 7/58 [00:04<00:31,  1.63it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=26.09 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.16 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.81it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.16 GB):  16%|█▌        | 9/58 [00:05<00:24,  1.98it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.18 GB):  16%|█▌        | 9/58 [00:05<00:24,  1.98it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=26.18 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=26.21 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.14it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=26.21 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=26.24 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.34it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=26.24 GB):  21%|██        | 12/58 [00:06<00:17,  2.56it/s]Capturing num tokens (num_tokens=3072 avail_mem=26.26 GB):  21%|██        | 12/58 [00:06<00:17,  2.56it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=26.26 GB):  22%|██▏       | 13/58 [00:07<00:16,  2.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.29 GB):  22%|██▏       | 13/58 [00:07<00:16,  2.77it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.29 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.32 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.07it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.32 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.35 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.35 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.40 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.77it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=26.40 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.21it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.40 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.21it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.40 GB):  31%|███       | 18/58 [00:08<00:08,  4.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.62 GB):  31%|███       | 18/58 [00:08<00:08,  4.70it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=26.62 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.26it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.61 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.26it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.61 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.58 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.00it/s]

    Capturing num tokens (num_tokens=960 avail_mem=26.57 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.00it/s] Capturing num tokens (num_tokens=960 avail_mem=26.57 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.24it/s]Capturing num tokens (num_tokens=896 avail_mem=26.56 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.24it/s]

    Capturing num tokens (num_tokens=832 avail_mem=26.55 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.24it/s]Capturing num tokens (num_tokens=832 avail_mem=26.55 GB):  41%|████▏     | 24/58 [00:08<00:04,  8.47it/s]Capturing num tokens (num_tokens=768 avail_mem=26.53 GB):  41%|████▏     | 24/58 [00:08<00:04,  8.47it/s]Capturing num tokens (num_tokens=704 avail_mem=26.52 GB):  41%|████▏     | 24/58 [00:08<00:04,  8.47it/s]

    Capturing num tokens (num_tokens=704 avail_mem=26.52 GB):  45%|████▍     | 26/58 [00:08<00:03,  9.64it/s]Capturing num tokens (num_tokens=640 avail_mem=26.52 GB):  45%|████▍     | 26/58 [00:08<00:03,  9.64it/s]Capturing num tokens (num_tokens=576 avail_mem=26.51 GB):  45%|████▍     | 26/58 [00:08<00:03,  9.64it/s]Capturing num tokens (num_tokens=576 avail_mem=26.51 GB):  48%|████▊     | 28/58 [00:08<00:02, 10.78it/s]Capturing num tokens (num_tokens=512 avail_mem=26.49 GB):  48%|████▊     | 28/58 [00:08<00:02, 10.78it/s]

    Capturing num tokens (num_tokens=480 avail_mem=26.48 GB):  48%|████▊     | 28/58 [00:09<00:02, 10.78it/s]Capturing num tokens (num_tokens=480 avail_mem=26.48 GB):  52%|█████▏    | 30/58 [00:09<00:02, 12.06it/s]Capturing num tokens (num_tokens=448 avail_mem=26.47 GB):  52%|█████▏    | 30/58 [00:09<00:02, 12.06it/s]Capturing num tokens (num_tokens=416 avail_mem=26.46 GB):  52%|█████▏    | 30/58 [00:09<00:02, 12.06it/s]Capturing num tokens (num_tokens=416 avail_mem=26.46 GB):  55%|█████▌    | 32/58 [00:09<00:02, 12.95it/s]Capturing num tokens (num_tokens=384 avail_mem=26.45 GB):  55%|█████▌    | 32/58 [00:09<00:02, 12.95it/s]

    Capturing num tokens (num_tokens=352 avail_mem=26.43 GB):  55%|█████▌    | 32/58 [00:09<00:02, 12.95it/s]Capturing num tokens (num_tokens=352 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.17it/s]Capturing num tokens (num_tokens=320 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.17it/s]Capturing num tokens (num_tokens=288 avail_mem=26.44 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.17it/s]Capturing num tokens (num_tokens=288 avail_mem=26.44 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.46it/s]Capturing num tokens (num_tokens=256 avail_mem=26.42 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.46it/s]

    Capturing num tokens (num_tokens=240 avail_mem=26.41 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.46it/s]Capturing num tokens (num_tokens=224 avail_mem=26.40 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.46it/s]Capturing num tokens (num_tokens=224 avail_mem=26.40 GB):  67%|██████▋   | 39/58 [00:09<00:01, 17.11it/s]Capturing num tokens (num_tokens=208 avail_mem=26.39 GB):  67%|██████▋   | 39/58 [00:09<00:01, 17.11it/s]Capturing num tokens (num_tokens=192 avail_mem=26.38 GB):  67%|██████▋   | 39/58 [00:09<00:01, 17.11it/s]Capturing num tokens (num_tokens=176 avail_mem=26.37 GB):  67%|██████▋   | 39/58 [00:09<00:01, 17.11it/s]

    Capturing num tokens (num_tokens=176 avail_mem=26.37 GB):  72%|███████▏  | 42/58 [00:09<00:00, 18.24it/s]Capturing num tokens (num_tokens=160 avail_mem=26.36 GB):  72%|███████▏  | 42/58 [00:09<00:00, 18.24it/s]Capturing num tokens (num_tokens=144 avail_mem=26.35 GB):  72%|███████▏  | 42/58 [00:09<00:00, 18.24it/s]Capturing num tokens (num_tokens=128 avail_mem=26.35 GB):  72%|███████▏  | 42/58 [00:09<00:00, 18.24it/s]Capturing num tokens (num_tokens=128 avail_mem=26.35 GB):  78%|███████▊  | 45/58 [00:09<00:00, 19.04it/s]Capturing num tokens (num_tokens=112 avail_mem=26.34 GB):  78%|███████▊  | 45/58 [00:09<00:00, 19.04it/s]Capturing num tokens (num_tokens=96 avail_mem=26.33 GB):  78%|███████▊  | 45/58 [00:09<00:00, 19.04it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=26.31 GB):  78%|███████▊  | 45/58 [00:09<00:00, 19.04it/s]Capturing num tokens (num_tokens=80 avail_mem=26.31 GB):  83%|████████▎ | 48/58 [00:10<00:00, 19.77it/s]Capturing num tokens (num_tokens=64 avail_mem=26.31 GB):  83%|████████▎ | 48/58 [00:10<00:00, 19.77it/s]Capturing num tokens (num_tokens=48 avail_mem=26.30 GB):  83%|████████▎ | 48/58 [00:10<00:00, 19.77it/s]Capturing num tokens (num_tokens=32 avail_mem=26.29 GB):  83%|████████▎ | 48/58 [00:10<00:00, 19.77it/s]Capturing num tokens (num_tokens=32 avail_mem=26.29 GB):  88%|████████▊ | 51/58 [00:10<00:00, 20.43it/s]Capturing num tokens (num_tokens=28 avail_mem=26.28 GB):  88%|████████▊ | 51/58 [00:10<00:00, 20.43it/s]

    Capturing num tokens (num_tokens=24 avail_mem=26.27 GB):  88%|████████▊ | 51/58 [00:10<00:00, 20.43it/s]Capturing num tokens (num_tokens=20 avail_mem=26.26 GB):  88%|████████▊ | 51/58 [00:10<00:00, 20.43it/s]Capturing num tokens (num_tokens=20 avail_mem=26.26 GB):  93%|█████████▎| 54/58 [00:10<00:00, 21.13it/s]Capturing num tokens (num_tokens=16 avail_mem=26.25 GB):  93%|█████████▎| 54/58 [00:10<00:00, 21.13it/s]Capturing num tokens (num_tokens=12 avail_mem=26.23 GB):  93%|█████████▎| 54/58 [00:10<00:00, 21.13it/s]Capturing num tokens (num_tokens=8 avail_mem=26.23 GB):  93%|█████████▎| 54/58 [00:10<00:00, 21.13it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=26.23 GB):  98%|█████████▊| 57/58 [00:10<00:00, 21.52it/s]Capturing num tokens (num_tokens=4 avail_mem=26.22 GB):  98%|█████████▊| 57/58 [00:10<00:00, 21.52it/s]Capturing num tokens (num_tokens=4 avail_mem=26.22 GB): 100%|██████████| 58/58 [00:10<00:00,  5.54it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should check if that's correct.<br><br>Wait, I think the population might have changed a bit over the years. I recall reading somewhere that Paris has grown a lot, especially with the influx of people moving there for work. So maybe it's more than 20 million now. I'm trying to remember if it's closer to 21 or 22 million. I think it's around 21.5 million, but I'm not 100% sure. I should probably look up the latest data to confirm.<br><br>Looking it up, I see that as of 2023, the population of Paris is approximately 21,609,357. That's about 21.6 million. So I was close with my initial thought of around 21.5 million. I guess the number has increased a bit since I last heard about it. It makes sense because Paris is a major economic hub and attracts a lot of people for jobs and studies.<br><br>I should also consider the source of this information. Population numbers can vary depending on when they're measured and how they're reported. It's important to use reliable sources like official statistics from the French National Institute of Statistics and Registration (INSEE) or other reputable organizations that track population data.<br><br>So, putting it all together, the capital of France is Paris, and its population is approximately 21.6 million as of the latest data. I should present this information in a clear and concise JSON format, making sure to include both the city name and the population number accurately.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21609357<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think the population has been growing over the years. I recall that in recent years, Paris has been increasing its population. Maybe it's over 21 million now? I'm not sure if it's 2021 or 2022 data. Also, I should consider whether the population figure includes just the city proper or the entire metropolitan area. Sometimes, people talk about the metro area, which might be larger.<br><br>I think the metropolitan area of Paris is bigger, maybe around 12 million? But the city itself is smaller. I'm a bit confused about the exact numbers. I should look up the latest data to be accurate. But since I can't access the internet right now, I'll have to go with my best guess based on what I remember.<br><br>So, putting it all together, the capital is Paris, and the population is approximately 21 million. I'll present this information in JSON format as requested.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21000000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light." I've heard people talk about the Eiffel Tower being in Paris, which is a famous landmark. But is Paris the capital? I think it is, but I'm not entirely certain. Maybe I should consider other major cities in France. There's Lyon, which I think is the second largest city, and then there's Marseille. But I don't recall hearing them referred to as capitals. Then there's also the capital region, which includes Paris, but I believe the actual capital is just Paris itself. I don't think it's a region or a larger city. So, putting it all together, I'm pretty confident that Paris is the capital of France.<br><br><br>content: Paris</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants to know the current date and time as well as the weather. I need to figure out how to get both pieces of information using the functions available to me.<br><br>First, for the current date and time, I remember that there's a function called get_current_date. It requires a timezone parameter. Since the user is in New York, I should provide the timezone as 'America/New_York'. That should give me the correct date and time when I call the function.<br><br>Next, for the weather. The user mentioned they're in New York, so I can use the get_current_weather function. This function needs a city, state, and unit. The city is 'New York', the state would be 'NY' since that's the two-letter abbreviation, and the unit should be 'fahrenheit' if the user prefers Fahrenheit. I should make sure all the parameters are correctly formatted in an object.<br><br>Now, I need to structure the function calls properly. I'll start with get_current_date, include the timezone parameter, and then immediately follow it with get_current_weather, providing all the necessary details. I'll make sure each function call is on its own line with the correct syntax, using the required parameters as specified.<br><br>I should also remember to include the source in my response. Since I'm using these built-in functions, the source would be the assistant's knowledge or the predefined functions. So I'll add a line stating "Source: Assistant's knowledge of available functions."<br><br>Putting it all together, I'll write the two function calls one after the other, each properly formatted with their respective parameters. That should give the user exactly what they're asking for.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '5553f5333bcd4184acc38bee87e07190', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.17729683406651, 'response_sent_to_client_ts': 1778998544.7980487}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': 'db01e064171b4b8e8254a6936fc8db3e', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 21.211320956237614, 'response_sent_to_client_ts': 1778998566.0184917}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'ded33412e4634689abd563a421c1d8fd', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.10547596495598555, 'response_sent_to_client_ts': 1778998566.1604652}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '8992d3a206dd42ef8a47f79fe43dabb6', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.10541579872369766, 'response_sent_to_client_ts': 1778998566.1604786}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '2688ff79376b44a1b2b9b8e070b90b2b', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1053805137053132, 'response_sent_to_client_ts': 1778998566.1604838}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'a559b427b705465e8d8762192793c247', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.36381791997701, 'response_sent_to_client_ts': 1778998585.531702}}


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


<strong style='color: #00008B;'>{'text': 'Alright, I\'ve got a query here where the user is asking for the capital of France and its population. Wait, the user wrote "information" instead of "capital". That\'s a typo, but it\'s clear they want the capital. They also asked for the population in JSON format.\n\nFirst, I need to figure out who this user is. They might be a student, maybe in middle school or high school, studying geography or possibly looking for a quick fact to use somewhere else. Alternatively, they could be someone just curious about France\'s capital city.\n\nThey specified JSON format, so they might be using that data for programming or some kind of database where structured data is needed. Maybe they\'re building a project, integrating this info into another application, or just looking to display the population dynamically.\n\nI should make sure to get the correct information. The capital is definitely Paris, no doubt there. As for the population, I need to verify the latest numbers. I recall that Paris is the largest city in France, but its population might have changed a bit over the years. I think it\'s around 9 million people. I should double-check that to ensure accuracy.\n\nSince the user included "information and population", I should provide both pieces of data, but for the population, emphasize it as they asked. Also, keeping in mind that population figures can fluctuate, so it\'s good to mention the approximate range to show I understand uncertainty in stats.\n\nPutting it into JSON format, I\'ll structure it with a key for "capital" and "population", maybe as an object within a "data" field. That way, it\'s organized and easy for the user to read, especially if they\'re using this data in a script or app.\n\nI should keep the response concise but informative, just what they asked for without any extra fluff. No need for a lengthy description, just the necessary data in the correct format. That way, it\'s straightforward and meets their user request efficiently.\n</think>\n\nSure! Here\'s the information and population of the capital of France in JSON format:\n\n```json\n{\n  "capital": "Paris",\n  "population": 9063515\n}\n```', 'output_ids': [71486, 11, 358, 3003, 2684, 264, 3239, 1588, 1380, 279, 1196, 374, 10161, 369, 279, 6722, 315, 9625, 323, 1181, 7042, 13, 13824, 11, 279, 1196, 6139, 330, 25069, 1, 4518, 315, 330, 65063, 3263, 2938, 594, 264, 85105, 11, 714, 432, 594, 2797, 807, 1366, 279, 6722, 13, 2379, 1083, 4588, 369, 279, 7042, 304, 4718, 3561, 382, 5338, 11, 358, 1184, 311, 7071, 700, 879, 419, 1196, 374, 13, 2379, 2578, 387, 264, 5458, 11, 7196, 304, 6149, 2906, 476, 1550, 2906, 11, 20956, 53142, 476, 10767, 3330, 369, 264, 3974, 2097, 311, 990, 14696, 770, 13, 38478, 11, 807, 1410, 387, 4325, 1101, 22208, 911, 9625, 594, 6722, 3283, 382, 6865, 5189, 4718, 3561, 11, 773, 807, 2578, 387, 1667, 429, 821, 369, 15473, 476, 1045, 3093, 315, 4625, 1380, 32930, 821, 374, 4362, 13, 10696, 807, 2299, 4752, 264, 2390, 11, 53852, 419, 3546, 1119, 2441, 3766, 11, 476, 1101, 3330, 311, 3037, 279, 7042, 42011, 382, 40, 1265, 1281, 2704, 311, 633, 279, 4396, 1995, 13, 576, 6722, 374, 8491, 12095, 11, 902, 10492, 1052, 13, 1634, 369, 279, 7042, 11, 358, 1184, 311, 10146, 279, 5535, 5109, 13, 358, 19091, 429, 12095, 374, 279, 7772, 3283, 304, 9625, 11, 714, 1181, 7042, 2578, 614, 5497, 264, 2699, 916, 279, 1635, 13, 358, 1744, 432, 594, 2163, 220, 24, 3526, 1251, 13, 358, 1265, 1990, 15934, 429, 311, 5978, 13403, 382, 12549, 279, 1196, 5230, 330, 25069, 323, 7042, 497, 358, 1265, 3410, 2176, 9666, 315, 821, 11, 714, 369, 279, 7042, 11, 45932, 432, 438, 807, 4588, 13, 7281, 11, 10282, 304, 3971, 429, 7042, 12396, 646, 38288, 6292, 11, 773, 432, 594, 1661, 311, 6286, 279, 44868, 2088, 311, 1473, 358, 3535, 26826, 304, 10472, 382, 97904, 432, 1119, 4718, 3561, 11, 358, 3278, 5944, 432, 448, 264, 1376, 369, 330, 65063, 1, 323, 330, 44441, 497, 7196, 438, 458, 1633, 2878, 264, 330, 691, 1, 2070, 13, 2938, 1616, 11, 432, 594, 16645, 323, 4135, 369, 279, 1196, 311, 1349, 11, 5310, 421, 807, 2299, 1667, 419, 821, 304, 264, 5316, 476, 906, 382, 40, 1265, 2506, 279, 2033, 63594, 714, 38219, 11, 1101, 1128, 807, 4588, 369, 2041, 894, 4960, 1320, 1362, 13, 2308, 1184, 369, 264, 34206, 4008, 11, 1101, 279, 5871, 821, 304, 279, 4396, 3561, 13, 2938, 1616, 11, 432, 594, 30339, 323, 20027, 862, 1196, 1681, 29720, 624, 151649, 271, 39814, 0, 5692, 594, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 24, 15, 21, 18, 20, 16, 20, 198, 532, 73594, 151643], 'meta_info': {'id': '9451885be5fe414fa6bfff063ee5a402', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 402, 'completion_tokens': 447, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.9601967465132475, 'response_sent_to_client_ts': 1778998589.5001287}}</strong>



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

    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.10s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.00s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.02s/it]


    2026-05-17 06:16:45,379 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 06:16:45] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:03,  5.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:03,  5.32s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:18,  2.47s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:18,  2.47s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:24,  1.54s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:24,  1.54s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:58,  1.08s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:58,  1.08s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:43,  1.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:43,  1.23it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:33,  1.54it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:33,  1.54it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:27,  1.88it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:27,  1.88it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:22,  2.25it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:22,  2.25it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:18,  2.68it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:18,  2.68it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:15,  3.16it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:15,  3.16it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:12,  3.68it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:12,  3.68it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:10,  4.23it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:10,  4.23it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:09,  4.92it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:09,  4.92it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:07,  5.60it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:07,  5.60it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:06,  6.42it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:06,  6.42it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:08<00:06,  6.42it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:05,  8.20it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:05,  8.20it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:09<00:05,  8.20it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:09<00:03,  9.91it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:09<00:03,  9.91it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:09<00:03,  9.91it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:09<00:03, 11.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:09<00:03, 11.81it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:09<00:03, 11.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:09<00:03, 11.81it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:09<00:02, 15.75it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:09<00:02, 15.75it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:09<00:02, 15.75it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:09<00:02, 15.75it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:09<00:02, 15.75it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:09<00:01, 20.49it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:09<00:01, 20.49it/s]

    Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:09<00:01, 20.49it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:09<00:01, 20.49it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:09<00:01, 20.49it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:09<00:01, 24.38it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:09<00:01, 24.38it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:09<00:01, 24.38it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:09<00:01, 24.38it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:09<00:01, 24.38it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:09<00:01, 24.38it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:09<00:01, 24.38it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:09<00:01, 24.38it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:09<00:00, 35.87it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:09<00:00, 35.87it/s]

    Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:09<00:00, 35.87it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:09<00:00, 35.87it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:09<00:00, 35.87it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:09<00:00, 35.87it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:09<00:00, 35.87it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:09<00:00, 35.87it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:09<00:00, 35.87it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:09<00:00, 46.08it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:09<00:00, 46.08it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:09<00:00, 46.08it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:09<00:00, 46.08it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:09<00:00, 46.08it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:09<00:00, 46.08it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:09<00:00, 46.08it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:09<00:00, 46.08it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:09<00:00, 46.08it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:09<00:00, 46.08it/s]

    Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:09<00:00, 56.58it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:09<00:00, 56.58it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:10<00:00, 56.58it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.25 GB):   2%|▏         | 1/58 [00:00<00:37,  1.54it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.28 GB):   2%|▏         | 1/58 [00:00<00:37,  1.54it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.28 GB):   3%|▎         | 2/58 [00:01<00:34,  1.62it/s]Capturing num tokens (num_tokens=7168 avail_mem=24.34 GB):   3%|▎         | 2/58 [00:01<00:34,  1.62it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=24.34 GB):   5%|▌         | 3/58 [00:01<00:30,  1.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=24.45 GB):   5%|▌         | 3/58 [00:01<00:30,  1.81it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=24.45 GB):   7%|▋         | 4/58 [00:02<00:26,  2.00it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.20 GB):   7%|▋         | 4/58 [00:02<00:26,  2.00it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.20 GB):   9%|▊         | 5/58 [00:02<00:24,  2.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.20 GB):   9%|▊         | 5/58 [00:02<00:24,  2.16it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.20 GB):  10%|█         | 6/58 [00:02<00:21,  2.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.20 GB):  10%|█         | 6/58 [00:02<00:21,  2.40it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.20 GB):  12%|█▏        | 7/58 [00:03<00:19,  2.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=24.70 GB):  12%|█▏        | 7/58 [00:03<00:19,  2.59it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=24.70 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=24.73 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.81it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=24.73 GB):  16%|█▌        | 9/58 [00:03<00:16,  3.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=24.76 GB):  16%|█▌        | 9/58 [00:03<00:16,  3.06it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=24.76 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=24.86 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=24.86 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=25.18 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=25.18 GB):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=25.18 GB):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=25.18 GB):  22%|██▏       | 13/58 [00:04<00:09,  4.53it/s]Capturing num tokens (num_tokens=2816 avail_mem=25.17 GB):  22%|██▏       | 13/58 [00:04<00:09,  4.53it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=25.17 GB):  24%|██▍       | 14/58 [00:04<00:08,  5.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=25.17 GB):  24%|██▍       | 14/58 [00:04<00:08,  5.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=25.17 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=25.16 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.59it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=25.16 GB):  28%|██▊       | 16/58 [00:04<00:06,  6.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.98 GB):  28%|██▊       | 16/58 [00:04<00:06,  6.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.98 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=25.11 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.67it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=25.10 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=25.10 GB):  33%|███▎      | 19/58 [00:05<00:04,  8.18it/s]Capturing num tokens (num_tokens=1280 avail_mem=25.10 GB):  33%|███▎      | 19/58 [00:05<00:04,  8.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=25.08 GB):  33%|███▎      | 19/58 [00:05<00:04,  8.18it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=25.08 GB):  36%|███▌      | 21/58 [00:05<00:03,  9.82it/s]Capturing num tokens (num_tokens=960 avail_mem=25.07 GB):  36%|███▌      | 21/58 [00:05<00:03,  9.82it/s] Capturing num tokens (num_tokens=896 avail_mem=25.07 GB):  36%|███▌      | 21/58 [00:05<00:03,  9.82it/s]Capturing num tokens (num_tokens=896 avail_mem=25.07 GB):  40%|███▉      | 23/58 [00:05<00:03, 10.98it/s]Capturing num tokens (num_tokens=832 avail_mem=25.06 GB):  40%|███▉      | 23/58 [00:05<00:03, 10.98it/s]

    Capturing num tokens (num_tokens=768 avail_mem=25.05 GB):  40%|███▉      | 23/58 [00:05<00:03, 10.98it/s]Capturing num tokens (num_tokens=768 avail_mem=25.05 GB):  43%|████▎     | 25/58 [00:05<00:02, 11.23it/s]Capturing num tokens (num_tokens=704 avail_mem=24.99 GB):  43%|████▎     | 25/58 [00:05<00:02, 11.23it/s]Capturing num tokens (num_tokens=640 avail_mem=24.99 GB):  43%|████▎     | 25/58 [00:05<00:02, 11.23it/s]Capturing num tokens (num_tokens=640 avail_mem=24.99 GB):  47%|████▋     | 27/58 [00:05<00:02, 13.06it/s]Capturing num tokens (num_tokens=576 avail_mem=24.98 GB):  47%|████▋     | 27/58 [00:05<00:02, 13.06it/s]

    Capturing num tokens (num_tokens=512 avail_mem=24.99 GB):  47%|████▋     | 27/58 [00:05<00:02, 13.06it/s]Capturing num tokens (num_tokens=480 avail_mem=24.97 GB):  47%|████▋     | 27/58 [00:05<00:02, 13.06it/s]Capturing num tokens (num_tokens=480 avail_mem=24.97 GB):  52%|█████▏    | 30/58 [00:05<00:01, 15.60it/s]Capturing num tokens (num_tokens=448 avail_mem=24.97 GB):  52%|█████▏    | 30/58 [00:05<00:01, 15.60it/s]Capturing num tokens (num_tokens=416 avail_mem=25.01 GB):  52%|█████▏    | 30/58 [00:05<00:01, 15.60it/s]Capturing num tokens (num_tokens=384 avail_mem=24.96 GB):  52%|█████▏    | 30/58 [00:06<00:01, 15.60it/s]

    Capturing num tokens (num_tokens=384 avail_mem=24.96 GB):  57%|█████▋    | 33/58 [00:06<00:01, 17.61it/s]Capturing num tokens (num_tokens=352 avail_mem=24.99 GB):  57%|█████▋    | 33/58 [00:06<00:01, 17.61it/s]Capturing num tokens (num_tokens=320 avail_mem=24.98 GB):  57%|█████▋    | 33/58 [00:06<00:01, 17.61it/s]Capturing num tokens (num_tokens=288 avail_mem=24.95 GB):  57%|█████▋    | 33/58 [00:06<00:01, 17.61it/s]Capturing num tokens (num_tokens=288 avail_mem=24.95 GB):  62%|██████▏   | 36/58 [00:06<00:01, 20.15it/s]Capturing num tokens (num_tokens=256 avail_mem=24.94 GB):  62%|██████▏   | 36/58 [00:06<00:01, 20.15it/s]Capturing num tokens (num_tokens=240 avail_mem=24.95 GB):  62%|██████▏   | 36/58 [00:06<00:01, 20.15it/s]Capturing num tokens (num_tokens=224 avail_mem=24.95 GB):  62%|██████▏   | 36/58 [00:06<00:01, 20.15it/s]

    Capturing num tokens (num_tokens=224 avail_mem=24.95 GB):  67%|██████▋   | 39/58 [00:06<00:00, 21.77it/s]Capturing num tokens (num_tokens=208 avail_mem=24.93 GB):  67%|██████▋   | 39/58 [00:06<00:00, 21.77it/s]Capturing num tokens (num_tokens=192 avail_mem=24.93 GB):  67%|██████▋   | 39/58 [00:06<00:00, 21.77it/s]Capturing num tokens (num_tokens=176 avail_mem=24.92 GB):  67%|██████▋   | 39/58 [00:06<00:00, 21.77it/s]Capturing num tokens (num_tokens=176 avail_mem=24.92 GB):  72%|███████▏  | 42/58 [00:06<00:00, 23.28it/s]Capturing num tokens (num_tokens=160 avail_mem=24.94 GB):  72%|███████▏  | 42/58 [00:06<00:00, 23.28it/s]Capturing num tokens (num_tokens=144 avail_mem=24.93 GB):  72%|███████▏  | 42/58 [00:06<00:00, 23.28it/s]Capturing num tokens (num_tokens=128 avail_mem=24.91 GB):  72%|███████▏  | 42/58 [00:06<00:00, 23.28it/s]

    Capturing num tokens (num_tokens=128 avail_mem=24.91 GB):  78%|███████▊  | 45/58 [00:06<00:00, 24.12it/s]Capturing num tokens (num_tokens=112 avail_mem=24.92 GB):  78%|███████▊  | 45/58 [00:06<00:00, 24.12it/s]Capturing num tokens (num_tokens=96 avail_mem=24.91 GB):  78%|███████▊  | 45/58 [00:06<00:00, 24.12it/s] Capturing num tokens (num_tokens=80 avail_mem=24.90 GB):  78%|███████▊  | 45/58 [00:06<00:00, 24.12it/s]Capturing num tokens (num_tokens=80 avail_mem=24.90 GB):  83%|████████▎ | 48/58 [00:06<00:00, 25.45it/s]Capturing num tokens (num_tokens=64 avail_mem=24.88 GB):  83%|████████▎ | 48/58 [00:06<00:00, 25.45it/s]Capturing num tokens (num_tokens=48 avail_mem=24.87 GB):  83%|████████▎ | 48/58 [00:06<00:00, 25.45it/s]Capturing num tokens (num_tokens=32 avail_mem=24.86 GB):  83%|████████▎ | 48/58 [00:06<00:00, 25.45it/s]Capturing num tokens (num_tokens=28 avail_mem=24.88 GB):  83%|████████▎ | 48/58 [00:06<00:00, 25.45it/s]

    Capturing num tokens (num_tokens=28 avail_mem=24.88 GB):  90%|████████▉ | 52/58 [00:06<00:00, 27.21it/s]Capturing num tokens (num_tokens=24 avail_mem=24.87 GB):  90%|████████▉ | 52/58 [00:06<00:00, 27.21it/s]Capturing num tokens (num_tokens=20 avail_mem=24.87 GB):  90%|████████▉ | 52/58 [00:06<00:00, 27.21it/s]Capturing num tokens (num_tokens=16 avail_mem=24.85 GB):  90%|████████▉ | 52/58 [00:06<00:00, 27.21it/s]Capturing num tokens (num_tokens=12 avail_mem=24.85 GB):  90%|████████▉ | 52/58 [00:06<00:00, 27.21it/s]Capturing num tokens (num_tokens=12 avail_mem=24.85 GB):  97%|█████████▋| 56/58 [00:06<00:00, 29.27it/s]Capturing num tokens (num_tokens=8 avail_mem=24.85 GB):  97%|█████████▋| 56/58 [00:06<00:00, 29.27it/s] Capturing num tokens (num_tokens=4 avail_mem=24.84 GB):  97%|█████████▋| 56/58 [00:06<00:00, 29.27it/s]Capturing num tokens (num_tokens=4 avail_mem=24.84 GB): 100%|██████████| 58/58 [00:06<00:00,  8.38it/s]


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
    
    Generated text: Alright, so the user is asking for the information and population of the capital of France in JSON format. Hmm, let's break this down. They probably want a structured data response, which makes sense because JSON is commonly used for data interchange. 
    
    First, I need to identify what the capital of France is. That's straightforward—it's Paris. Now, for the population. I should check the latest available data. From what I remember, as of the most recent estimates, Paris has a population of around 2 million. But wait, I should make sure that's accurate. Maybe I should double-check the current statistics to confirm.
    
    I wonder if the user needs exact numbers or just an approximate figure. Since they specified "population" without a time frame, I'll go with the most recent estimate, which I believe is approximately 2,172,000 as of 2023. 
    
    Next, structuring this into JSON. The user wants the information and population, so the JSON should include a "capital" key with the name and a "population" key with the number. Also, maybe adding a "description" to provide context about Paris's significance would be helpful. That way, the information is both accurate and informative.
    
    I should ensure that the JSON is properly formatted so it's easy to read and use for any application the user might be developing. Keeping the structure simple with the necessary fields should meet their needs. 
    
    Wait, do they want the population as a number or a formatted string? JSON can handle both, but providing it as a number makes it easier for any integrating software to process. So, keeping it as 2172000 is fine.
    
    Finally, I should present the response clearly, making sure that the JSON is correctly enclosed in code blocks so it displays properly. That should cover everything the user asked for.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "capital": {
        "name": "Paris",
        "description": "The capital of France, known for its rich history, landmarks, and cultural significance."
      },
      "population": 2172000
    }
    ```



```python
llm.shutdown()
```
