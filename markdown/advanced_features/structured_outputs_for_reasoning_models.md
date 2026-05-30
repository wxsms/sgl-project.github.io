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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.41s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.33s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:00,  2.15s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:00,  2.15s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:10,  1.29s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:10,  1.29s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:23,  2.23it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:23,  2.23it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.90it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.90it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.66it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.66it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.54it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.54it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.54it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.26it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.26it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.26it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.85it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.85it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.85it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.48it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.48it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.48it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.46it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.46it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.46it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:06<00:03, 11.46it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:02, 14.81it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:02, 14.81it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:02, 14.81it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:06<00:02, 14.81it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.81it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 20.36it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 20.36it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 20.36it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 20.36it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.36it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.36it/s]

    Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.36it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 29.30it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 29.30it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 29.30it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 29.30it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 29.30it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 29.30it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 29.30it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.92it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.92it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.92it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.92it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.92it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.92it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.92it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.92it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.92it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 47.42it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 47.42it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 47.42it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 47.42it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 47.42it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 47.42it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 47.42it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 47.42it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 47.42it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 55.99it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 55.99it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 55.99it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 55.99it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 55.99it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 55.99it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 55.99it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=40.61 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=40.61 GB):   2%|▏         | 1/58 [00:00<00:18,  3.16it/s]Capturing num tokens (num_tokens=7680 avail_mem=40.53 GB):   2%|▏         | 1/58 [00:00<00:18,  3.16it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=40.53 GB):   3%|▎         | 2/58 [00:00<00:16,  3.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=40.53 GB):   3%|▎         | 2/58 [00:00<00:16,  3.40it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=40.53 GB):   5%|▌         | 3/58 [00:00<00:15,  3.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=40.52 GB):   5%|▌         | 3/58 [00:00<00:15,  3.64it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=40.52 GB):   7%|▋         | 4/58 [00:01<00:14,  3.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.52 GB):   7%|▋         | 4/58 [00:01<00:14,  3.83it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=40.52 GB):   9%|▊         | 5/58 [00:01<00:12,  4.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=40.52 GB):   9%|▊         | 5/58 [00:01<00:12,  4.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=40.52 GB):  10%|█         | 6/58 [00:01<00:11,  4.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.52 GB):  10%|█         | 6/58 [00:01<00:11,  4.49it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=40.52 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.52 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.52 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=40.52 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.35it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=40.52 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=40.52 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.50it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=40.52 GB):  17%|█▋        | 10/58 [00:02<00:10,  4.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=40.52 GB):  17%|█▋        | 10/58 [00:02<00:10,  4.59it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=40.52 GB):  19%|█▉        | 11/58 [00:02<00:10,  4.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.52 GB):  19%|█▉        | 11/58 [00:02<00:10,  4.39it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=40.52 GB):  21%|██        | 12/58 [00:02<00:10,  4.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=40.51 GB):  21%|██        | 12/58 [00:02<00:10,  4.34it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=40.51 GB):  22%|██▏       | 13/58 [00:02<00:10,  4.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.51 GB):  22%|██▏       | 13/58 [00:02<00:10,  4.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.51 GB):  24%|██▍       | 14/58 [00:03<00:09,  4.55it/s]Capturing num tokens (num_tokens=2560 avail_mem=40.51 GB):  24%|██▍       | 14/58 [00:03<00:09,  4.55it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=40.51 GB):  26%|██▌       | 15/58 [00:03<00:08,  5.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.51 GB):  26%|██▌       | 15/58 [00:03<00:08,  5.00it/s]Capturing num tokens (num_tokens=2048 avail_mem=40.50 GB):  26%|██▌       | 15/58 [00:03<00:08,  5.00it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=40.50 GB):  29%|██▉       | 17/58 [00:03<00:06,  6.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.50 GB):  29%|██▉       | 17/58 [00:03<00:06,  6.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.50 GB):  31%|███       | 18/58 [00:03<00:06,  6.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=40.50 GB):  31%|███       | 18/58 [00:03<00:06,  6.30it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=40.50 GB):  33%|███▎      | 19/58 [00:03<00:06,  6.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.50 GB):  33%|███▎      | 19/58 [00:03<00:06,  6.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.50 GB):  34%|███▍      | 20/58 [00:03<00:05,  6.70it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.49 GB):  34%|███▍      | 20/58 [00:03<00:05,  6.70it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=40.49 GB):  36%|███▌      | 21/58 [00:04<00:05,  7.14it/s]Capturing num tokens (num_tokens=960 avail_mem=40.48 GB):  36%|███▌      | 21/58 [00:04<00:05,  7.14it/s] Capturing num tokens (num_tokens=960 avail_mem=40.48 GB):  38%|███▊      | 22/58 [00:04<00:04,  7.58it/s]Capturing num tokens (num_tokens=896 avail_mem=40.48 GB):  38%|███▊      | 22/58 [00:04<00:04,  7.58it/s]

    Capturing num tokens (num_tokens=896 avail_mem=40.48 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.95it/s]Capturing num tokens (num_tokens=832 avail_mem=40.48 GB):  40%|███▉      | 23/58 [00:04<00:04,  7.95it/s]Capturing num tokens (num_tokens=832 avail_mem=40.48 GB):  41%|████▏     | 24/58 [00:04<00:04,  8.37it/s]Capturing num tokens (num_tokens=768 avail_mem=40.47 GB):  41%|████▏     | 24/58 [00:04<00:04,  8.37it/s]Capturing num tokens (num_tokens=704 avail_mem=40.47 GB):  41%|████▏     | 24/58 [00:04<00:04,  8.37it/s]

    Capturing num tokens (num_tokens=704 avail_mem=40.47 GB):  45%|████▍     | 26/58 [00:04<00:03,  9.31it/s]Capturing num tokens (num_tokens=640 avail_mem=40.47 GB):  45%|████▍     | 26/58 [00:04<00:03,  9.31it/s]Capturing num tokens (num_tokens=576 avail_mem=40.46 GB):  45%|████▍     | 26/58 [00:04<00:03,  9.31it/s]Capturing num tokens (num_tokens=576 avail_mem=40.46 GB):  48%|████▊     | 28/58 [00:04<00:02, 10.50it/s]Capturing num tokens (num_tokens=512 avail_mem=40.46 GB):  48%|████▊     | 28/58 [00:04<00:02, 10.50it/s]

    Capturing num tokens (num_tokens=480 avail_mem=40.45 GB):  48%|████▊     | 28/58 [00:04<00:02, 10.50it/s]Capturing num tokens (num_tokens=480 avail_mem=40.45 GB):  52%|█████▏    | 30/58 [00:04<00:02, 12.21it/s]Capturing num tokens (num_tokens=448 avail_mem=40.45 GB):  52%|█████▏    | 30/58 [00:04<00:02, 12.21it/s]Capturing num tokens (num_tokens=416 avail_mem=40.45 GB):  52%|█████▏    | 30/58 [00:04<00:02, 12.21it/s]Capturing num tokens (num_tokens=416 avail_mem=40.45 GB):  55%|█████▌    | 32/58 [00:05<00:01, 13.11it/s]Capturing num tokens (num_tokens=384 avail_mem=40.44 GB):  55%|█████▌    | 32/58 [00:05<00:01, 13.11it/s]

    Capturing num tokens (num_tokens=352 avail_mem=40.44 GB):  55%|█████▌    | 32/58 [00:05<00:01, 13.11it/s]Capturing num tokens (num_tokens=352 avail_mem=40.44 GB):  59%|█████▊    | 34/58 [00:05<00:01, 14.48it/s]Capturing num tokens (num_tokens=320 avail_mem=40.43 GB):  59%|█████▊    | 34/58 [00:05<00:01, 14.48it/s]Capturing num tokens (num_tokens=288 avail_mem=40.44 GB):  59%|█████▊    | 34/58 [00:05<00:01, 14.48it/s]Capturing num tokens (num_tokens=288 avail_mem=40.44 GB):  62%|██████▏   | 36/58 [00:05<00:01, 15.77it/s]Capturing num tokens (num_tokens=256 avail_mem=40.43 GB):  62%|██████▏   | 36/58 [00:05<00:01, 15.77it/s]

    Capturing num tokens (num_tokens=240 avail_mem=40.43 GB):  62%|██████▏   | 36/58 [00:05<00:01, 15.77it/s]Capturing num tokens (num_tokens=224 avail_mem=40.43 GB):  62%|██████▏   | 36/58 [00:05<00:01, 15.77it/s]Capturing num tokens (num_tokens=224 avail_mem=40.43 GB):  67%|██████▋   | 39/58 [00:05<00:01, 17.61it/s]Capturing num tokens (num_tokens=208 avail_mem=40.42 GB):  67%|██████▋   | 39/58 [00:05<00:01, 17.61it/s]Capturing num tokens (num_tokens=192 avail_mem=40.42 GB):  67%|██████▋   | 39/58 [00:05<00:01, 17.61it/s]Capturing num tokens (num_tokens=176 avail_mem=40.42 GB):  67%|██████▋   | 39/58 [00:05<00:01, 17.61it/s]

    Capturing num tokens (num_tokens=176 avail_mem=40.42 GB):  72%|███████▏  | 42/58 [00:05<00:00, 18.70it/s]Capturing num tokens (num_tokens=160 avail_mem=40.41 GB):  72%|███████▏  | 42/58 [00:05<00:00, 18.70it/s]Capturing num tokens (num_tokens=144 avail_mem=40.41 GB):  72%|███████▏  | 42/58 [00:05<00:00, 18.70it/s]Capturing num tokens (num_tokens=128 avail_mem=40.41 GB):  72%|███████▏  | 42/58 [00:05<00:00, 18.70it/s]Capturing num tokens (num_tokens=128 avail_mem=40.41 GB):  78%|███████▊  | 45/58 [00:05<00:00, 19.48it/s]Capturing num tokens (num_tokens=112 avail_mem=40.41 GB):  78%|███████▊  | 45/58 [00:05<00:00, 19.48it/s]Capturing num tokens (num_tokens=96 avail_mem=40.40 GB):  78%|███████▊  | 45/58 [00:05<00:00, 19.48it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=40.40 GB):  78%|███████▊  | 45/58 [00:05<00:00, 19.48it/s]Capturing num tokens (num_tokens=80 avail_mem=40.40 GB):  83%|████████▎ | 48/58 [00:05<00:00, 20.14it/s]Capturing num tokens (num_tokens=64 avail_mem=40.40 GB):  83%|████████▎ | 48/58 [00:05<00:00, 20.14it/s]Capturing num tokens (num_tokens=48 avail_mem=40.39 GB):  83%|████████▎ | 48/58 [00:05<00:00, 20.14it/s]Capturing num tokens (num_tokens=32 avail_mem=40.39 GB):  83%|████████▎ | 48/58 [00:05<00:00, 20.14it/s]Capturing num tokens (num_tokens=32 avail_mem=40.39 GB):  88%|████████▊ | 51/58 [00:05<00:00, 21.17it/s]Capturing num tokens (num_tokens=28 avail_mem=40.39 GB):  88%|████████▊ | 51/58 [00:05<00:00, 21.17it/s]

    Capturing num tokens (num_tokens=24 avail_mem=40.38 GB):  88%|████████▊ | 51/58 [00:05<00:00, 21.17it/s]Capturing num tokens (num_tokens=20 avail_mem=40.38 GB):  88%|████████▊ | 51/58 [00:06<00:00, 21.17it/s]Capturing num tokens (num_tokens=20 avail_mem=40.38 GB):  93%|█████████▎| 54/58 [00:06<00:00, 21.79it/s]Capturing num tokens (num_tokens=16 avail_mem=40.38 GB):  93%|█████████▎| 54/58 [00:06<00:00, 21.79it/s]Capturing num tokens (num_tokens=12 avail_mem=40.37 GB):  93%|█████████▎| 54/58 [00:06<00:00, 21.79it/s]Capturing num tokens (num_tokens=8 avail_mem=40.37 GB):  93%|█████████▎| 54/58 [00:06<00:00, 21.79it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=40.37 GB):  98%|█████████▊| 57/58 [00:06<00:00, 22.18it/s]Capturing num tokens (num_tokens=4 avail_mem=40.37 GB):  98%|█████████▊| 57/58 [00:06<00:00, 22.18it/s]Capturing num tokens (num_tokens=4 avail_mem=40.37 GB): 100%|██████████| 58/58 [00:06<00:00,  9.31it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, I need to figure out how to get the current date and time in New York and the weather there using the given functions. The user mentioned they're in New York, so the timezone is already set. <br><br>First, I should use the get_current_date function. The function requires a timezone parameter. Since the user is in New York, I can input 'America/New_York' as the timezone. I don't need any other parameters for this function.<br><br>Next, for the weather, I should use get_current_weather. The function needs a city, state, and unit. The city is New York, but I need to find the correct two-letter state abbreviation. New York is NY, so the state parameter should be 'NY'. The unit is probably Fahrenheit since that's common in the US, but I'm not sure. Wait, the user didn't specify, but since the example uses unit as 'fahrenheit', I'll go with that. So the parameters for this function will be city: 'New York', state: 'NY', unit: 'fahrenheit'.<br><br>I should make sure to format the function calls correctly. Each function call should be on its own line, using the specified JSON format. So first, I'll write the date function, then the weather function. I also need to include the sources in the answer, so I'll add a note that the information is from current data sources.<br><br><br>content: <br><br><function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>  <br><br>The current date and time in New York is [date]. The current weather in New York is [weather_info], with a temperature of [temperature]°F, [conditions] and [wind_speed] mph wind. This information is sourced from current data.</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '431276d6d573409d9e414abad025210d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.221155953593552, 'response_sent_to_client_ts': 1780122082.2740462}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'c4bd9621e3d045639f827b2238a31d45', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.357800356112421, 'response_sent_to_client_ts': 1780122087.6415856}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '6b6f0a35ffea48d8b7f80cd3e1a071cc', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2089721243828535, 'response_sent_to_client_ts': 1780122087.8796995}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '9769f23d722b4d179e0bc0de33daa1f5', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.20872727315872908, 'response_sent_to_client_ts': 1780122087.879716}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'b43e3ac2c685466c8670ef92941a689d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2086741914972663, 'response_sent_to_client_ts': 1780122087.8797202}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '02472b99035b42c087094ab577c8b7ad', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 18.63853903580457, 'response_sent_to_client_ts': 1780122106.5263147}}


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


<strong style='color: #00008B;'>{'text': 'Alright, the user is asking for the information and population of France\'s capital, which is Paris. They specifically want this in JSON format, so I should format it correctly.\n\nFirst, I\'ll need to make sure I have the correct capital. Paris is indeed the capital of France, so that\'s confirmed.\n\nNext, I should find the most recent population figure. I recall the latest estimate being around 2.2 million as of 2023, but I should double-check that to ensure accuracy.\n\nI also need to provide some key information about Paris. Known facts include its location in the northern part of France, the(geometry) of its being a bi modal river, the approximate area size, and famous landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame.\n\nStructurally, Paris is a grand城区, meaning it\'s extensively planned and designed, which is a notable point. Also, mentioning the diverse population and cultural significance, with flags of over 200 countries flying, would add depth.\n\nIncluding population statistics, I should note the total population, the urban area population, and metro population since these can vary and are important for context.\n\nI should structure the JSON with separate sections for information and population. The information section can have keys like capital, location, description, landmarks, notable_structures, cultural_significance, and population_stats. Under population_stats, including total, urban, metro, and under cultural_significance, the number of countries with flags.\n\nIt\'s important to keep the JSON valid, so proper indentation and structure are a must. Also, ensuring that special characters like accents in "巴黎" are correctly handled.\n\nI should avoid any technical jargon that might confuse the user, but still provide all the necessary details they requested.\n\nFinally, I\'ll make sure the response is clear, concise, and covers all aspects the user asked for, providing a comprehensive answer in JSON format.\n</think>\n\nHere is the information about the capital of France (Paris) in JSON format:\n\n```json\n{\n  "capital": {\n    "name": "Paris",\n    "location": {\n      "country": "France",\n      "region": "Paris",\n      "coordinates": {\n        "latitude": "48.8566",\n        "longitude": "2.3522"\n      }\n    },\n    "description": "The capital of France, located in the northern part of the country, is a major city in terms of culture, history, and economics.",\n    "landmarks": [\n      "Eiffel Tower",\n      "Louvre Museum",\n      "Notre-Dame Cathedral",\n      "Oper长江 Theaters",\n      "Jardin du Luxembourg"\n    ],\n    "notable_structures": [\n      "Arc de Triomphe",\n      "Seine Tower",\n      "Le Marais",\n      "Château de France",\n      "La Défense"\n    ],\n    "cultural_significance": "Paris is renowned for its rich cultural heritage, serving as the birthplace of modern art, literature, and music. It is home to iconic landmarks and attractions that attract millions of visitors annually.",\n    "population_stats": {\n      "total": "Approx. 2.2 million",\n      "urban": "Approx. 13 million",\n      "metro": "Approx. 11 million"\n    }\n  }\n}\n```\n\nLet me know if you need further details!', 'output_ids': [71486, 11, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 9625, 594, 6722, 11, 892, 374, 12095, 13, 2379, 11689, 1366, 419, 304, 4718, 3561, 11, 773, 358, 1265, 3561, 432, 12440, 382, 5338, 11, 358, 3278, 1184, 311, 1281, 2704, 358, 614, 279, 4396, 6722, 13, 12095, 374, 12824, 279, 6722, 315, 9625, 11, 773, 429, 594, 10774, 382, 5847, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 7071, 13, 358, 19091, 279, 5535, 16045, 1660, 2163, 220, 17, 13, 17, 3526, 438, 315, 220, 17, 15, 17, 18, 11, 714, 358, 1265, 1990, 15934, 429, 311, 5978, 13403, 382, 40, 1083, 1184, 311, 3410, 1045, 1376, 1995, 911, 12095, 13, 48286, 13064, 2924, 1181, 3728, 304, 279, 18172, 949, 315, 9625, 11, 279, 86828, 8, 315, 1181, 1660, 264, 6032, 13228, 14796, 11, 279, 44868, 3082, 1379, 11, 323, 11245, 59924, 1075, 279, 468, 3092, 301, 21938, 11, 279, 9729, 48506, 16328, 11, 323, 43464, 9420, 373, 382, 9422, 41924, 11, 12095, 374, 264, 6662, 101318, 11, 7290, 432, 594, 41717, 12909, 323, 6188, 11, 892, 374, 264, 27190, 1459, 13, 7281, 11, 44291, 279, 16807, 7042, 323, 12752, 25361, 11, 448, 8042, 315, 916, 220, 17, 15, 15, 5837, 16307, 11, 1035, 912, 7990, 382, 83449, 7042, 13142, 11, 358, 1265, 5185, 279, 2790, 7042, 11, 279, 15662, 3082, 7042, 11, 323, 33482, 7042, 2474, 1493, 646, 13289, 323, 525, 2989, 369, 2266, 382, 40, 1265, 5944, 279, 4718, 448, 8651, 14158, 369, 1995, 323, 7042, 13, 576, 1995, 3772, 646, 614, 6894, 1075, 6722, 11, 3728, 11, 4008, 11, 59924, 11, 27190, 15126, 1413, 11, 12752, 11172, 99004, 11, 323, 7042, 15381, 13, 9449, 7042, 15381, 11, 2670, 2790, 11, 15662, 11, 33482, 11, 323, 1212, 12752, 11172, 99004, 11, 279, 1372, 315, 5837, 448, 8042, 382, 2132, 594, 2989, 311, 2506, 279, 4718, 2697, 11, 773, 6169, 69592, 323, 5944, 525, 264, 1969, 13, 7281, 11, 22573, 429, 3281, 5766, 1075, 58470, 304, 330, 106004, 1, 525, 12440, 17608, 382, 40, 1265, 5648, 894, 10916, 502, 70821, 429, 2578, 58117, 279, 1196, 11, 714, 2058, 3410, 678, 279, 5871, 3565, 807, 11223, 382, 23949, 11, 358, 3278, 1281, 2704, 279, 2033, 374, 2797, 11, 63594, 11, 323, 14521, 678, 13566, 279, 1196, 4588, 369, 11, 8241, 264, 15817, 4226, 304, 4718, 3561, 624, 151649, 271, 8420, 374, 279, 1995, 911, 279, 6722, 315, 9625, 320, 59604, 8, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 341, 262, 330, 606, 788, 330, 59604, 756, 262, 330, 2527, 788, 341, 414, 330, 11141, 788, 330, 49000, 756, 414, 330, 3943, 788, 330, 59604, 756, 414, 330, 34739, 788, 341, 286, 330, 23718, 788, 330, 19, 23, 13, 23, 20, 21, 21, 756, 286, 330, 25446, 788, 330, 17, 13, 18, 20, 17, 17, 698, 414, 456, 262, 1153, 262, 330, 4684, 788, 330, 785, 6722, 315, 9625, 11, 7407, 304, 279, 18172, 949, 315, 279, 3146, 11, 374, 264, 3598, 3283, 304, 3793, 315, 7674, 11, 3840, 11, 323, 27889, 10346, 262, 330, 1933, 15544, 788, 2278, 414, 330, 36, 3092, 301, 21938, 756, 414, 330, 92806, 48506, 16328, 756, 414, 330, 2623, 265, 9420, 373, 56729, 756, 414, 330, 5494, 104924, 576, 11535, 756, 414, 330, 41, 53790, 3845, 64871, 698, 262, 3211, 262, 330, 1921, 480, 15126, 1413, 788, 2278, 414, 330, 36809, 409, 12359, 14435, 383, 756, 414, 330, 1514, 482, 21938, 756, 414, 330, 2304, 2876, 2782, 756, 414, 330, 1143, 68528, 2863, 409, 9625, 756, 414, 330, 8747, 49401, 69, 1117, 698, 262, 3211, 262, 330, 66, 43447, 11172, 99004, 788, 330, 59604, 374, 35948, 369, 1181, 9080, 12752, 27848, 11, 13480, 438, 279, 7194, 2007, 315, 6481, 1947, 11, 17206, 11, 323, 4627, 13, 1084, 374, 2114, 311, 26277, 59924, 323, 38491, 429, 9320, 11728, 315, 15255, 29071, 10346, 262, 330, 44441, 15381, 788, 341, 414, 330, 5035, 788, 330, 69520, 13, 220, 17, 13, 17, 3526, 756, 414, 330, 59059, 788, 330, 69520, 13, 220, 16, 18, 3526, 756, 414, 330, 57047, 788, 330, 69520, 13, 220, 16, 16, 3526, 698, 262, 456, 220, 456, 532, 13874, 19324, 10061, 752, 1414, 421, 498, 1184, 4623, 3565, 0, 151643], 'meta_info': {'id': 'ada2ec66449e4d0dbe6f1ece362508fd', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 391, 'completion_tokens': 704, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 6.415542095899582, 'response_sent_to_client_ts': 1780122112.9508204}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.27s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.23s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.24s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:57,  2.09s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:57,  2.09s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:07,  1.23s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:07,  1.23s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:44,  1.22it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:44,  1.22it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:30,  1.72it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:30,  1.72it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:26,  1.98it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:26,  1.98it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:20,  2.43it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:20,  2.43it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:15,  3.13it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:15,  3.13it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:12,  3.94it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:12,  3.94it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:12,  3.94it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:08,  5.57it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:08,  5.57it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:08,  5.57it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:06,  7.10it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:06,  7.10it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:06,  7.10it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  8.66it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  8.66it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  8.66it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 10.55it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 10.55it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.55it/s]

    Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 10.55it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 13.75it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 13.75it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 13.75it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 13.75it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 13.75it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.23it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.23it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.23it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.23it/s]

    Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.23it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.23it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 19.23it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.06it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.06it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.06it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.06it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.06it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.06it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.06it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 35.63it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 35.63it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 35.63it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 35.63it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 35.63it/s]

    Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 35.63it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 35.63it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 35.63it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 35.63it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 46.06it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 46.06it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 46.06it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 46.06it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 46.06it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 46.06it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 46.06it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 46.06it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 46.06it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 54.84it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 54.84it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 54.84it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 54.84it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 54.84it/s]

    Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 54.84it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 54.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=38.02 GB):   2%|▏         | 1/58 [00:00<00:23,  2.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=37.23 GB):   2%|▏         | 1/58 [00:00<00:23,  2.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=37.23 GB):   3%|▎         | 2/58 [00:01<00:28,  1.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=37.23 GB):   3%|▎         | 2/58 [00:01<00:28,  1.94it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=37.23 GB):   5%|▌         | 3/58 [00:01<00:28,  1.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=37.23 GB):   5%|▌         | 3/58 [00:01<00:28,  1.91it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=37.23 GB):   7%|▋         | 4/58 [00:02<00:27,  1.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=37.23 GB):   7%|▋         | 4/58 [00:02<00:27,  1.98it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=37.23 GB):   9%|▊         | 5/58 [00:02<00:25,  2.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=37.23 GB):   9%|▊         | 5/58 [00:02<00:25,  2.05it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=37.23 GB):  10%|█         | 6/58 [00:02<00:23,  2.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=37.22 GB):  10%|█         | 6/58 [00:02<00:23,  2.19it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=37.22 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=37.23 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.33it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=37.23 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=37.23 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.51it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=37.23 GB):  16%|█▌        | 9/58 [00:03<00:18,  2.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=37.23 GB):  16%|█▌        | 9/58 [00:03<00:18,  2.71it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=37.23 GB):  17%|█▋        | 10/58 [00:04<00:16,  2.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=37.22 GB):  17%|█▋        | 10/58 [00:04<00:16,  2.90it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=37.22 GB):  19%|█▉        | 11/58 [00:04<00:15,  3.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=37.22 GB):  19%|█▉        | 11/58 [00:04<00:15,  3.10it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=37.22 GB):  21%|██        | 12/58 [00:04<00:13,  3.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=37.22 GB):  21%|██        | 12/58 [00:04<00:13,  3.39it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=37.22 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=37.22 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=37.22 GB):  24%|██▍       | 14/58 [00:05<00:10,  4.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=37.22 GB):  24%|██▍       | 14/58 [00:05<00:10,  4.09it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=37.22 GB):  26%|██▌       | 15/58 [00:05<00:09,  4.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=37.21 GB):  26%|██▌       | 15/58 [00:05<00:09,  4.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=37.21 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=37.21 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.26it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=37.21 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=37.21 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=37.21 GB):  31%|███       | 18/58 [00:05<00:05,  6.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=37.20 GB):  31%|███       | 18/58 [00:05<00:05,  6.85it/s]Capturing num tokens (num_tokens=1280 avail_mem=37.20 GB):  31%|███       | 18/58 [00:05<00:05,  6.85it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=37.20 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=37.19 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.73it/s]Capturing num tokens (num_tokens=960 avail_mem=37.18 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.73it/s] Capturing num tokens (num_tokens=960 avail_mem=37.18 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.50it/s]Capturing num tokens (num_tokens=896 avail_mem=37.18 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.50it/s]

    Capturing num tokens (num_tokens=832 avail_mem=37.18 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.50it/s]Capturing num tokens (num_tokens=832 avail_mem=37.18 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.70it/s]Capturing num tokens (num_tokens=768 avail_mem=37.17 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.70it/s]Capturing num tokens (num_tokens=704 avail_mem=37.17 GB):  41%|████▏     | 24/58 [00:06<00:02, 11.70it/s]Capturing num tokens (num_tokens=704 avail_mem=37.17 GB):  45%|████▍     | 26/58 [00:06<00:02, 13.40it/s]Capturing num tokens (num_tokens=640 avail_mem=37.16 GB):  45%|████▍     | 26/58 [00:06<00:02, 13.40it/s]Capturing num tokens (num_tokens=576 avail_mem=37.16 GB):  45%|████▍     | 26/58 [00:06<00:02, 13.40it/s]

    Capturing num tokens (num_tokens=576 avail_mem=37.16 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.71it/s]Capturing num tokens (num_tokens=512 avail_mem=37.15 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.71it/s]Capturing num tokens (num_tokens=480 avail_mem=37.15 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.71it/s]Capturing num tokens (num_tokens=448 avail_mem=37.15 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.71it/s]Capturing num tokens (num_tokens=448 avail_mem=37.15 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.17it/s]Capturing num tokens (num_tokens=416 avail_mem=36.18 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.17it/s]

    Capturing num tokens (num_tokens=384 avail_mem=36.17 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.17it/s]Capturing num tokens (num_tokens=384 avail_mem=36.17 GB):  57%|█████▋    | 33/58 [00:06<00:02, 12.40it/s]Capturing num tokens (num_tokens=352 avail_mem=36.17 GB):  57%|█████▋    | 33/58 [00:06<00:02, 12.40it/s]

    Capturing num tokens (num_tokens=320 avail_mem=37.10 GB):  57%|█████▋    | 33/58 [00:06<00:02, 12.40it/s]Capturing num tokens (num_tokens=320 avail_mem=37.10 GB):  60%|██████    | 35/58 [00:06<00:01, 12.22it/s]Capturing num tokens (num_tokens=288 avail_mem=36.29 GB):  60%|██████    | 35/58 [00:06<00:01, 12.22it/s]

    Capturing num tokens (num_tokens=256 avail_mem=36.28 GB):  60%|██████    | 35/58 [00:06<00:01, 12.22it/s]Capturing num tokens (num_tokens=256 avail_mem=36.28 GB):  64%|██████▍   | 37/58 [00:07<00:02, 10.40it/s]Capturing num tokens (num_tokens=240 avail_mem=37.10 GB):  64%|██████▍   | 37/58 [00:07<00:02, 10.40it/s]Capturing num tokens (num_tokens=224 avail_mem=37.10 GB):  64%|██████▍   | 37/58 [00:07<00:02, 10.40it/s]

    Capturing num tokens (num_tokens=224 avail_mem=37.10 GB):  67%|██████▋   | 39/58 [00:07<00:01, 11.30it/s]Capturing num tokens (num_tokens=208 avail_mem=36.32 GB):  67%|██████▋   | 39/58 [00:07<00:01, 11.30it/s]Capturing num tokens (num_tokens=192 avail_mem=36.32 GB):  67%|██████▋   | 39/58 [00:07<00:01, 11.30it/s]

    Capturing num tokens (num_tokens=192 avail_mem=36.32 GB):  71%|███████   | 41/58 [00:07<00:01, 10.75it/s]Capturing num tokens (num_tokens=176 avail_mem=36.32 GB):  71%|███████   | 41/58 [00:07<00:01, 10.75it/s]Capturing num tokens (num_tokens=160 avail_mem=37.08 GB):  71%|███████   | 41/58 [00:07<00:01, 10.75it/s]Capturing num tokens (num_tokens=160 avail_mem=37.08 GB):  74%|███████▍  | 43/58 [00:07<00:01, 10.71it/s]Capturing num tokens (num_tokens=144 avail_mem=36.37 GB):  74%|███████▍  | 43/58 [00:07<00:01, 10.71it/s]

    Capturing num tokens (num_tokens=128 avail_mem=36.37 GB):  74%|███████▍  | 43/58 [00:07<00:01, 10.71it/s]Capturing num tokens (num_tokens=128 avail_mem=36.37 GB):  78%|███████▊  | 45/58 [00:07<00:01, 10.53it/s]Capturing num tokens (num_tokens=112 avail_mem=37.08 GB):  78%|███████▊  | 45/58 [00:07<00:01, 10.53it/s]Capturing num tokens (num_tokens=96 avail_mem=36.41 GB):  78%|███████▊  | 45/58 [00:07<00:01, 10.53it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=36.41 GB):  81%|████████  | 47/58 [00:07<00:01, 10.73it/s]Capturing num tokens (num_tokens=80 avail_mem=36.41 GB):  81%|████████  | 47/58 [00:07<00:01, 10.73it/s]Capturing num tokens (num_tokens=64 avail_mem=36.40 GB):  81%|████████  | 47/58 [00:08<00:01, 10.73it/s]Capturing num tokens (num_tokens=64 avail_mem=36.40 GB):  84%|████████▍ | 49/58 [00:08<00:00, 10.97it/s]Capturing num tokens (num_tokens=48 avail_mem=37.06 GB):  84%|████████▍ | 49/58 [00:08<00:00, 10.97it/s]

    Capturing num tokens (num_tokens=32 avail_mem=36.45 GB):  84%|████████▍ | 49/58 [00:08<00:00, 10.97it/s]Capturing num tokens (num_tokens=32 avail_mem=36.45 GB):  88%|████████▊ | 51/58 [00:08<00:00, 10.87it/s]Capturing num tokens (num_tokens=28 avail_mem=36.45 GB):  88%|████████▊ | 51/58 [00:08<00:00, 10.87it/s]Capturing num tokens (num_tokens=24 avail_mem=37.05 GB):  88%|████████▊ | 51/58 [00:08<00:00, 10.87it/s]

    Capturing num tokens (num_tokens=24 avail_mem=37.05 GB):  91%|█████████▏| 53/58 [00:08<00:00, 11.33it/s]Capturing num tokens (num_tokens=20 avail_mem=36.50 GB):  91%|█████████▏| 53/58 [00:08<00:00, 11.33it/s]Capturing num tokens (num_tokens=16 avail_mem=37.05 GB):  91%|█████████▏| 53/58 [00:08<00:00, 11.33it/s]Capturing num tokens (num_tokens=16 avail_mem=37.05 GB):  95%|█████████▍| 55/58 [00:08<00:00, 11.78it/s]Capturing num tokens (num_tokens=12 avail_mem=36.95 GB):  95%|█████████▍| 55/58 [00:08<00:00, 11.78it/s]

    Capturing num tokens (num_tokens=8 avail_mem=36.54 GB):  95%|█████████▍| 55/58 [00:08<00:00, 11.78it/s] Capturing num tokens (num_tokens=8 avail_mem=36.54 GB):  98%|█████████▊| 57/58 [00:08<00:00, 11.60it/s]Capturing num tokens (num_tokens=4 avail_mem=37.03 GB):  98%|█████████▊| 57/58 [00:08<00:00, 11.60it/s]Capturing num tokens (num_tokens=4 avail_mem=37.03 GB): 100%|██████████| 58/58 [00:08<00:00,  6.54it/s]


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
    
    Generated text: Alright, so the user is asking for the information and population of the capital of France in JSON format. Hmm, let me break this down. First, I need to figure out where the capital of France is. I remember that Paris is the capital, so that's straightforward.
    
    Next, I should determine the population. I'm pretty sure the population of Paris is over 3 million, but I'm not 100% certain. Maybe it's around 3.5 million? I'll double-check that. Okay, so I'll go with an approximate population of 3,500,000 people.
    
    Now, I need to structure this information into a JSON format. JSON typically uses key-value pairs, so I'll need to define the keys. The main ones would be "capital" and "population." The population should be an integer since it's a count of people.
    
    I should make sure the JSON syntax is correct. Each key is in quotes, and the values are also in quotes. The keys themselves are in double quotes, and the entire data is enclosed in curly braces. So, the JSON should look like {"capital": "Paris", "population": 3500000}.
    
    I also need to consider the date of the population data because city populations can change over time. Maybe I should include a year, like 2023, to specify when the population figure is from. So, updating the JSON to include the year: {"capital": "Paris", "population": 3500000, "year": 2023}.
    
    Wait, is that necessary? The user didn't specify, but including the year might add more context and make the data more accurate. Plus, it's a good practice to include such details if they're relevant.
    
    I should also think about formatting the JSON correctly. Each key-value pair is on a new line for readability, but in code, it's common to have them on the same line. However, for clarity here, using new lines might be better. So, I'll format it with each key on its own line, separated by commas.
    
    Putting it all together, the JSON should clearly state that Paris is the capital, its population as of 2023 is 3.5 million, and be formatted correctly without any syntax errors. That way, the user gets the information they need in a structured and easy-to-understand format.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "capital": "Paris",
      "population": 3500000,
      "year": 2023
    }
    ```



```python
llm.shutdown()
```
