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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.54s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.40s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.43s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:27,  5.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:27,  5.75s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:35,  2.78s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:35,  2.78s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:07<01:39,  1.80s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:07<01:39,  1.80s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:11,  1.32s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:11,  1.32s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:08<00:54,  1.03s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:08<00:54,  1.03s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:08<00:44,  1.17it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:08<00:44,  1.17it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:09<00:36,  1.41it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:09<00:36,  1.41it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:09<00:30,  1.63it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:09<00:30,  1.63it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:25,  1.89it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:25,  1.89it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:10<00:22,  2.17it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:10<00:22,  2.17it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:10<00:19,  2.42it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:10<00:19,  2.42it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:10<00:16,  2.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:10<00:16,  2.71it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:10<00:14,  3.02it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:10<00:14,  3.02it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:11<00:13,  3.34it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:11<00:13,  3.34it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:11<00:11,  3.72it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:11<00:11,  3.72it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:11<00:10,  4.13it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:11<00:10,  4.13it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:11<00:08,  4.60it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:11<00:08,  4.60it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:11<00:07,  5.18it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:11<00:07,  5.18it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:12<00:06,  5.83it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:12<00:06,  5.83it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:12<00:05,  6.50it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:12<00:05,  6.50it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:12<00:05,  6.50it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:12<00:04,  8.17it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:12<00:04,  8.17it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:12<00:04,  8.17it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:12<00:03,  9.76it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:12<00:03,  9.76it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:12<00:03,  9.76it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:12<00:02, 11.25it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:12<00:02, 11.25it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:12<00:02, 11.25it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:12<00:02, 12.97it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:12<00:02, 12.97it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:12<00:02, 12.97it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:12<00:02, 12.97it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:12<00:01, 15.68it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:12<00:01, 15.68it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:12<00:01, 15.68it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:12<00:01, 15.68it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:12<00:01, 18.01it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:12<00:01, 18.01it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:12<00:01, 18.01it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:13<00:01, 18.01it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:13<00:01, 20.25it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:13<00:01, 20.25it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:13<00:01, 20.25it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:13<00:01, 20.25it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:13<00:01, 20.25it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:13<00:00, 23.81it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:13<00:00, 23.81it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:13<00:00, 23.81it/s]

    Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:13<00:00, 23.81it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:13<00:00, 24.94it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:13<00:00, 24.94it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:13<00:00, 24.94it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:13<00:00, 24.94it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:13<00:00, 24.94it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:13<00:00, 27.57it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:13<00:00, 27.57it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:13<00:00, 27.57it/s]

    Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:13<00:00, 27.57it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:13<00:00, 27.57it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:13<00:00, 29.80it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:13<00:00, 29.80it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:13<00:00, 29.80it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:13<00:00, 29.80it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:13<00:00, 29.80it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:13<00:00, 29.80it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:13<00:00, 34.40it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:13<00:00, 34.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:13<00:00,  4.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   2%|▏         | 1/58 [00:00<00:51,  1.10it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.70 GB):   2%|▏         | 1/58 [00:00<00:51,  1.10it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.70 GB):   3%|▎         | 2/58 [00:01<00:48,  1.16it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   3%|▎         | 2/58 [00:01<00:48,  1.16it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   5%|▌         | 3/58 [00:02<00:44,  1.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=26.68 GB):   5%|▌         | 3/58 [00:02<00:44,  1.23it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=26.68 GB):   7%|▋         | 4/58 [00:03<00:40,  1.34it/s]Capturing num tokens (num_tokens=6144 avail_mem=26.68 GB):   7%|▋         | 4/58 [00:03<00:40,  1.34it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=26.68 GB):   9%|▊         | 5/58 [00:03<00:37,  1.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=26.68 GB):   9%|▊         | 5/58 [00:03<00:37,  1.42it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=26.68 GB):  10%|█         | 6/58 [00:04<00:33,  1.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=26.68 GB):  10%|█         | 6/58 [00:04<00:33,  1.54it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=26.68 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=26.68 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.66it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=26.68 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.68 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.83it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.68 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.68 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.00it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=26.68 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=26.67 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.17it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=26.67 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=26.67 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.35it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=26.67 GB):  21%|██        | 12/58 [00:06<00:17,  2.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=26.66 GB):  21%|██        | 12/58 [00:06<00:17,  2.57it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=26.66 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.66 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.77it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.66 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.65 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.06it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.65 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.65 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.38it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=26.65 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.65 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.65 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.64 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.14it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=26.64 GB):  31%|███       | 18/58 [00:07<00:08,  4.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.62 GB):  31%|███       | 18/58 [00:07<00:08,  4.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.62 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.62 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.14it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=26.62 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.42 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.80it/s]Capturing num tokens (num_tokens=960 avail_mem=26.57 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.80it/s] Capturing num tokens (num_tokens=960 avail_mem=26.57 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.24it/s]Capturing num tokens (num_tokens=896 avail_mem=26.56 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.24it/s]

    Capturing num tokens (num_tokens=832 avail_mem=26.55 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.24it/s]Capturing num tokens (num_tokens=832 avail_mem=26.55 GB):  41%|████▏     | 24/58 [00:08<00:04,  8.31it/s]Capturing num tokens (num_tokens=768 avail_mem=26.54 GB):  41%|████▏     | 24/58 [00:08<00:04,  8.31it/s]Capturing num tokens (num_tokens=704 avail_mem=26.53 GB):  41%|████▏     | 24/58 [00:08<00:04,  8.31it/s]

    Capturing num tokens (num_tokens=704 avail_mem=26.53 GB):  45%|████▍     | 26/58 [00:08<00:03,  9.43it/s]Capturing num tokens (num_tokens=640 avail_mem=26.52 GB):  45%|████▍     | 26/58 [00:08<00:03,  9.43it/s]Capturing num tokens (num_tokens=576 avail_mem=26.51 GB):  45%|████▍     | 26/58 [00:08<00:03,  9.43it/s]Capturing num tokens (num_tokens=576 avail_mem=26.51 GB):  48%|████▊     | 28/58 [00:08<00:02, 10.42it/s]Capturing num tokens (num_tokens=512 avail_mem=26.49 GB):  48%|████▊     | 28/58 [00:08<00:02, 10.42it/s]

    Capturing num tokens (num_tokens=480 avail_mem=26.48 GB):  48%|████▊     | 28/58 [00:09<00:02, 10.42it/s]Capturing num tokens (num_tokens=480 avail_mem=26.48 GB):  52%|█████▏    | 30/58 [00:09<00:02, 11.35it/s]Capturing num tokens (num_tokens=448 avail_mem=26.42 GB):  52%|█████▏    | 30/58 [00:09<00:02, 11.35it/s]Capturing num tokens (num_tokens=416 avail_mem=26.46 GB):  52%|█████▏    | 30/58 [00:09<00:02, 11.35it/s]

    Capturing num tokens (num_tokens=416 avail_mem=26.46 GB):  55%|█████▌    | 32/58 [00:09<00:02, 12.08it/s]Capturing num tokens (num_tokens=384 avail_mem=26.45 GB):  55%|█████▌    | 32/58 [00:09<00:02, 12.08it/s]Capturing num tokens (num_tokens=352 avail_mem=26.43 GB):  55%|█████▌    | 32/58 [00:09<00:02, 12.08it/s]Capturing num tokens (num_tokens=352 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:09<00:01, 13.08it/s]Capturing num tokens (num_tokens=320 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:09<00:01, 13.08it/s]Capturing num tokens (num_tokens=288 avail_mem=26.36 GB):  59%|█████▊    | 34/58 [00:09<00:01, 13.08it/s]

    Capturing num tokens (num_tokens=288 avail_mem=26.36 GB):  62%|██████▏   | 36/58 [00:09<00:01, 13.65it/s]Capturing num tokens (num_tokens=256 avail_mem=26.38 GB):  62%|██████▏   | 36/58 [00:09<00:01, 13.65it/s]Capturing num tokens (num_tokens=240 avail_mem=26.41 GB):  62%|██████▏   | 36/58 [00:09<00:01, 13.65it/s]Capturing num tokens (num_tokens=240 avail_mem=26.41 GB):  66%|██████▌   | 38/58 [00:09<00:01, 14.91it/s]Capturing num tokens (num_tokens=224 avail_mem=26.40 GB):  66%|██████▌   | 38/58 [00:09<00:01, 14.91it/s]Capturing num tokens (num_tokens=208 avail_mem=26.38 GB):  66%|██████▌   | 38/58 [00:09<00:01, 14.91it/s]

    Capturing num tokens (num_tokens=208 avail_mem=26.38 GB):  69%|██████▉   | 40/58 [00:09<00:01, 16.06it/s]Capturing num tokens (num_tokens=192 avail_mem=26.37 GB):  69%|██████▉   | 40/58 [00:09<00:01, 16.06it/s]Capturing num tokens (num_tokens=176 avail_mem=26.36 GB):  69%|██████▉   | 40/58 [00:09<00:01, 16.06it/s]Capturing num tokens (num_tokens=160 avail_mem=26.36 GB):  69%|██████▉   | 40/58 [00:09<00:01, 16.06it/s]Capturing num tokens (num_tokens=160 avail_mem=26.36 GB):  74%|███████▍  | 43/58 [00:09<00:00, 17.47it/s]Capturing num tokens (num_tokens=144 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:00, 17.47it/s]Capturing num tokens (num_tokens=128 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:00, 17.47it/s]

    Capturing num tokens (num_tokens=112 avail_mem=26.34 GB):  74%|███████▍  | 43/58 [00:09<00:00, 17.47it/s]Capturing num tokens (num_tokens=112 avail_mem=26.34 GB):  79%|███████▉  | 46/58 [00:09<00:00, 18.56it/s]Capturing num tokens (num_tokens=96 avail_mem=26.32 GB):  79%|███████▉  | 46/58 [00:09<00:00, 18.56it/s] Capturing num tokens (num_tokens=80 avail_mem=26.31 GB):  79%|███████▉  | 46/58 [00:10<00:00, 18.56it/s]Capturing num tokens (num_tokens=64 avail_mem=26.30 GB):  79%|███████▉  | 46/58 [00:10<00:00, 18.56it/s]Capturing num tokens (num_tokens=64 avail_mem=26.30 GB):  84%|████████▍ | 49/58 [00:10<00:00, 19.67it/s]Capturing num tokens (num_tokens=48 avail_mem=26.27 GB):  84%|████████▍ | 49/58 [00:10<00:00, 19.67it/s]

    Capturing num tokens (num_tokens=32 avail_mem=26.26 GB):  84%|████████▍ | 49/58 [00:10<00:00, 19.67it/s]Capturing num tokens (num_tokens=28 avail_mem=26.28 GB):  84%|████████▍ | 49/58 [00:10<00:00, 19.67it/s]Capturing num tokens (num_tokens=28 avail_mem=26.28 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.01it/s]Capturing num tokens (num_tokens=24 avail_mem=26.26 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.01it/s]Capturing num tokens (num_tokens=20 avail_mem=26.25 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.01it/s]Capturing num tokens (num_tokens=16 avail_mem=26.24 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.01it/s]

    Capturing num tokens (num_tokens=16 avail_mem=26.24 GB):  95%|█████████▍| 55/58 [00:10<00:00, 20.28it/s]Capturing num tokens (num_tokens=12 avail_mem=26.23 GB):  95%|█████████▍| 55/58 [00:10<00:00, 20.28it/s]Capturing num tokens (num_tokens=8 avail_mem=26.21 GB):  95%|█████████▍| 55/58 [00:10<00:00, 20.28it/s] Capturing num tokens (num_tokens=4 avail_mem=26.20 GB):  95%|█████████▍| 55/58 [00:10<00:00, 20.28it/s]Capturing num tokens (num_tokens=4 avail_mem=26.20 GB): 100%|██████████| 58/58 [00:10<00:00, 21.56it/s]Capturing num tokens (num_tokens=4 avail_mem=26.20 GB): 100%|██████████| 58/58 [00:10<00:00,  5.51it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants to know the current date and time, along with the weather. Let me break down how I should approach this.<br><br>First, I need to figure out which functions to use. The user mentioned two functions: get_current_weather and get_current_date. I should use both since the query asks for both the date/time and the weather.<br><br>Starting with the date and time. I remember that the get_current_date function requires a timezone parameter. New York is in the 'America/New_York' timezone, so I'll set that as the parameter. I'll structure the function call accordingly.<br><br>Next, for the weather, the get_current_weather function needs a city, state, and unit. The city is New York, the state is NY, and since the user didn't specify, I'll default to Fahrenheit. I'll make sure to include all these parameters in the function call.<br><br>I should also remember to format each function call correctly, using the specified JSON structure within the tags. Each function call should be on its own line, as per the instructions.<br><br>Putting it all together, I'll write two separate function calls: one for the date/time and another for the weather, each with their respective parameters. This way, the user gets both pieces of information they requested.<br><br><br>content: <br><br><function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '658a41b78d43477cbb5a0c4c1cfff15b', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.006246147677302, 'response_sent_to_client_ts': 1780560147.8783643}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '4b31446e8937421fb5250927f1b8f2dc', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.035677692852914, 'response_sent_to_client_ts': 1780560166.9377916}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '35c91d1219c147c091c8a687b48a7bc8', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14918392430990934, 'response_sent_to_client_ts': 1780560167.1348555}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '6ff7582c6f684a929a769bc9d7aa0765', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14912764448672533, 'response_sent_to_client_ts': 1780560167.1348705}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '5b86f3b302534d379734b1a3b6112ce2', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14906987361609936, 'response_sent_to_client_ts': 1780560167.1348758}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '197e1712e8334a03a64933d7a4d34337', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 21.425900230184197, 'response_sent_to_client_ts': 1780560188.5699074}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Hmm, I need to make sure I understand what they\'re requiring. The capital of France is Paris, so that\'s the key point. \n\nFirst, I should gather the necessary data. Paris\' population is something that\'s known to change over time. As of the latest estimates, I think it\'s around 2.1 million people. I remember reading that the city is continuously growing, so maybe it\'s a bit higher now, but recent data is probably available.\n\nNext, I need to present this information in a JSON format. JSON is all about keys and values, so I\'ll structure it accordingly. The main key should be \'capital\', with the value being \'Paris\'. Then, inside that object, I\'ll have a \'city\' section where I include the population, city area, something unique like a district with a notable monument, and perhaps some languages spoken there since those are common fields.\n\nWait, the user specifically asked for the population, but maybe including additional details like the city area would add value and make the response more comprehensive. Also, mentioning Le Marais with its historical significance could add depth. Including languages spoken might be a nice touch too, as it\'s a multilingual city.\n\nLet me think about the structure again. The root should be an object with \'capital\' as a key, and an object within a \'city\' key containing population, city area, unique districts, and languages. That seems organized and user-friendly.\n\nI should double-check the current population of Paris to ensure accuracy. If I\'m not sure, perhaps I can note that the data is an estimate or based on recent information. Also, I should format everything properly in JSON, using commas and quotation marks correctly to avoid any errors.\n\nWait, did the user mention if they want the population as a number or in words? They just said population, so numerical should be fine. \n\nI should also make sure the JSON is valid. Maybe I\'ll run it through a validator to check. Oh, and the structure should be clean and easy to read, so perhaps indentation is helpful. \n\nPutting it all together, I\'ll start with the outermost object for the city, then the inner objects. Population first, then city area, followed by unique district and languages. Each key should be clear and descriptive to make the data easy to parse.\n\nI think that\'s all. Let me draft the JSON accordingly and present it to the user.\n</think>\n\n```json\n{\n  "city": {\n    "population": 2175000,\n    "city_area": 10.55,\n    "unique_district": "Le Marais",\n    "languages_spoken": ["French", "English", "Spanish", "German"]\n  },\n  "capital": "Paris"\n}\n```', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 88190, 11, 358, 1184, 311, 1281, 2704, 358, 3535, 1128, 807, 2299, 22703, 13, 576, 6722, 315, 9625, 374, 12095, 11, 773, 429, 594, 279, 1376, 1459, 13, 4710, 5338, 11, 358, 1265, 9567, 279, 5871, 821, 13, 12095, 6, 7042, 374, 2494, 429, 594, 3881, 311, 2297, 916, 882, 13, 1634, 315, 279, 5535, 17530, 11, 358, 1744, 432, 594, 2163, 220, 17, 13, 16, 3526, 1251, 13, 358, 6099, 5290, 429, 279, 3283, 374, 30878, 7826, 11, 773, 7196, 432, 594, 264, 2699, 5080, 1431, 11, 714, 3213, 821, 374, 4658, 2500, 382, 5847, 11, 358, 1184, 311, 3042, 419, 1995, 304, 264, 4718, 3561, 13, 4718, 374, 678, 911, 6894, 323, 2750, 11, 773, 358, 3278, 5944, 432, 27079, 13, 576, 1887, 1376, 1265, 387, 364, 65063, 516, 448, 279, 897, 1660, 364, 59604, 4427, 5005, 11, 4766, 429, 1633, 11, 358, 3278, 614, 264, 364, 8926, 6, 3772, 1380, 358, 2924, 279, 7042, 11, 3283, 3082, 11, 2494, 4911, 1075, 264, 9290, 448, 264, 27190, 36897, 11, 323, 8365, 1045, 15459, 21355, 1052, 2474, 1846, 525, 4185, 5043, 382, 14190, 11, 279, 1196, 11689, 4588, 369, 279, 7042, 11, 714, 7196, 2670, 5107, 3565, 1075, 279, 3283, 3082, 1035, 912, 897, 323, 1281, 279, 2033, 803, 15817, 13, 7281, 11, 44291, 1967, 2876, 2782, 448, 1181, 13656, 25361, 1410, 912, 7990, 13, 55121, 15459, 21355, 2578, 387, 264, 6419, 5796, 2238, 11, 438, 432, 594, 264, 2745, 49823, 3283, 382, 10061, 752, 1744, 911, 279, 5944, 1549, 13, 576, 3704, 1265, 387, 458, 1633, 448, 364, 65063, 6, 438, 264, 1376, 11, 323, 458, 1633, 2878, 264, 364, 8926, 6, 1376, 8482, 7042, 11, 3283, 3082, 11, 4911, 26438, 11, 323, 15459, 13, 2938, 4977, 16645, 323, 1196, 21896, 382, 40, 1265, 1990, 15934, 279, 1482, 7042, 315, 12095, 311, 5978, 13403, 13, 1416, 358, 2776, 537, 2704, 11, 8365, 358, 646, 5185, 429, 279, 821, 374, 458, 16045, 476, 3118, 389, 3213, 1995, 13, 7281, 11, 358, 1265, 3561, 4297, 10277, 304, 4718, 11, 1667, 76602, 323, 54231, 15423, 12440, 311, 5648, 894, 5975, 382, 14190, 11, 1521, 279, 1196, 6286, 421, 807, 1366, 279, 7042, 438, 264, 1372, 476, 304, 4244, 30, 2379, 1101, 1053, 7042, 11, 773, 34776, 1265, 387, 6915, 13, 4710, 40, 1265, 1083, 1281, 2704, 279, 4718, 374, 2697, 13, 10696, 358, 3278, 1598, 432, 1526, 264, 22935, 311, 1779, 13, 8670, 11, 323, 279, 5944, 1265, 387, 4240, 323, 4135, 311, 1349, 11, 773, 8365, 69592, 374, 10950, 13, 4710, 97904, 432, 678, 3786, 11, 358, 3278, 1191, 448, 279, 15955, 3562, 1633, 369, 279, 3283, 11, 1221, 279, 9179, 6171, 13, 39529, 1156, 11, 1221, 3283, 3082, 11, 8110, 553, 4911, 9290, 323, 15459, 13, 8886, 1376, 1265, 387, 2797, 323, 52844, 311, 1281, 279, 821, 4135, 311, 4715, 382, 40, 1744, 429, 594, 678, 13, 6771, 752, 9960, 279, 4718, 27079, 323, 3042, 432, 311, 279, 1196, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 8926, 788, 341, 262, 330, 44441, 788, 220, 17, 16, 22, 20, 15, 15, 15, 345, 262, 330, 8926, 15030, 788, 220, 16, 15, 13, 20, 20, 345, 262, 330, 9587, 75440, 788, 330, 2304, 2876, 2782, 756, 262, 330, 42824, 10123, 1679, 788, 4383, 43197, 497, 330, 22574, 497, 330, 61797, 497, 330, 32079, 7026, 220, 1153, 220, 330, 65063, 788, 330, 59604, 698, 532, 73594, 151643], 'meta_info': {'id': '619ccb46d24c4e97bbda7564f92450ec', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 513, 'completion_tokens': 588, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.820028206333518, 'response_sent_to_client_ts': 1780560194.4003813}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.23s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.14s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.15s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:23,  5.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:23,  5.67s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:26,  2.61s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:26,  2.61s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:29,  1.62s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:29,  1.62s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<01:01,  1.14s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<01:01,  1.14s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:45,  1.17it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:45,  1.17it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:35,  1.47it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:35,  1.47it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:28,  1.81it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:28,  1.81it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:23,  2.16it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:23,  2.16it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:19,  2.56it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:19,  2.56it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:16,  2.98it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:16,  2.98it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:13,  3.38it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:13,  3.38it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:12,  3.82it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:12,  3.82it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:10,  4.29it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:10,  4.29it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:09,  4.74it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:09,  4.74it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:08,  5.29it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:08,  5.29it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:07,  5.91it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:07,  5.91it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:06,  6.58it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:06,  6.58it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:09<00:06,  6.58it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:09<00:04,  7.99it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:09<00:04,  7.99it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:10<00:04,  7.99it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:10<00:03,  9.47it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:10<00:03,  9.47it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:10<00:03,  9.47it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:10<00:03, 11.58it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:10<00:03, 11.58it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:10<00:03, 11.58it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:10<00:03, 11.58it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:10<00:02, 14.48it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:10<00:02, 14.48it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:10<00:02, 14.48it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:10<00:02, 14.48it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:10<00:01, 17.21it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:10<00:01, 17.21it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:10<00:01, 17.21it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:10<00:01, 17.21it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:10<00:01, 19.87it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:10<00:01, 19.87it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:10<00:01, 19.87it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:10<00:01, 19.87it/s]

    Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:10<00:01, 19.87it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:10<00:00, 23.47it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:10<00:00, 23.47it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:10<00:00, 23.47it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:10<00:00, 23.47it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:10<00:00, 23.47it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:10<00:00, 23.47it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:10<00:00, 28.28it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:10<00:00, 28.28it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:10<00:00, 28.28it/s]

    Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:10<00:00, 28.28it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:10<00:00, 28.28it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:10<00:00, 30.88it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:10<00:00, 30.88it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:10<00:00, 30.88it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:10<00:00, 30.88it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:11<00:00, 30.88it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:11<00:00, 33.21it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:11<00:00, 33.21it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:11<00:00, 33.21it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:11<00:00, 33.21it/s]

    Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:11<00:00, 33.21it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:11<00:00, 33.21it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:11<00:00, 37.12it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:11<00:00, 37.12it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:11<00:00, 37.12it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:11<00:00, 37.12it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:11<00:00, 37.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  5.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=28.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=28.59 GB):   2%|▏         | 1/58 [00:00<00:32,  1.74it/s]Capturing num tokens (num_tokens=7680 avail_mem=28.52 GB):   2%|▏         | 1/58 [00:00<00:32,  1.74it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=28.52 GB):   3%|▎         | 2/58 [00:01<00:31,  1.79it/s]Capturing num tokens (num_tokens=7168 avail_mem=27.54 GB):   3%|▎         | 2/58 [00:01<00:31,  1.79it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=27.54 GB):   5%|▌         | 3/58 [00:01<00:29,  1.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=27.60 GB):   5%|▌         | 3/58 [00:01<00:29,  1.85it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=27.60 GB):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=27.66 GB):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=27.66 GB):   9%|▊         | 5/58 [00:02<00:25,  2.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=27.73 GB):   9%|▊         | 5/58 [00:02<00:25,  2.07it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=27.73 GB):  10%|█         | 6/58 [00:02<00:22,  2.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=27.79 GB):  10%|█         | 6/58 [00:02<00:22,  2.27it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=27.79 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=27.80 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.44it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=27.80 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=27.86 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.68it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=27.86 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=27.93 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.93it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=27.93 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=27.45 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.15it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=27.45 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=24.69 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.43it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=24.69 GB):  21%|██        | 12/58 [00:04<00:12,  3.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=24.72 GB):  21%|██        | 12/58 [00:04<00:12,  3.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=24.72 GB):  22%|██▏       | 13/58 [00:04<00:11,  4.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=24.75 GB):  22%|██▏       | 13/58 [00:04<00:11,  4.08it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=24.75 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.43it/s]Capturing num tokens (num_tokens=2560 avail_mem=24.78 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.43it/s]Capturing num tokens (num_tokens=2560 avail_mem=24.78 GB):  26%|██▌       | 15/58 [00:05<00:08,  4.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=24.81 GB):  26%|██▌       | 15/58 [00:05<00:08,  4.89it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=24.81 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.83 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.83 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=24.87 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.08it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=24.87 GB):  31%|███       | 18/58 [00:05<00:06,  6.55it/s]Capturing num tokens (num_tokens=1536 avail_mem=25.12 GB):  31%|███       | 18/58 [00:05<00:06,  6.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=24.91 GB):  31%|███       | 18/58 [00:05<00:06,  6.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=24.91 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.93 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s]

    Capturing num tokens (num_tokens=960 avail_mem=24.96 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s] Capturing num tokens (num_tokens=960 avail_mem=24.96 GB):  38%|███▊      | 22/58 [00:05<00:03,  9.66it/s]Capturing num tokens (num_tokens=896 avail_mem=24.96 GB):  38%|███▊      | 22/58 [00:05<00:03,  9.66it/s]Capturing num tokens (num_tokens=832 avail_mem=24.96 GB):  38%|███▊      | 22/58 [00:05<00:03,  9.66it/s]

    Capturing num tokens (num_tokens=832 avail_mem=24.96 GB):  41%|████▏     | 24/58 [00:05<00:03, 11.06it/s]Capturing num tokens (num_tokens=768 avail_mem=25.01 GB):  41%|████▏     | 24/58 [00:05<00:03, 11.06it/s]Capturing num tokens (num_tokens=704 avail_mem=24.96 GB):  41%|████▏     | 24/58 [00:05<00:03, 11.06it/s]Capturing num tokens (num_tokens=704 avail_mem=24.96 GB):  45%|████▍     | 26/58 [00:06<00:02, 12.60it/s]Capturing num tokens (num_tokens=640 avail_mem=24.96 GB):  45%|████▍     | 26/58 [00:06<00:02, 12.60it/s]Capturing num tokens (num_tokens=576 avail_mem=25.05 GB):  45%|████▍     | 26/58 [00:06<00:02, 12.60it/s]

    Capturing num tokens (num_tokens=576 avail_mem=25.05 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.53it/s]Capturing num tokens (num_tokens=512 avail_mem=25.04 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.53it/s]Capturing num tokens (num_tokens=480 avail_mem=25.03 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.53it/s]Capturing num tokens (num_tokens=448 avail_mem=25.02 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.53it/s]Capturing num tokens (num_tokens=448 avail_mem=25.02 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.56it/s]Capturing num tokens (num_tokens=416 avail_mem=24.97 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.56it/s]Capturing num tokens (num_tokens=384 avail_mem=24.96 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.56it/s]

    Capturing num tokens (num_tokens=352 avail_mem=24.96 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.56it/s]Capturing num tokens (num_tokens=352 avail_mem=24.96 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.97it/s]Capturing num tokens (num_tokens=320 avail_mem=24.97 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.97it/s]Capturing num tokens (num_tokens=288 avail_mem=24.96 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.97it/s]Capturing num tokens (num_tokens=256 avail_mem=24.96 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.97it/s]

    Capturing num tokens (num_tokens=256 avail_mem=24.96 GB):  64%|██████▍   | 37/58 [00:06<00:01, 15.04it/s]Capturing num tokens (num_tokens=240 avail_mem=24.94 GB):  64%|██████▍   | 37/58 [00:06<00:01, 15.04it/s]Capturing num tokens (num_tokens=224 avail_mem=24.94 GB):  64%|██████▍   | 37/58 [00:06<00:01, 15.04it/s]Capturing num tokens (num_tokens=208 avail_mem=24.93 GB):  64%|██████▍   | 37/58 [00:06<00:01, 15.04it/s]Capturing num tokens (num_tokens=208 avail_mem=24.93 GB):  69%|██████▉   | 40/58 [00:06<00:01, 17.76it/s]Capturing num tokens (num_tokens=192 avail_mem=24.93 GB):  69%|██████▉   | 40/58 [00:06<00:01, 17.76it/s]Capturing num tokens (num_tokens=176 avail_mem=24.92 GB):  69%|██████▉   | 40/58 [00:06<00:01, 17.76it/s]Capturing num tokens (num_tokens=160 avail_mem=24.94 GB):  69%|██████▉   | 40/58 [00:06<00:01, 17.76it/s]

    Capturing num tokens (num_tokens=160 avail_mem=24.94 GB):  74%|███████▍  | 43/58 [00:06<00:00, 19.31it/s]Capturing num tokens (num_tokens=144 avail_mem=24.94 GB):  74%|███████▍  | 43/58 [00:06<00:00, 19.31it/s]Capturing num tokens (num_tokens=128 avail_mem=24.94 GB):  74%|███████▍  | 43/58 [00:06<00:00, 19.31it/s]Capturing num tokens (num_tokens=112 avail_mem=24.94 GB):  74%|███████▍  | 43/58 [00:06<00:00, 19.31it/s]Capturing num tokens (num_tokens=112 avail_mem=24.94 GB):  79%|███████▉  | 46/58 [00:07<00:00, 21.14it/s]Capturing num tokens (num_tokens=96 avail_mem=24.93 GB):  79%|███████▉  | 46/58 [00:07<00:00, 21.14it/s] Capturing num tokens (num_tokens=80 avail_mem=24.92 GB):  79%|███████▉  | 46/58 [00:07<00:00, 21.14it/s]Capturing num tokens (num_tokens=64 avail_mem=24.91 GB):  79%|███████▉  | 46/58 [00:07<00:00, 21.14it/s]

    Capturing num tokens (num_tokens=64 avail_mem=24.91 GB):  84%|████████▍ | 49/58 [00:07<00:00, 22.73it/s]Capturing num tokens (num_tokens=48 avail_mem=24.90 GB):  84%|████████▍ | 49/58 [00:07<00:00, 22.73it/s]Capturing num tokens (num_tokens=32 avail_mem=24.90 GB):  84%|████████▍ | 49/58 [00:07<00:00, 22.73it/s]Capturing num tokens (num_tokens=28 avail_mem=24.89 GB):  84%|████████▍ | 49/58 [00:07<00:00, 22.73it/s]Capturing num tokens (num_tokens=28 avail_mem=24.89 GB):  90%|████████▉ | 52/58 [00:07<00:00, 24.19it/s]Capturing num tokens (num_tokens=24 avail_mem=24.89 GB):  90%|████████▉ | 52/58 [00:07<00:00, 24.19it/s]Capturing num tokens (num_tokens=20 avail_mem=24.86 GB):  90%|████████▉ | 52/58 [00:07<00:00, 24.19it/s]Capturing num tokens (num_tokens=16 avail_mem=24.85 GB):  90%|████████▉ | 52/58 [00:07<00:00, 24.19it/s]

    Capturing num tokens (num_tokens=16 avail_mem=24.85 GB):  95%|█████████▍| 55/58 [00:07<00:00, 25.49it/s]Capturing num tokens (num_tokens=12 avail_mem=24.84 GB):  95%|█████████▍| 55/58 [00:07<00:00, 25.49it/s]Capturing num tokens (num_tokens=8 avail_mem=24.86 GB):  95%|█████████▍| 55/58 [00:07<00:00, 25.49it/s] Capturing num tokens (num_tokens=4 avail_mem=24.85 GB):  95%|█████████▍| 55/58 [00:07<00:00, 25.49it/s]Capturing num tokens (num_tokens=4 avail_mem=24.85 GB): 100%|██████████| 58/58 [00:07<00:00, 25.90it/s]Capturing num tokens (num_tokens=4 avail_mem=24.85 GB): 100%|██████████| 58/58 [00:07<00:00,  7.80it/s]


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
    
    Generated text: Alright, the user is asking for the information and population of the capital of France in JSON format. So first, I need to figure out who the user is and what they're using this for.
    
    They might be a student working on a project or maybe a casual learner interested in geography and demographics. Either way, they need a straightforward and accurate response.
    
    The capital of France is definitely Paris. I should confirm that to make sure. I'll state that Paris is the capital of France.
    
    Now, for the population. I remember that Paris has a large population, but the exact number can change. The latest data I have is from 2023, which puts it at around 2,155,000. But I should note that this can vary a bit depending on the source and the year.
    
    Putting it all together in JSON format is straightforward. I'll structure it with "capital" and "population" as keys. Under "capital," I'll have a string, and "population" will be a number. I need to make sure the syntax is correct so it doesn't break when they use it in their application or whatever they're working on.
    
    I'll also add a brief summary to explain the response clearly. That should cover the user's needs effectively.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "capital": "Paris",
      "population": 2155000
    }
    ```



```python
llm.shutdown()
```
