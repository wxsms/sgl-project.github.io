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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.58s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.48s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.50s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:54,  5.17s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:54,  5.17s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:06,  2.25s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:06,  2.25s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.15it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:28,  1.81it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:28,  1.81it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:23,  2.15it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:23,  2.15it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:20,  2.47it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:20,  2.47it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:17,  2.86it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:17,  2.86it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:14,  3.26it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:14,  3.26it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:12,  3.68it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:12,  3.68it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:11,  4.07it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:11,  4.07it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:10,  4.47it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:10,  4.47it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:09,  4.87it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:09,  4.87it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:08,  5.37it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:08,  5.37it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:07,  5.92it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:07,  5.92it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:08<00:06,  6.52it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:08<00:06,  6.52it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:08<00:05,  7.20it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:08<00:05,  7.20it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:08<00:05,  7.20it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:08<00:04,  9.43it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:08<00:04,  9.43it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:08<00:04,  9.43it/s] 

    Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:08<00:03, 11.36it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:08<00:03, 11.36it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:08<00:03, 11.36it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:08<00:02, 12.70it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:08<00:02, 12.70it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:09<00:02, 12.70it/s]

    Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:09<00:02, 12.70it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:09<00:01, 15.74it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:09<00:01, 15.74it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:09<00:01, 15.74it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:09<00:01, 15.74it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:09<00:01, 18.51it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:09<00:01, 18.51it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:09<00:01, 18.51it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:09<00:01, 18.51it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:09<00:01, 19.30it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:09<00:01, 19.30it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:09<00:01, 19.30it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:09<00:01, 19.30it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:09<00:01, 20.95it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:09<00:01, 20.95it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:09<00:01, 20.95it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:09<00:01, 20.95it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:09<00:01, 20.95it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:09<00:00, 23.76it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:09<00:00, 23.76it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:09<00:00, 23.76it/s]

    Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:09<00:00, 23.76it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:09<00:00, 23.76it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:09<00:00, 27.64it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:09<00:00, 27.64it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:09<00:00, 27.64it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:09<00:00, 27.64it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:09<00:00, 27.64it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:09<00:00, 29.36it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:09<00:00, 29.36it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:09<00:00, 29.36it/s]

    Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:09<00:00, 29.36it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:09<00:00, 29.51it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:09<00:00, 29.51it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:09<00:00, 29.51it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:10<00:00, 29.51it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:10<00:00, 29.51it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:10<00:00, 31.64it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:10<00:00, 31.64it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:10<00:00, 31.64it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:10<00:00, 31.64it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.94 GB):   2%|▏         | 1/58 [00:00<00:37,  1.51it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.91 GB):   2%|▏         | 1/58 [00:00<00:37,  1.51it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.91 GB):   3%|▎         | 2/58 [00:01<00:31,  1.77it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.91 GB):   3%|▎         | 2/58 [00:01<00:31,  1.77it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.91 GB):   5%|▌         | 3/58 [00:01<00:22,  2.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.91 GB):   5%|▌         | 3/58 [00:01<00:22,  2.40it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.91 GB):   7%|▋         | 4/58 [00:01<00:18,  2.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.91 GB):   7%|▋         | 4/58 [00:01<00:18,  2.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.91 GB):   9%|▊         | 5/58 [00:01<00:17,  3.01it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.91 GB):   9%|▊         | 5/58 [00:01<00:17,  3.01it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.91 GB):  10%|█         | 6/58 [00:02<00:18,  2.85it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.91 GB):  10%|█         | 6/58 [00:02<00:18,  2.85it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.91 GB):  12%|█▏        | 7/58 [00:02<00:18,  2.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.91 GB):  12%|█▏        | 7/58 [00:02<00:18,  2.73it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.91 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.91 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.69it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.91 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.91 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.91 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.91 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.41it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.91 GB):  19%|█▉        | 11/58 [00:03<00:11,  3.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.64 GB):  19%|█▉        | 11/58 [00:03<00:11,  3.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.64 GB):  21%|██        | 12/58 [00:03<00:09,  4.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.63 GB):  21%|██        | 12/58 [00:03<00:09,  4.84it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=61.63 GB):  21%|██        | 12/58 [00:03<00:09,  4.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.63 GB):  24%|██▍       | 14/58 [00:04<00:06,  6.47it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.63 GB):  24%|██▍       | 14/58 [00:04<00:06,  6.47it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.63 GB):  24%|██▍       | 14/58 [00:04<00:06,  6.47it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=61.63 GB):  28%|██▊       | 16/58 [00:04<00:05,  8.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.62 GB):  28%|██▊       | 16/58 [00:04<00:05,  8.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.62 GB):  28%|██▊       | 16/58 [00:04<00:05,  8.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.62 GB):  31%|███       | 18/58 [00:04<00:04,  9.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.62 GB):  31%|███       | 18/58 [00:04<00:04,  9.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.62 GB):  31%|███       | 18/58 [00:04<00:04,  9.93it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=61.61 GB):  31%|███       | 18/58 [00:04<00:04,  9.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.61 GB):  36%|███▌      | 21/58 [00:04<00:02, 13.07it/s]Capturing num tokens (num_tokens=960 avail_mem=61.60 GB):  36%|███▌      | 21/58 [00:04<00:02, 13.07it/s] Capturing num tokens (num_tokens=896 avail_mem=61.60 GB):  36%|███▌      | 21/58 [00:04<00:02, 13.07it/s]Capturing num tokens (num_tokens=832 avail_mem=61.60 GB):  36%|███▌      | 21/58 [00:04<00:02, 13.07it/s]Capturing num tokens (num_tokens=832 avail_mem=61.60 GB):  41%|████▏     | 24/58 [00:04<00:02, 16.18it/s]Capturing num tokens (num_tokens=768 avail_mem=61.59 GB):  41%|████▏     | 24/58 [00:04<00:02, 16.18it/s]Capturing num tokens (num_tokens=704 avail_mem=61.59 GB):  41%|████▏     | 24/58 [00:04<00:02, 16.18it/s]

    Capturing num tokens (num_tokens=640 avail_mem=61.59 GB):  41%|████▏     | 24/58 [00:04<00:02, 16.18it/s]Capturing num tokens (num_tokens=640 avail_mem=61.59 GB):  47%|████▋     | 27/58 [00:04<00:01, 19.24it/s]Capturing num tokens (num_tokens=576 avail_mem=61.58 GB):  47%|████▋     | 27/58 [00:04<00:01, 19.24it/s]Capturing num tokens (num_tokens=512 avail_mem=61.58 GB):  47%|████▋     | 27/58 [00:04<00:01, 19.24it/s]Capturing num tokens (num_tokens=480 avail_mem=61.57 GB):  47%|████▋     | 27/58 [00:04<00:01, 19.24it/s]Capturing num tokens (num_tokens=448 avail_mem=61.57 GB):  47%|████▋     | 27/58 [00:04<00:01, 19.24it/s]Capturing num tokens (num_tokens=448 avail_mem=61.57 GB):  53%|█████▎    | 31/58 [00:04<00:01, 22.86it/s]Capturing num tokens (num_tokens=416 avail_mem=61.57 GB):  53%|█████▎    | 31/58 [00:04<00:01, 22.86it/s]Capturing num tokens (num_tokens=384 avail_mem=61.56 GB):  53%|█████▎    | 31/58 [00:04<00:01, 22.86it/s]

    Capturing num tokens (num_tokens=352 avail_mem=61.56 GB):  53%|█████▎    | 31/58 [00:04<00:01, 22.86it/s]Capturing num tokens (num_tokens=352 avail_mem=61.56 GB):  59%|█████▊    | 34/58 [00:04<00:00, 24.53it/s]Capturing num tokens (num_tokens=320 avail_mem=61.55 GB):  59%|█████▊    | 34/58 [00:04<00:00, 24.53it/s]Capturing num tokens (num_tokens=288 avail_mem=61.56 GB):  59%|█████▊    | 34/58 [00:04<00:00, 24.53it/s]Capturing num tokens (num_tokens=256 avail_mem=61.55 GB):  59%|█████▊    | 34/58 [00:04<00:00, 24.53it/s]Capturing num tokens (num_tokens=240 avail_mem=61.55 GB):  59%|█████▊    | 34/58 [00:04<00:00, 24.53it/s]Capturing num tokens (num_tokens=240 avail_mem=61.55 GB):  66%|██████▌   | 38/58 [00:04<00:00, 28.24it/s]Capturing num tokens (num_tokens=224 avail_mem=61.55 GB):  66%|██████▌   | 38/58 [00:04<00:00, 28.24it/s]Capturing num tokens (num_tokens=208 avail_mem=61.54 GB):  66%|██████▌   | 38/58 [00:05<00:00, 28.24it/s]Capturing num tokens (num_tokens=192 avail_mem=61.54 GB):  66%|██████▌   | 38/58 [00:05<00:00, 28.24it/s]

    Capturing num tokens (num_tokens=176 avail_mem=61.54 GB):  66%|██████▌   | 38/58 [00:05<00:00, 28.24it/s]Capturing num tokens (num_tokens=176 avail_mem=61.54 GB):  72%|███████▏  | 42/58 [00:05<00:00, 31.27it/s]Capturing num tokens (num_tokens=160 avail_mem=61.53 GB):  72%|███████▏  | 42/58 [00:05<00:00, 31.27it/s]Capturing num tokens (num_tokens=144 avail_mem=61.53 GB):  72%|███████▏  | 42/58 [00:05<00:00, 31.27it/s]Capturing num tokens (num_tokens=128 avail_mem=61.53 GB):  72%|███████▏  | 42/58 [00:05<00:00, 31.27it/s]Capturing num tokens (num_tokens=112 avail_mem=61.53 GB):  72%|███████▏  | 42/58 [00:05<00:00, 31.27it/s]Capturing num tokens (num_tokens=112 avail_mem=61.53 GB):  79%|███████▉  | 46/58 [00:05<00:00, 33.41it/s]Capturing num tokens (num_tokens=96 avail_mem=61.52 GB):  79%|███████▉  | 46/58 [00:05<00:00, 33.41it/s] Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  79%|███████▉  | 46/58 [00:05<00:00, 33.41it/s]Capturing num tokens (num_tokens=64 avail_mem=61.51 GB):  79%|███████▉  | 46/58 [00:05<00:00, 33.41it/s]

    Capturing num tokens (num_tokens=48 avail_mem=61.51 GB):  79%|███████▉  | 46/58 [00:05<00:00, 33.41it/s]Capturing num tokens (num_tokens=32 avail_mem=61.51 GB):  79%|███████▉  | 46/58 [00:05<00:00, 33.41it/s]Capturing num tokens (num_tokens=32 avail_mem=61.51 GB):  88%|████████▊ | 51/58 [00:05<00:00, 35.55it/s]Capturing num tokens (num_tokens=28 avail_mem=61.51 GB):  88%|████████▊ | 51/58 [00:05<00:00, 35.55it/s]Capturing num tokens (num_tokens=24 avail_mem=61.50 GB):  88%|████████▊ | 51/58 [00:05<00:00, 35.55it/s]Capturing num tokens (num_tokens=20 avail_mem=61.50 GB):  88%|████████▊ | 51/58 [00:05<00:00, 35.55it/s]Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  88%|████████▊ | 51/58 [00:05<00:00, 35.55it/s]Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  95%|█████████▍| 55/58 [00:05<00:00, 36.57it/s]Capturing num tokens (num_tokens=12 avail_mem=61.49 GB):  95%|█████████▍| 55/58 [00:05<00:00, 36.57it/s]Capturing num tokens (num_tokens=8 avail_mem=61.49 GB):  95%|█████████▍| 55/58 [00:05<00:00, 36.57it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=61.49 GB):  95%|█████████▍| 55/58 [00:05<00:00, 36.57it/s]Capturing num tokens (num_tokens=4 avail_mem=61.49 GB): 100%|██████████| 58/58 [00:05<00:00, 10.55it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants the current date and time along with the weather. I need to figure out how to respond using the allowed functions. First, I should identify which functions to use. The user mentioned two specific functions: 'get_current_weather' and 'get_current_date'. <br><br>I'll start with the date because it's straightforward. The function requires a timezone parameter. Since the user is in New York, I can assume the timezone is 'America/New_York'. So I'll call 'get_current_date' with that parameter.<br><br>Next, for the weather, I need the city and state. The user is in New York, so the city is 'New York' and the state is 'NY'. The unit isn't specified, so I'll default to Celsius. I'll call 'get_current_weather' with city, state, and unit as parameters.<br><br>I should structure the response by first calling 'get_current_date' with the timezone and then 'get_current_weather' with the necessary details. I'll make sure each function call is on a separate line and follows the required format. Also, I'll include the sources in the response to show where the information came from.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "celsius"}</function>  <br><br>Sources:  <br>- Date and Time retrieved using 'get_current_date' function  <br>- Weather data retrieved using 'get_current_weather' function</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '8e4e4eee751d45da82ee60675c7d5db9', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 6.240534095093608, 'response_sent_to_client_ts': 1779093413.1458063}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '7f76605b557346a99b7743921f834da2', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.286185767501593, 'response_sent_to_client_ts': 1779093417.4414675}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'be7c0c7aecf6471097bd92cc0718b937', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09599060006439686, 'response_sent_to_client_ts': 1779093417.5626361}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '1249de5b349740039dfd04ede0c38fc7', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.0958926547318697, 'response_sent_to_client_ts': 1779093417.5626478}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '8acb35828c3e4a3fad076612e00f6380', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09585428982973099, 'response_sent_to_client_ts': 1779093417.5626514}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '182d72f556de403390e623df308b5cbc', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 17.128341175615788, 'response_sent_to_client_ts': 1779093434.698863}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so I need to figure out how to respond to this query. The user is asking for information and the population of France\'s capital in JSON format. Hmm, the capital of France is Paris, I remember that. Now, I should gather the necessary data about Paris.\n\nFirst, I need to find out the population. I think the population of Paris is over 3 million, but I\'m not exactly sure. I recall that Paris is the most populous city in France, so the population figure should be quite large. Let me check my memory or maybe think about recent stats. I believe it\'s around 3.5 to 4 million people as of 2023. I should verify that.\n\nNext, I should include other relevant information about Paris. Paris is known for its history, being the capital for centuries, so that\'s an important point. It\'s also a major cultural and economic center. The approximate area of the urban area is another piece of information. I think it\'s about 104 square kilometers, but I\'m not entirely certain. I should make sure to get that number right.\n\nAlso, considering recent data, I know that city populations in France can fluctuate a bit, so it\'s good to mention that the figure is approximate. Similarly, the population of the urban area can sometimes change due to factors like migration. I\'ll keep that in mind when structuring the information.\n\nNow, thinking about the JSON format. The user wants the information in JSON, so I need to structure it properly. I\'ll create an object with a headline title and an "information" key containing the details. Then, an "approximate_population" key with the population number, and an "approximate_population urban" key with the urban area figure.\n\nI should also include geographical coordinates for Paris. Those are standard and helpful for mapping purposes. I\'ll note them as latitude and longitude.\n\nPutting it all together, I\'ll structure the JSON with the main title as "Information about the Capital of France," then include the title, various pieces of information, population data, and geographical details. I\'ll make sure everything is concise and accurate as possible.\n\nI might also consider the user\'s potential intent. They might be a student needing this info for a project or a researcher compiling data. Either way, providing accurate and up-to-date information is crucial. I\'ll double-check the population numbers to ensure they\'re recent and reliable.\n\nIn summary, I\'ll structure the JSON with key information about Paris\'s population, notable facts, and geographical data, making sure the figures are current and the content is clear and helpful.\n</think>\n\nHere is the information about the capital of France, Paris, in JSON format:\n\n```json\n{\n  "title": "Information about the Capital of France",\n  "description": "Details about Paris, the capital city of France.",\n  "title2": "Paris",\n  "population": {\n    "city": "3,500,000",\n    "approximate": "Approximately 3.5 million as of 2023."\n  },\n  "population_urban": {\n    "area": "104 km²",\n    "approximate": "Approximately 104 square kilometers (32.3 square miles)."\n  },\n  "notable_facts": [\n    {\n      "text": "Paris is the most populous city in France."\n    },\n    {\n      "text": "It is also the economic and cultural capital of France."\n    },\n    {\n      "text": "Paris has been the capital of France since the Battle of Crécy in 1346."\n    }\n  ],\n  "geographical_coordinates": {\n    "latitude": "48.8566",\n    "longitude": "-2.3522"\n  }\n}\n```', 'output_ids': [32313, 11, 773, 358, 1184, 311, 7071, 700, 1246, 311, 5889, 311, 419, 3239, 13, 576, 1196, 374, 10161, 369, 1995, 323, 279, 7042, 315, 9625, 594, 6722, 304, 4718, 3561, 13, 88190, 11, 279, 6722, 315, 9625, 374, 12095, 11, 358, 6099, 429, 13, 4695, 11, 358, 1265, 9567, 279, 5871, 821, 911, 12095, 382, 5338, 11, 358, 1184, 311, 1477, 700, 279, 7042, 13, 358, 1744, 279, 7042, 315, 12095, 374, 916, 220, 18, 3526, 11, 714, 358, 2776, 537, 6896, 2704, 13, 358, 19091, 429, 12095, 374, 279, 1429, 94451, 3283, 304, 9625, 11, 773, 279, 7042, 7071, 1265, 387, 5008, 3460, 13, 6771, 752, 1779, 847, 4938, 476, 7196, 1744, 911, 3213, 10472, 13, 358, 4411, 432, 594, 2163, 220, 18, 13, 20, 311, 220, 19, 3526, 1251, 438, 315, 220, 17, 15, 17, 18, 13, 358, 1265, 10146, 429, 382, 5847, 11, 358, 1265, 2924, 1008, 9760, 1995, 911, 12095, 13, 12095, 374, 3881, 369, 1181, 3840, 11, 1660, 279, 6722, 369, 23631, 11, 773, 429, 594, 458, 2989, 1459, 13, 1084, 594, 1083, 264, 3598, 12752, 323, 6955, 4126, 13, 576, 44868, 3082, 315, 279, 15662, 3082, 374, 2441, 6573, 315, 1995, 13, 358, 1744, 432, 594, 911, 220, 16, 15, 19, 9334, 40568, 11, 714, 358, 2776, 537, 11368, 3654, 13, 358, 1265, 1281, 2704, 311, 633, 429, 1372, 1290, 382, 13394, 11, 12831, 3213, 821, 11, 358, 1414, 429, 3283, 21910, 304, 9625, 646, 38288, 6292, 264, 2699, 11, 773, 432, 594, 1661, 311, 6286, 429, 279, 7071, 374, 44868, 13, 34239, 11, 279, 7042, 315, 279, 15662, 3082, 646, 7025, 2297, 4152, 311, 9363, 1075, 11906, 13, 358, 3278, 2506, 429, 304, 3971, 979, 2036, 1677, 279, 1995, 382, 7039, 11, 7274, 911, 279, 4718, 3561, 13, 576, 1196, 6801, 279, 1995, 304, 4718, 11, 773, 358, 1184, 311, 5944, 432, 10277, 13, 358, 3278, 1855, 458, 1633, 448, 264, 31163, 2265, 323, 458, 330, 25069, 1, 1376, 8482, 279, 3565, 13, 5005, 11, 458, 330, 48053, 3426, 74572, 1, 1376, 448, 279, 7042, 1372, 11, 323, 458, 330, 48053, 3426, 74572, 15662, 1, 1376, 448, 279, 15662, 3082, 7071, 382, 40, 1265, 1083, 2924, 52901, 13934, 369, 12095, 13, 12969, 525, 5297, 323, 10950, 369, 12731, 9895, 13, 358, 3278, 5185, 1105, 438, 20849, 323, 20515, 382, 97904, 432, 678, 3786, 11, 358, 3278, 5944, 279, 4718, 448, 279, 1887, 2265, 438, 330, 14873, 911, 279, 18374, 315, 9625, 1335, 1221, 2924, 279, 2265, 11, 5257, 9666, 315, 1995, 11, 7042, 821, 11, 323, 52901, 3565, 13, 358, 3278, 1281, 2704, 4297, 374, 63594, 323, 13382, 438, 3204, 382, 40, 2578, 1083, 2908, 279, 1196, 594, 4650, 7385, 13, 2379, 2578, 387, 264, 5458, 32821, 419, 3546, 369, 264, 2390, 476, 264, 31085, 54220, 821, 13, 20988, 1616, 11, 8241, 13382, 323, 705, 4686, 18413, 1995, 374, 16587, 13, 358, 3278, 1990, 15934, 279, 7042, 5109, 311, 5978, 807, 2299, 3213, 323, 14720, 382, 641, 12126, 11, 358, 3278, 5944, 279, 4718, 448, 1376, 1995, 911, 12095, 594, 7042, 11, 27190, 13064, 11, 323, 52901, 821, 11, 3259, 2704, 279, 12396, 525, 1482, 323, 279, 2213, 374, 2797, 323, 10950, 624, 151649, 271, 8420, 374, 279, 1995, 911, 279, 6722, 315, 9625, 11, 12095, 11, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 2102, 788, 330, 14873, 911, 279, 18374, 315, 9625, 756, 220, 330, 4684, 788, 330, 7799, 911, 12095, 11, 279, 6722, 3283, 315, 9625, 10346, 220, 330, 2102, 17, 788, 330, 59604, 756, 220, 330, 44441, 788, 341, 262, 330, 8926, 788, 330, 18, 11, 20, 15, 15, 11, 15, 15, 15, 756, 262, 330, 48053, 3426, 788, 330, 69520, 7108, 220, 18, 13, 20, 3526, 438, 315, 220, 17, 15, 17, 18, 10040, 220, 1153, 220, 330, 44441, 62, 59059, 788, 341, 262, 330, 4798, 788, 330, 16, 15, 19, 13136, 29456, 756, 262, 330, 48053, 3426, 788, 330, 69520, 7108, 220, 16, 15, 19, 9334, 40568, 320, 18, 17, 13, 18, 9334, 8756, 568, 698, 220, 1153, 220, 330, 1921, 480, 761, 11359, 788, 2278, 262, 341, 414, 330, 1318, 788, 330, 59604, 374, 279, 1429, 94451, 3283, 304, 9625, 10040, 262, 1153, 262, 341, 414, 330, 1318, 788, 330, 2132, 374, 1083, 279, 6955, 323, 12752, 6722, 315, 9625, 10040, 262, 1153, 262, 341, 414, 330, 1318, 788, 330, 59604, 702, 1012, 279, 6722, 315, 9625, 2474, 279, 16115, 315, 135893, 11130, 304, 220, 16, 18, 19, 21, 10040, 262, 456, 220, 3211, 220, 330, 709, 31177, 50084, 788, 341, 262, 330, 23718, 788, 330, 19, 23, 13, 23, 20, 21, 21, 756, 262, 330, 25446, 788, 6523, 17, 13, 18, 20, 17, 17, 698, 220, 456, 532, 73594, 151643], 'meta_info': {'id': '1e5e06fed2214eb18701a8c07f951888', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 532, 'completion_tokens': 788, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 6.52350066229701, 'response_sent_to_client_ts': 1779093441.2305374}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.47s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.46s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.46s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:45,  5.00s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:45,  5.00s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:09,  2.32s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:09,  2.32s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:19,  1.44s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:19,  1.44s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:54,  1.00s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:54,  1.00s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:39,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:39,  1.36it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:29,  1.78it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:29,  1.78it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:22,  2.28it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:22,  2.28it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:17,  2.85it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:17,  2.85it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:13,  3.51it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:13,  3.51it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:11,  4.25it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:11,  4.25it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:09,  4.98it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:09,  4.98it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:09,  4.98it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  6.92it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  6.92it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  6.92it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  8.77it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  8.77it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  8.77it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:07<00:03, 11.85it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:07<00:03, 11.85it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:08<00:03, 11.85it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:08<00:02, 12.87it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:08<00:02, 12.87it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:08<00:02, 12.87it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:08<00:02, 14.00it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:08<00:02, 14.00it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:08<00:02, 14.00it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:08<00:02, 14.00it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:08<00:01, 16.61it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:08<00:01, 16.61it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:08<00:01, 16.61it/s]

    Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:08<00:01, 16.61it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:08<00:01, 19.71it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:08<00:01, 19.71it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:08<00:01, 19.71it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:08<00:01, 19.71it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:08<00:01, 21.62it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:08<00:01, 21.62it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:08<00:01, 21.62it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:08<00:01, 21.62it/s]

    Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:08<00:01, 21.62it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:08<00:00, 25.48it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:08<00:00, 25.48it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:08<00:00, 25.48it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:08<00:00, 25.48it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:08<00:00, 25.48it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:08<00:00, 25.48it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:08<00:00, 30.76it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:08<00:00, 30.76it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:08<00:00, 30.76it/s]

    Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:08<00:00, 30.76it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:08<00:00, 30.76it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:08<00:00, 30.76it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:08<00:00, 34.04it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:08<00:00, 34.04it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:08<00:00, 34.04it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:08<00:00, 34.04it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:08<00:00, 34.04it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:08<00:00, 34.04it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:08<00:00, 37.25it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:08<00:00, 37.25it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:08<00:00, 37.25it/s]

    Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:09<00:00, 37.25it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:09<00:00, 37.25it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:09<00:00, 37.25it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:09<00:00, 37.25it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:09<00:00, 37.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00, 43.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  6.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=56.03 GB):   2%|▏         | 1/58 [00:00<00:21,  2.64it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.01 GB):   2%|▏         | 1/58 [00:00<00:21,  2.64it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=56.01 GB):   3%|▎         | 2/58 [00:00<00:19,  2.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.00 GB):   3%|▎         | 2/58 [00:00<00:19,  2.85it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=56.00 GB):   5%|▌         | 3/58 [00:01<00:17,  3.08it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.98 GB):   5%|▌         | 3/58 [00:01<00:17,  3.08it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.98 GB):   7%|▋         | 4/58 [00:01<00:16,  3.34it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.98 GB):   7%|▋         | 4/58 [00:01<00:16,  3.34it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.98 GB):   9%|▊         | 5/58 [00:01<00:14,  3.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.97 GB):   9%|▊         | 5/58 [00:01<00:14,  3.56it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.97 GB):  10%|█         | 6/58 [00:01<00:13,  3.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.96 GB):  10%|█         | 6/58 [00:01<00:13,  3.89it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=55.96 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.93 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.93 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.94 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.63it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=55.94 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.94 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.94 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.93 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.45it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=55.93 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.92 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.92 GB):  21%|██        | 12/58 [00:02<00:07,  6.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.91 GB):  21%|██        | 12/58 [00:02<00:07,  6.40it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=55.91 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.90 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.90 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.47it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.89 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.47it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.88 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.47it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=55.88 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.88 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.88 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.88 GB):  31%|███       | 18/58 [00:03<00:03, 11.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.87 GB):  31%|███       | 18/58 [00:03<00:03, 11.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.87 GB):  31%|███       | 18/58 [00:03<00:03, 11.12it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=55.86 GB):  31%|███       | 18/58 [00:03<00:03, 11.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.86 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.47it/s]Capturing num tokens (num_tokens=960 avail_mem=55.86 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.47it/s] Capturing num tokens (num_tokens=896 avail_mem=55.85 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.47it/s]Capturing num tokens (num_tokens=832 avail_mem=55.85 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.47it/s]Capturing num tokens (num_tokens=832 avail_mem=55.85 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.63it/s]Capturing num tokens (num_tokens=768 avail_mem=55.85 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.63it/s]Capturing num tokens (num_tokens=704 avail_mem=55.84 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.63it/s]

    Capturing num tokens (num_tokens=640 avail_mem=55.84 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.63it/s]Capturing num tokens (num_tokens=576 avail_mem=55.84 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.63it/s]Capturing num tokens (num_tokens=576 avail_mem=55.84 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.47it/s]Capturing num tokens (num_tokens=512 avail_mem=55.83 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.47it/s]Capturing num tokens (num_tokens=480 avail_mem=55.83 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.47it/s]Capturing num tokens (num_tokens=448 avail_mem=55.83 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.47it/s]Capturing num tokens (num_tokens=416 avail_mem=55.82 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.47it/s]Capturing num tokens (num_tokens=416 avail_mem=55.82 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.76it/s]Capturing num tokens (num_tokens=384 avail_mem=55.82 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.76it/s]

    Capturing num tokens (num_tokens=352 avail_mem=55.81 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.76it/s]Capturing num tokens (num_tokens=320 avail_mem=55.81 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.76it/s]Capturing num tokens (num_tokens=288 avail_mem=55.82 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.76it/s]Capturing num tokens (num_tokens=288 avail_mem=55.82 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.03it/s]Capturing num tokens (num_tokens=256 avail_mem=55.81 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.03it/s]Capturing num tokens (num_tokens=240 avail_mem=55.81 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.03it/s]Capturing num tokens (num_tokens=224 avail_mem=55.80 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.03it/s]Capturing num tokens (num_tokens=208 avail_mem=55.80 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.03it/s]Capturing num tokens (num_tokens=208 avail_mem=55.80 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.00it/s]Capturing num tokens (num_tokens=192 avail_mem=55.80 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.00it/s]

    Capturing num tokens (num_tokens=176 avail_mem=55.79 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.00it/s]Capturing num tokens (num_tokens=160 avail_mem=55.79 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.00it/s]Capturing num tokens (num_tokens=144 avail_mem=55.78 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.00it/s]Capturing num tokens (num_tokens=144 avail_mem=55.78 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.27it/s]Capturing num tokens (num_tokens=128 avail_mem=55.79 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.27it/s]Capturing num tokens (num_tokens=112 avail_mem=55.78 GB):  76%|███████▌  | 44/58 [00:04<00:00, 33.27it/s]Capturing num tokens (num_tokens=96 avail_mem=55.78 GB):  76%|███████▌  | 44/58 [00:04<00:00, 33.27it/s] Capturing num tokens (num_tokens=80 avail_mem=55.77 GB):  76%|███████▌  | 44/58 [00:04<00:00, 33.27it/s]Capturing num tokens (num_tokens=64 avail_mem=55.77 GB):  76%|███████▌  | 44/58 [00:04<00:00, 33.27it/s]Capturing num tokens (num_tokens=64 avail_mem=55.77 GB):  84%|████████▍ | 49/58 [00:04<00:00, 35.59it/s]Capturing num tokens (num_tokens=48 avail_mem=55.77 GB):  84%|████████▍ | 49/58 [00:04<00:00, 35.59it/s]

    Capturing num tokens (num_tokens=32 avail_mem=55.76 GB):  84%|████████▍ | 49/58 [00:04<00:00, 35.59it/s]Capturing num tokens (num_tokens=28 avail_mem=55.76 GB):  84%|████████▍ | 49/58 [00:04<00:00, 35.59it/s]Capturing num tokens (num_tokens=24 avail_mem=55.76 GB):  84%|████████▍ | 49/58 [00:04<00:00, 35.59it/s]Capturing num tokens (num_tokens=20 avail_mem=55.75 GB):  84%|████████▍ | 49/58 [00:04<00:00, 35.59it/s]Capturing num tokens (num_tokens=20 avail_mem=55.75 GB):  93%|█████████▎| 54/58 [00:04<00:00, 37.29it/s]Capturing num tokens (num_tokens=16 avail_mem=55.75 GB):  93%|█████████▎| 54/58 [00:04<00:00, 37.29it/s]Capturing num tokens (num_tokens=12 avail_mem=55.75 GB):  93%|█████████▎| 54/58 [00:04<00:00, 37.29it/s]Capturing num tokens (num_tokens=8 avail_mem=55.74 GB):  93%|█████████▎| 54/58 [00:04<00:00, 37.29it/s] Capturing num tokens (num_tokens=4 avail_mem=55.74 GB):  93%|█████████▎| 54/58 [00:04<00:00, 37.29it/s]Capturing num tokens (num_tokens=4 avail_mem=55.74 GB): 100%|██████████| 58/58 [00:04<00:00, 37.90it/s]Capturing num tokens (num_tokens=4 avail_mem=55.74 GB): 100%|██████████| 58/58 [00:04<00:00, 13.38it/s]


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
    
    Generated text: Okay, so the user asked for the information and population of the capital of France in JSON format. Let me break this down.
    
    First, I know that the capital of France is Paris. That's pretty straightforward. Now, I need to find the most recent population data for Paris. I remember that populations can change, so I should look for the latest figures. 
    
    I think the population figure I have is around 2 million, but I'm not 100% sure. Maybe I should double-check that. Paris is a major city, so its population is significant. I also recall that it's divided into several arrondissements or districts, with the Île-de-France region having a larger population. 
    
    When structuring the JSON, I need to make sure it's properly formatted with the correct keys. The user probably wants a clear and concise response, so keeping it simple with "name" and "population" should suffice. I should format the population as an integer since it's a whole number.
    
    I should also consider the date, but since the user didn't specify, I'll include it to indicate that the data is current as of 2023. It's good practice to include a date in such responses unless the user requests otherwise.
    
    Wait, should I mention the Île-de-France region separately? The user asked for the capital, which is Paris, so maybe that's unnecessary unless it's part of the population data they're expecting. I think just providing Paris as the name and the population figure is enough.
    
    I also need to ensure that the JSON is valid. That means correct syntax, commas in the right places, and proper use of quotation marks. I'll format it with proper indentation for readability.
    
    Putting it all together, the JSON should have a key-value pair with "name" as "Paris" and "population" as 21587000, along with the date. I'll make sure to write it clearly so the user can easily understand and use the data.
    </think>
    
    ```json
    {
      "name": "Paris",
      "population": 21587000,
      "date": "2023"
    }
    ```



```python
llm.shutdown()
```
