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

    Multi-thread loading shards:  50% Completed | 1/2 [00:03<00:03,  3.72s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:06<00:00,  3.05s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:06<00:00,  3.15s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:37,  5.92s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:37,  5.92s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:23,  2.57s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:23,  2.57s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:21,  1.49s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:21,  1.49s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:52,  1.03it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:52,  1.03it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:26,  1.99it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:26,  1.99it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:19,  2.62it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:19,  2.62it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:15,  3.33it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:15,  3.33it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:11,  4.17it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:11,  4.17it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:07<00:11,  4.17it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:08,  5.83it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:08,  5.83it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:08,  5.83it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.39it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.39it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.39it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.00it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.00it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.00it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.10it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.10it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.10it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.10it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:08<00:02, 14.10it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:08<00:01, 19.33it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:08<00:01, 19.33it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:08<00:01, 19.33it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:08<00:01, 19.33it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:08<00:01, 19.33it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:08<00:01, 19.33it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:08<00:01, 26.39it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:08<00:01, 26.39it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:08<00:01, 26.39it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:08<00:01, 26.39it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:08<00:01, 26.39it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:08<00:01, 26.39it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:08<00:01, 26.39it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:08<00:00, 34.08it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:08<00:00, 34.08it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:08<00:00, 34.08it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:08<00:00, 34.08it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:08<00:00, 34.08it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:08<00:00, 34.08it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:08<00:00, 34.08it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:08<00:00, 34.08it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:08<00:00, 43.03it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 49.63it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 49.63it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 49.63it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 49.63it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 49.63it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:08<00:00, 49.63it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:08<00:00, 49.63it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:08<00:00, 49.63it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:08<00:00, 49.63it/s] 

    Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:08<00:00, 57.62it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:08<00:00, 57.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=61.68 GB):   2%|▏         | 1/58 [00:00<00:17,  3.24it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.65 GB):   2%|▏         | 1/58 [00:00<00:17,  3.24it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=61.65 GB):   3%|▎         | 2/58 [00:00<00:16,  3.41it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.64 GB):   3%|▎         | 2/58 [00:00<00:16,  3.41it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=61.64 GB):   5%|▌         | 3/58 [00:00<00:15,  3.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.64 GB):   5%|▌         | 3/58 [00:00<00:15,  3.60it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=61.64 GB):   7%|▋         | 4/58 [00:01<00:16,  3.33it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.64 GB):   7%|▋         | 4/58 [00:01<00:16,  3.33it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.64 GB):   9%|▊         | 5/58 [00:01<00:14,  3.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.64 GB):   9%|▊         | 5/58 [00:01<00:14,  3.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.64 GB):  10%|█         | 6/58 [00:01<00:12,  4.18it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.64 GB):  10%|█         | 6/58 [00:01<00:12,  4.18it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=61.64 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.64 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.64 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.64 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.15it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=61.64 GB):  16%|█▌        | 9/58 [00:02<00:09,  4.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.62 GB):  16%|█▌        | 9/58 [00:02<00:09,  4.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.62 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.13 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.28it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=61.13 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.95it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.97 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.95it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.97 GB):  21%|██        | 12/58 [00:02<00:06,  6.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=60.96 GB):  21%|██        | 12/58 [00:02<00:06,  6.67it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=60.96 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.96 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.96 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.96 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.96 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.90it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=60.96 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.96 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.96 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.95 GB):  29%|██▉       | 17/58 [00:03<00:03, 10.32it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=60.95 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.95 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.94 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.93it/s]Capturing num tokens (num_tokens=960 avail_mem=60.93 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.93it/s] Capturing num tokens (num_tokens=960 avail_mem=60.93 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.06it/s]Capturing num tokens (num_tokens=896 avail_mem=60.93 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.06it/s]Capturing num tokens (num_tokens=832 avail_mem=60.93 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.06it/s]

    Capturing num tokens (num_tokens=768 avail_mem=60.92 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.06it/s]Capturing num tokens (num_tokens=768 avail_mem=60.92 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.04it/s]Capturing num tokens (num_tokens=704 avail_mem=60.92 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.04it/s]Capturing num tokens (num_tokens=640 avail_mem=60.92 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.04it/s]Capturing num tokens (num_tokens=576 avail_mem=60.91 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.04it/s]Capturing num tokens (num_tokens=576 avail_mem=60.91 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.48it/s]Capturing num tokens (num_tokens=512 avail_mem=60.91 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.48it/s]Capturing num tokens (num_tokens=480 avail_mem=60.91 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.48it/s]

    Capturing num tokens (num_tokens=448 avail_mem=60.90 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.48it/s]Capturing num tokens (num_tokens=448 avail_mem=60.90 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.84it/s]Capturing num tokens (num_tokens=416 avail_mem=60.90 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.84it/s]Capturing num tokens (num_tokens=384 avail_mem=60.90 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.84it/s]Capturing num tokens (num_tokens=352 avail_mem=60.89 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.84it/s]Capturing num tokens (num_tokens=352 avail_mem=60.89 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.76it/s]Capturing num tokens (num_tokens=320 avail_mem=60.89 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.76it/s]Capturing num tokens (num_tokens=288 avail_mem=60.89 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.76it/s]

    Capturing num tokens (num_tokens=256 avail_mem=60.89 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.76it/s]Capturing num tokens (num_tokens=240 avail_mem=60.88 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.76it/s]Capturing num tokens (num_tokens=240 avail_mem=60.88 GB):  66%|██████▌   | 38/58 [00:03<00:00, 26.56it/s]Capturing num tokens (num_tokens=224 avail_mem=60.88 GB):  66%|██████▌   | 38/58 [00:03<00:00, 26.56it/s]Capturing num tokens (num_tokens=208 avail_mem=60.87 GB):  66%|██████▌   | 38/58 [00:03<00:00, 26.56it/s]Capturing num tokens (num_tokens=192 avail_mem=60.87 GB):  66%|██████▌   | 38/58 [00:03<00:00, 26.56it/s]Capturing num tokens (num_tokens=176 avail_mem=60.87 GB):  66%|██████▌   | 38/58 [00:03<00:00, 26.56it/s]Capturing num tokens (num_tokens=176 avail_mem=60.87 GB):  72%|███████▏  | 42/58 [00:03<00:00, 29.37it/s]Capturing num tokens (num_tokens=160 avail_mem=60.87 GB):  72%|███████▏  | 42/58 [00:03<00:00, 29.37it/s]Capturing num tokens (num_tokens=144 avail_mem=60.86 GB):  72%|███████▏  | 42/58 [00:03<00:00, 29.37it/s]

    Capturing num tokens (num_tokens=128 avail_mem=60.86 GB):  72%|███████▏  | 42/58 [00:03<00:00, 29.37it/s]Capturing num tokens (num_tokens=112 avail_mem=60.86 GB):  72%|███████▏  | 42/58 [00:03<00:00, 29.37it/s]Capturing num tokens (num_tokens=112 avail_mem=60.86 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.13it/s]Capturing num tokens (num_tokens=96 avail_mem=60.85 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.13it/s] Capturing num tokens (num_tokens=80 avail_mem=60.85 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.13it/s]Capturing num tokens (num_tokens=64 avail_mem=60.85 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.13it/s]Capturing num tokens (num_tokens=48 avail_mem=60.84 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.13it/s]Capturing num tokens (num_tokens=48 avail_mem=60.84 GB):  86%|████████▌ | 50/58 [00:04<00:00, 32.31it/s]Capturing num tokens (num_tokens=32 avail_mem=60.84 GB):  86%|████████▌ | 50/58 [00:04<00:00, 32.31it/s]Capturing num tokens (num_tokens=28 avail_mem=60.84 GB):  86%|████████▌ | 50/58 [00:04<00:00, 32.31it/s]

    Capturing num tokens (num_tokens=24 avail_mem=60.83 GB):  86%|████████▌ | 50/58 [00:04<00:00, 32.31it/s]Capturing num tokens (num_tokens=20 avail_mem=60.83 GB):  86%|████████▌ | 50/58 [00:04<00:00, 32.31it/s]Capturing num tokens (num_tokens=20 avail_mem=60.83 GB):  93%|█████████▎| 54/58 [00:04<00:00, 31.64it/s]Capturing num tokens (num_tokens=16 avail_mem=60.83 GB):  93%|█████████▎| 54/58 [00:04<00:00, 31.64it/s]Capturing num tokens (num_tokens=12 avail_mem=60.82 GB):  93%|█████████▎| 54/58 [00:04<00:00, 31.64it/s]Capturing num tokens (num_tokens=8 avail_mem=60.82 GB):  93%|█████████▎| 54/58 [00:04<00:00, 31.64it/s] Capturing num tokens (num_tokens=4 avail_mem=60.82 GB):  93%|█████████▎| 54/58 [00:04<00:00, 31.64it/s]Capturing num tokens (num_tokens=4 avail_mem=60.82 GB): 100%|██████████| 58/58 [00:04<00:00, 32.35it/s]Capturing num tokens (num_tokens=4 avail_mem=60.82 GB): 100%|██████████| 58/58 [00:04<00:00, 13.24it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants the current date and time along with the weather. Let me figure out how to approach this step by step.<br><br>First, I need to determine which functions to use. The user mentioned two functions: 'get_current_weather' and 'get_current_date'. Both are available, so I can use both.<br><br>Starting with the date and time, the 'get_current_date' function requires a timezone parameter. Since the user is in New York, I should provide the timezone as 'America/New_York'. That should give me the exact current date and time for that location.<br><br>Next, for the weather, I'll use 'get_current_weather'. The user is in New York, so I'll set the city as 'New York' and the state as 'NY' since that's the two-letter abbreviation. The unit isn't specified, so I'll default to Celsius to provide a temperature in a familiar unit for most users, but I should note that Fahrenheit is also an option in case they prefer it.<br><br>I need to make sure each function call is properly formatted. For the date, it's a single function call, so I'll structure it accordingly. For the weather, I'll include both city and state parameters. I'll also add a note that the user can switch to Fahrenheit if needed.<br><br>Putting it all together, I'll first call 'get_current_date' with the timezone parameter. Then, I'll call 'get_current_weather' with city and state, and include a note about the temperature unit. This way, the user gets both the date and weather information they requested.<br><br><br>content: <br><br><function=get_current_date>{"timezone": "America/New_York"}</function><br/>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "celsius"}</function>  <br>The current temperature in New York is 22°C. You can switch to Fahrenheit by changing the unit parameter to "fahrenheit".</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '91fc882331c341d2a4abe88e572124b0', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 6.263896256219596, 'response_sent_to_client_ts': 1779937674.134998}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '85a3b4e930a240069278e5deebe43969', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 7.346333076246083, 'response_sent_to_client_ts': 1779937681.4960804}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '3216026d258f4b13bf0cbe78de51a8fa', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2781123211607337, 'response_sent_to_client_ts': 1779937681.8203533}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '61afb0df9c6d4303ab0e264c80a3c3aa', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2780214622616768, 'response_sent_to_client_ts': 1779937681.820366}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'bc7a42f61c3944578f70ab883df4d06f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.27797211799770594, 'response_sent_to_client_ts': 1779937681.8203704}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'e1f851385e1742109b0f09bd726f95e4', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 21.190895502921194, 'response_sent_to_client_ts': 1779937703.018492}}


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


<strong style='color: #00008B;'>{'text': 'Alright, the user is asking for the information and population of the capital of France in JSON format. \n\nFirst, I need to recall that the capital of France is Paris.\n\nNext, I\'ll gather the required details: population and maybe some key facts like the administrative region within Paris.\n\nI should find the approximate current population, which is around 2.2 million people as of 2023.\n\nIt would also be helpful to include information about the city\'s elevation and climate to give a more comprehensive answer.\n\nPutting this all together, I\'ll structure it in JSON format properly, making sure each piece of data is accurately represented.\n\nI should ensure the JSON syntax is correct, with proper commas and quotation marks to avoid errors.\n\nFinally, I\'ll present the JSON in a clear and concise manner so the user can easily understand and use the information.\n</think>\n\nHere is the information about the capital of France, Paris, in JSON format:\n\n```json\n{\n  "capital": "Paris",\n  "population": 2197500,\n  "administrativeRegion": "Greater Paris",\n  "elevation": "53 m above sea level",\n  "climate": "Temperate, Rhin繁",\n  "founded": "1348",\n  "nation": "France"\n}\n```\n\nThis JSON object contains details about the city, including its population, administrative region, elevation, climate, and more.', 'output_ids': [71486, 11, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 4710, 5338, 11, 358, 1184, 311, 19091, 429, 279, 6722, 315, 9625, 374, 12095, 382, 5847, 11, 358, 3278, 9567, 279, 2567, 3565, 25, 7042, 323, 7196, 1045, 1376, 13064, 1075, 279, 22707, 5537, 2878, 12095, 382, 40, 1265, 1477, 279, 44868, 1482, 7042, 11, 892, 374, 2163, 220, 17, 13, 17, 3526, 1251, 438, 315, 220, 17, 15, 17, 18, 382, 2132, 1035, 1083, 387, 10950, 311, 2924, 1995, 911, 279, 3283, 594, 26163, 323, 9977, 311, 2968, 264, 803, 15817, 4226, 382, 97904, 419, 678, 3786, 11, 358, 3278, 5944, 432, 304, 4718, 3561, 10277, 11, 3259, 2704, 1817, 6573, 315, 821, 374, 29257, 15251, 382, 40, 1265, 5978, 279, 4718, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 304, 264, 2797, 323, 63594, 11566, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 624, 151649, 271, 8420, 374, 279, 1995, 911, 279, 6722, 315, 9625, 11, 12095, 11, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 24, 22, 20, 15, 15, 345, 220, 330, 68849, 1388, 14091, 788, 330, 41366, 12095, 756, 220, 330, 68, 43757, 788, 330, 20, 18, 296, 3403, 9396, 2188, 756, 220, 330, 93774, 788, 330, 21988, 79113, 11, 17968, 258, 100112, 756, 220, 330, 69, 13082, 788, 330, 16, 18, 19, 23, 756, 220, 330, 98795, 788, 330, 49000, 698, 532, 13874, 19324, 1986, 4718, 1633, 5610, 3565, 911, 279, 3283, 11, 2670, 1181, 7042, 11, 22707, 5537, 11, 26163, 11, 9977, 11, 323, 803, 13, 151643], 'meta_info': {'id': '9b552421781e4ee5a38793d18d791e95', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 172, 'completion_tokens': 293, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 2.973167919088155, 'response_sent_to_client_ts': 1779937705.9997447}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.32s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.18s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.20s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:18,  5.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:18,  5.58s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:17,  2.46s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:17,  2.46s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:20,  1.46s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:20,  1.46s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:53,  1.02it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:53,  1.02it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:27,  1.87it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:27,  1.87it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:21,  2.38it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:21,  2.38it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:16,  2.95it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:16,  2.95it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:13,  3.63it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:13,  3.63it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:11,  4.00it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:11,  4.00it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:10,  4.46it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:10,  4.46it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:09,  4.83it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:09,  4.83it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:08,  5.12it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:08,  5.12it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:07,  5.53it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:07,  5.53it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:06,  6.31it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:06,  6.31it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:05,  7.04it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:05,  7.04it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:08<00:05,  7.04it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:08<00:04,  8.93it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:08<00:04,  8.93it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:08<00:04,  8.93it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:08<00:03, 11.44it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:08<00:03, 11.44it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:08<00:03, 11.44it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:08<00:03, 11.44it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:08<00:03, 11.44it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:08<00:01, 17.55it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:08<00:01, 17.55it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:08<00:01, 17.55it/s]

    Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:08<00:01, 17.55it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:08<00:01, 17.55it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:08<00:01, 17.55it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:08<00:01, 24.66it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:08<00:01, 24.66it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:08<00:01, 24.66it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:08<00:01, 24.66it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:08<00:01, 24.66it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:08<00:01, 24.66it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:08<00:01, 24.66it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:08<00:00, 32.84it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:08<00:00, 32.84it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:08<00:00, 32.84it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:08<00:00, 32.84it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:08<00:00, 32.84it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:08<00:00, 32.84it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:09<00:00, 32.84it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:09<00:00, 32.84it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:09<00:00, 42.17it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:09<00:00, 42.17it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:09<00:00, 42.17it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:09<00:00, 42.17it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:09<00:00, 42.17it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:09<00:00, 42.17it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:09<00:00, 42.17it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:09<00:00, 42.17it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:09<00:00, 49.46it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:09<00:00, 49.46it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:09<00:00, 49.46it/s]

    Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:09<00:00, 49.46it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:09<00:00, 49.46it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:09<00:00, 49.46it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:09<00:00, 49.46it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:09<00:00, 49.46it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:09<00:00, 49.46it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:09<00:00, 49.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00, 60.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  6.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.99 GB):   2%|▏         | 1/58 [00:00<00:16,  3.45it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.95 GB):   2%|▏         | 1/58 [00:00<00:16,  3.45it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.95 GB):   3%|▎         | 2/58 [00:00<00:15,  3.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.95 GB):   3%|▎         | 2/58 [00:00<00:15,  3.61it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.95 GB):   5%|▌         | 3/58 [00:00<00:14,  3.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.95 GB):   5%|▌         | 3/58 [00:00<00:14,  3.78it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.95 GB):   7%|▋         | 4/58 [00:01<00:13,  4.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.95 GB):   7%|▋         | 4/58 [00:01<00:13,  4.08it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.95 GB):   9%|▊         | 5/58 [00:01<00:12,  4.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.95 GB):   9%|▊         | 5/58 [00:01<00:12,  4.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.95 GB):  10%|█         | 6/58 [00:01<00:10,  4.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.95 GB):  10%|█         | 6/58 [00:01<00:10,  4.74it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.95 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.95 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.95 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.95 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.61it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.95 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.95 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.95 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.95 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.51it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.95 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.94 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.94 GB):  21%|██        | 12/58 [00:02<00:06,  7.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.94 GB):  21%|██        | 12/58 [00:02<00:06,  7.57it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.94 GB):  21%|██        | 12/58 [00:02<00:06,  7.57it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.94 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.61it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.91 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.61it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=43.91 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.43it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.90 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.90 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.90 GB):  29%|██▉       | 17/58 [00:02<00:04, 10.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.90 GB):  29%|██▉       | 17/58 [00:02<00:04, 10.14it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=43.89 GB):  29%|██▉       | 17/58 [00:02<00:04, 10.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.89 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.89 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.88 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.93it/s]Capturing num tokens (num_tokens=960 avail_mem=43.88 GB):  33%|███▎      | 19/58 [00:02<00:03, 11.93it/s] Capturing num tokens (num_tokens=960 avail_mem=43.88 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.83it/s]Capturing num tokens (num_tokens=896 avail_mem=43.87 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.83it/s]

    Capturing num tokens (num_tokens=832 avail_mem=43.87 GB):  38%|███▊      | 22/58 [00:02<00:02, 14.83it/s]Capturing num tokens (num_tokens=768 avail_mem=43.86 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.83it/s]Capturing num tokens (num_tokens=768 avail_mem=43.86 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.02it/s]Capturing num tokens (num_tokens=704 avail_mem=43.86 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.02it/s]Capturing num tokens (num_tokens=640 avail_mem=43.86 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.02it/s]Capturing num tokens (num_tokens=576 avail_mem=43.85 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.02it/s]

    Capturing num tokens (num_tokens=576 avail_mem=43.85 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.95it/s]Capturing num tokens (num_tokens=512 avail_mem=43.85 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.95it/s]Capturing num tokens (num_tokens=480 avail_mem=43.85 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.95it/s]Capturing num tokens (num_tokens=448 avail_mem=43.84 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.95it/s]Capturing num tokens (num_tokens=448 avail_mem=43.84 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.18it/s]Capturing num tokens (num_tokens=416 avail_mem=43.84 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.18it/s]Capturing num tokens (num_tokens=384 avail_mem=43.84 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.18it/s]Capturing num tokens (num_tokens=352 avail_mem=43.83 GB):  53%|█████▎    | 31/58 [00:03<00:01, 20.18it/s]

    Capturing num tokens (num_tokens=352 avail_mem=43.83 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.48it/s]Capturing num tokens (num_tokens=320 avail_mem=43.83 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.48it/s]Capturing num tokens (num_tokens=288 avail_mem=43.83 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.48it/s]Capturing num tokens (num_tokens=256 avail_mem=43.83 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.48it/s]Capturing num tokens (num_tokens=240 avail_mem=43.82 GB):  59%|█████▊    | 34/58 [00:03<00:01, 22.48it/s]Capturing num tokens (num_tokens=240 avail_mem=43.82 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.85it/s]Capturing num tokens (num_tokens=224 avail_mem=43.82 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.85it/s]Capturing num tokens (num_tokens=208 avail_mem=43.82 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.85it/s]Capturing num tokens (num_tokens=192 avail_mem=43.81 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.85it/s]

    Capturing num tokens (num_tokens=176 avail_mem=43.81 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.85it/s]Capturing num tokens (num_tokens=176 avail_mem=43.81 GB):  72%|███████▏  | 42/58 [00:03<00:00, 28.59it/s]Capturing num tokens (num_tokens=160 avail_mem=43.81 GB):  72%|███████▏  | 42/58 [00:03<00:00, 28.59it/s]Capturing num tokens (num_tokens=144 avail_mem=43.80 GB):  72%|███████▏  | 42/58 [00:03<00:00, 28.59it/s]Capturing num tokens (num_tokens=128 avail_mem=43.80 GB):  72%|███████▏  | 42/58 [00:03<00:00, 28.59it/s]Capturing num tokens (num_tokens=112 avail_mem=43.80 GB):  72%|███████▏  | 42/58 [00:03<00:00, 28.59it/s]Capturing num tokens (num_tokens=112 avail_mem=43.80 GB):  79%|███████▉  | 46/58 [00:03<00:00, 30.36it/s]Capturing num tokens (num_tokens=96 avail_mem=43.80 GB):  79%|███████▉  | 46/58 [00:03<00:00, 30.36it/s] Capturing num tokens (num_tokens=80 avail_mem=43.79 GB):  79%|███████▉  | 46/58 [00:03<00:00, 30.36it/s]Capturing num tokens (num_tokens=64 avail_mem=43.79 GB):  79%|███████▉  | 46/58 [00:03<00:00, 30.36it/s]

    Capturing num tokens (num_tokens=48 avail_mem=43.78 GB):  79%|███████▉  | 46/58 [00:03<00:00, 30.36it/s]Capturing num tokens (num_tokens=48 avail_mem=43.78 GB):  86%|████████▌ | 50/58 [00:03<00:00, 32.06it/s]Capturing num tokens (num_tokens=32 avail_mem=43.78 GB):  86%|████████▌ | 50/58 [00:03<00:00, 32.06it/s]Capturing num tokens (num_tokens=28 avail_mem=43.78 GB):  86%|████████▌ | 50/58 [00:03<00:00, 32.06it/s]Capturing num tokens (num_tokens=24 avail_mem=43.78 GB):  86%|████████▌ | 50/58 [00:03<00:00, 32.06it/s]Capturing num tokens (num_tokens=20 avail_mem=43.77 GB):  86%|████████▌ | 50/58 [00:03<00:00, 32.06it/s]Capturing num tokens (num_tokens=20 avail_mem=43.77 GB):  93%|█████████▎| 54/58 [00:03<00:00, 33.17it/s]Capturing num tokens (num_tokens=16 avail_mem=43.77 GB):  93%|█████████▎| 54/58 [00:03<00:00, 33.17it/s]Capturing num tokens (num_tokens=12 avail_mem=43.76 GB):  93%|█████████▎| 54/58 [00:04<00:00, 33.17it/s]Capturing num tokens (num_tokens=8 avail_mem=43.76 GB):  93%|█████████▎| 54/58 [00:04<00:00, 33.17it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=43.76 GB):  93%|█████████▎| 54/58 [00:04<00:00, 33.17it/s]Capturing num tokens (num_tokens=4 avail_mem=43.76 GB): 100%|██████████| 58/58 [00:04<00:00, 33.66it/s]Capturing num tokens (num_tokens=4 avail_mem=43.76 GB): 100%|██████████| 58/58 [00:04<00:00, 14.18it/s]


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
    
    Generated text: Okay, so the user asked me to provide the information and population of the capital of France in JSON format. The capital is Paris, obviously. I need to figure out the correct population number. I remember that Paris is one of the most populous cities globally, but I'm not sure about the exact number. I think it's around 2 million, but I'm not 100% certain. Maybe I should check some recent data to confirm. 
    
    Wait, I think the population has been increasing each year. I recall something about it being over 2 million in recent years. Let me try to recall the exact number. Oh, yes, it's approximately 2,100,000 people. But I should double-check this to make sure it's accurate. Maybe I can think about how it's been reported in news articles or official statistics. I'm pretty confident, but it's better to have the correct figure.
    
    So, putting it all together, I'll structure the JSON with the "city" as "Paris" and the "population" as "2100000". That should fulfill the user's request accurately.
    </think>
    
    ```json
    {
      "city": "Paris",
      "population": 2100000
    }
    ```



```python
llm.shutdown()
```
