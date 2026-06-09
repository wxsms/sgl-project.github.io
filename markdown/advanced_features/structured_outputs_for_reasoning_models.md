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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.38s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.33s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:54,  5.17s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:54,  5.17s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:14,  2.40s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:14,  2.40s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:22,  1.50s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:22,  1.50s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:57,  1.06s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:57,  1.06s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:42,  1.25it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:42,  1.25it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:33,  1.56it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:33,  1.56it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:26,  1.90it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:26,  1.90it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:22,  2.25it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:22,  2.25it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:18,  2.68it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:18,  2.68it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:15,  3.12it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:15,  3.12it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:13,  3.54it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:13,  3.54it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:11,  4.08it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:11,  4.08it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:09,  4.56it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:09,  4.56it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:08,  5.16it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:08,  5.16it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:07,  5.81it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:07,  5.81it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:06,  6.52it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:06,  6.52it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:09<00:06,  6.52it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:05,  7.94it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:05,  7.94it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:09<00:05,  7.94it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:04,  9.48it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:04,  9.48it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:04,  9.48it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:09<00:04,  9.48it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:09<00:02, 13.05it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:09<00:02, 13.05it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:09<00:02, 13.05it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:09<00:02, 13.05it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:09<00:01, 16.74it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:09<00:01, 16.74it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:09<00:01, 16.74it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:09<00:01, 16.74it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:09<00:01, 16.74it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:09<00:01, 21.25it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:09<00:01, 21.25it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:09<00:01, 21.25it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:09<00:01, 21.25it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:09<00:01, 21.25it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:09<00:00, 25.27it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:09<00:00, 25.27it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:09<00:00, 25.27it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:09<00:00, 25.27it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:09<00:00, 25.27it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:09<00:00, 28.88it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:09<00:00, 28.88it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:09<00:00, 28.88it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:09<00:00, 28.88it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:09<00:00, 28.88it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:09<00:00, 28.88it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:09<00:00, 33.08it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:09<00:00, 33.08it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:10<00:00, 33.08it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:10<00:00, 33.08it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:10<00:00, 33.08it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:10<00:00, 33.08it/s]

    Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:10<00:00, 33.08it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:10<00:00, 39.59it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:10<00:00, 39.59it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:10<00:00, 39.59it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:10<00:00, 39.59it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:10<00:00, 39.59it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:10<00:00, 39.59it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:10<00:00, 39.59it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:10<00:00, 39.59it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:10<00:00, 39.59it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:10<00:00, 39.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00, 52.73it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=34.48 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=34.48 GB):   2%|▏         | 1/58 [00:00<00:35,  1.60it/s]Capturing num tokens (num_tokens=7680 avail_mem=34.45 GB):   2%|▏         | 1/58 [00:00<00:35,  1.60it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=34.45 GB):   3%|▎         | 2/58 [00:01<00:32,  1.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=34.44 GB):   3%|▎         | 2/58 [00:01<00:32,  1.73it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=34.44 GB):   5%|▌         | 3/58 [00:01<00:29,  1.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=33.66 GB):   5%|▌         | 3/58 [00:01<00:29,  1.87it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=33.66 GB):   7%|▋         | 4/58 [00:02<00:26,  2.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=33.72 GB):   7%|▋         | 4/58 [00:02<00:26,  2.02it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=33.72 GB):   9%|▊         | 5/58 [00:02<00:24,  2.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=33.78 GB):   9%|▊         | 5/58 [00:02<00:24,  2.15it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=33.78 GB):  10%|█         | 6/58 [00:02<00:22,  2.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=33.85 GB):  10%|█         | 6/58 [00:02<00:22,  2.33it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=33.85 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=33.92 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.52it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=33.92 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=33.95 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.77it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=33.95 GB):  16%|█▌        | 9/58 [00:03<00:16,  3.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=33.98 GB):  16%|█▌        | 9/58 [00:03<00:16,  3.04it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=33.98 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=34.01 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.30it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=34.01 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=34.06 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=34.06 GB):  21%|██        | 12/58 [00:04<00:11,  4.02it/s]Capturing num tokens (num_tokens=3072 avail_mem=34.43 GB):  21%|██        | 12/58 [00:04<00:11,  4.02it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=34.43 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=34.43 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=34.43 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=34.42 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.84it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=34.42 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=34.42 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=34.42 GB):  28%|██▊       | 16/58 [00:04<00:07,  5.92it/s]Capturing num tokens (num_tokens=2048 avail_mem=34.41 GB):  28%|██▊       | 16/58 [00:04<00:07,  5.92it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=34.41 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.56it/s]Capturing num tokens (num_tokens=1792 avail_mem=34.41 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.56it/s]Capturing num tokens (num_tokens=1792 avail_mem=34.41 GB):  31%|███       | 18/58 [00:05<00:05,  7.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=34.26 GB):  31%|███       | 18/58 [00:05<00:05,  7.04it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=34.39 GB):  31%|███       | 18/58 [00:05<00:05,  7.04it/s]Capturing num tokens (num_tokens=1280 avail_mem=34.39 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=34.37 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.61it/s]Capturing num tokens (num_tokens=960 avail_mem=34.36 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.61it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=34.36 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.36it/s]Capturing num tokens (num_tokens=896 avail_mem=34.35 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.36it/s]Capturing num tokens (num_tokens=832 avail_mem=34.35 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.36it/s]Capturing num tokens (num_tokens=832 avail_mem=34.35 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.84it/s]Capturing num tokens (num_tokens=768 avail_mem=34.34 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.84it/s]Capturing num tokens (num_tokens=704 avail_mem=34.33 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.84it/s]

    Capturing num tokens (num_tokens=704 avail_mem=34.33 GB):  45%|████▍     | 26/58 [00:05<00:02, 13.37it/s]Capturing num tokens (num_tokens=640 avail_mem=34.33 GB):  45%|████▍     | 26/58 [00:05<00:02, 13.37it/s]Capturing num tokens (num_tokens=576 avail_mem=34.32 GB):  45%|████▍     | 26/58 [00:05<00:02, 13.37it/s]Capturing num tokens (num_tokens=576 avail_mem=34.32 GB):  48%|████▊     | 28/58 [00:05<00:02, 14.70it/s]Capturing num tokens (num_tokens=512 avail_mem=34.31 GB):  48%|████▊     | 28/58 [00:05<00:02, 14.70it/s]Capturing num tokens (num_tokens=480 avail_mem=34.30 GB):  48%|████▊     | 28/58 [00:05<00:02, 14.70it/s]Capturing num tokens (num_tokens=448 avail_mem=34.29 GB):  48%|████▊     | 28/58 [00:05<00:02, 14.70it/s]

    Capturing num tokens (num_tokens=448 avail_mem=34.29 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.89it/s]Capturing num tokens (num_tokens=416 avail_mem=34.29 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.89it/s]Capturing num tokens (num_tokens=384 avail_mem=34.29 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.89it/s]Capturing num tokens (num_tokens=352 avail_mem=34.28 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.89it/s]Capturing num tokens (num_tokens=352 avail_mem=34.28 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.82it/s]Capturing num tokens (num_tokens=320 avail_mem=34.27 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.82it/s]Capturing num tokens (num_tokens=288 avail_mem=34.27 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.82it/s]

    Capturing num tokens (num_tokens=256 avail_mem=34.26 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.82it/s]Capturing num tokens (num_tokens=256 avail_mem=34.26 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.80it/s]Capturing num tokens (num_tokens=240 avail_mem=34.16 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.80it/s]Capturing num tokens (num_tokens=224 avail_mem=33.98 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.80it/s]Capturing num tokens (num_tokens=208 avail_mem=33.81 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.80it/s]Capturing num tokens (num_tokens=208 avail_mem=33.81 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.96it/s]Capturing num tokens (num_tokens=192 avail_mem=33.63 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.96it/s]Capturing num tokens (num_tokens=176 avail_mem=33.49 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.96it/s]

    Capturing num tokens (num_tokens=160 avail_mem=33.33 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.96it/s]Capturing num tokens (num_tokens=160 avail_mem=33.33 GB):  74%|███████▍  | 43/58 [00:06<00:00, 22.96it/s]Capturing num tokens (num_tokens=144 avail_mem=33.14 GB):  74%|███████▍  | 43/58 [00:06<00:00, 22.96it/s]Capturing num tokens (num_tokens=128 avail_mem=32.92 GB):  74%|███████▍  | 43/58 [00:06<00:00, 22.96it/s]Capturing num tokens (num_tokens=112 avail_mem=32.88 GB):  74%|███████▍  | 43/58 [00:06<00:00, 22.96it/s]Capturing num tokens (num_tokens=112 avail_mem=32.88 GB):  79%|███████▉  | 46/58 [00:06<00:00, 23.90it/s]Capturing num tokens (num_tokens=96 avail_mem=32.87 GB):  79%|███████▉  | 46/58 [00:06<00:00, 23.90it/s] Capturing num tokens (num_tokens=80 avail_mem=32.86 GB):  79%|███████▉  | 46/58 [00:06<00:00, 23.90it/s]

    Capturing num tokens (num_tokens=64 avail_mem=32.86 GB):  79%|███████▉  | 46/58 [00:06<00:00, 23.90it/s]Capturing num tokens (num_tokens=64 avail_mem=32.86 GB):  84%|████████▍ | 49/58 [00:06<00:00, 24.87it/s]Capturing num tokens (num_tokens=48 avail_mem=32.83 GB):  84%|████████▍ | 49/58 [00:06<00:00, 24.87it/s]Capturing num tokens (num_tokens=32 avail_mem=32.82 GB):  84%|████████▍ | 49/58 [00:06<00:00, 24.87it/s]Capturing num tokens (num_tokens=28 avail_mem=32.72 GB):  84%|████████▍ | 49/58 [00:06<00:00, 24.87it/s]Capturing num tokens (num_tokens=28 avail_mem=32.72 GB):  90%|████████▉ | 52/58 [00:06<00:00, 26.00it/s]Capturing num tokens (num_tokens=24 avail_mem=32.72 GB):  90%|████████▉ | 52/58 [00:06<00:00, 26.00it/s]Capturing num tokens (num_tokens=20 avail_mem=32.67 GB):  90%|████████▉ | 52/58 [00:06<00:00, 26.00it/s]

    Capturing num tokens (num_tokens=16 avail_mem=32.66 GB):  90%|████████▉ | 52/58 [00:06<00:00, 26.00it/s]Capturing num tokens (num_tokens=16 avail_mem=32.66 GB):  95%|█████████▍| 55/58 [00:06<00:00, 26.77it/s]Capturing num tokens (num_tokens=12 avail_mem=32.51 GB):  95%|█████████▍| 55/58 [00:06<00:00, 26.77it/s]Capturing num tokens (num_tokens=8 avail_mem=32.12 GB):  95%|█████████▍| 55/58 [00:06<00:00, 26.77it/s] Capturing num tokens (num_tokens=4 avail_mem=31.55 GB):  95%|█████████▍| 55/58 [00:06<00:00, 26.77it/s]Capturing num tokens (num_tokens=4 avail_mem=31.55 GB): 100%|██████████| 58/58 [00:07<00:00,  8.27it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out how to get the current date and time in New York and the weather there using the functions provided. Let me break this down step by step.<br><br>First, I remember that the user mentioned they're in New York, so the timezone is probably New York's. I think the timezone for New York is Eastern Time, which is 'America/New_York'. So I should use the 'get_current_date' function with the timezone parameter set to 'America/New_York'.<br><br>Next, for the weather, since the user is in New York, I need to specify the city and state. The city is obviously New York, and the state abbreviation is NY. So I'll use the 'get_current_weather' function with city 'New York' and state 'NY'. I should also specify the unit as 'fahrenheit' since the user didn't mention it, but it's good to include it for clarity.<br><br>Now, I have to structure the function calls correctly. Each function needs to be called separately, so I'll handle the date and time first. I'll write the function call for 'get_current_date' with the timezone parameter. Then, I'll make another function call for 'get_current_weather' with the city and state, along with the unit.<br><br>Wait, the user asked for both the date and time and the weather. So I should include both function calls in the response. I'll make sure each function is properly formatted with the correct parameters inside curly braces and the entire call enclosed in the start and end tags as specified.<br><br>I should also add sources to my answer. For the date and time, I'll cite the function's documentation, and similarly for the weather function. That way, the user knows where the information comes from.<br><br>Putting it all together, I'll write two separate function calls: one for the date and time and another for the weather. Each will have their parameters correctly specified, and I'll include the sources at the end of each section.<br><br>I think that covers everything the user asked for. They wanted the current date and time in New York and the weather there, and I've structured the responses using the provided functions with the necessary parameters.<br><br><br>content: <br><br><function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>  <br><br>Source for date and time: https://docs.python.org/  <br>Source for weather: https://docs.python.org/</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'a7776c8cbb8a4416ac98272e41699115', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.112922478467226, 'response_sent_to_client_ts': 1781047991.452262}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': 'dc7307ab760a46b3b7db6d81b70291b2', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 26.182740211486816, 'response_sent_to_client_ts': 1781048017.644195}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '21b5842cd0db45c0a8346c23dd45f3bf', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.6482996717095375, 'response_sent_to_client_ts': 1781048018.360985}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '3126cfaefa70429c82fbb3d73054ad41', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.6482311114668846, 'response_sent_to_client_ts': 1781048018.3609996}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'ff3c66fe0ab344ecb4819675b058f21d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.6481977291405201, 'response_sent_to_client_ts': 1781048018.3610036}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'f4b7ede2c18a46d9b4cc005d53fdb9b7', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 20.9129774607718, 'response_sent_to_client_ts': 1781048039.2817235}}


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


<strong style='color: #00008B;'>{'text': 'Alright, so I need to figure out how to respond to this request. The user is asking for the information and population of the capital of France in JSON format. Let me start by breaking this down.\n\nFirst, I know that the capital of France is Paris. That’s pretty straightforward, but just to be sure, I should double-check. Paris is the administrative capital, and sometimes capitals can change, but as of now, it\'s definitely Paris.\n\nNext, the user wants the population. Population numbers can change over time, so I need to make sure I have the most recent data. I remember that the population figures for major cities vary, and sometimes they\'re approximate. I think the population of Paris is over 3 million, but I\'m not exactly sure of the exact number. Let me think... somehow, I recall it being around 3.5 million. I should confirm that, but since I can’t look it up right now, I\'ll go with that number for now.\n\nNow, structuring the response in JSON format is another aspect. JSON requires key-value pairs, and it should be properly formatted, with commas in the right places to avoid errors. I should make sure the syntax is correct, using quotation marks and brackets appropriately.\n\nThe user might be a student working on a project or perhaps someone setting up a website where they need accurate data quickly. They’re asking for a specific format, so they\'re probably aware of JSON and want the information structured that way for easy parsing or integration into another application.\n\nI should also consider if they need more detailed data, like the most recent year the population is from or if they need additional information beyond just the population. But since their request is specific, I\'ll stick to the capital, its name, and the population.\n\nPutting it all together, the JSON should have a key "capital" with the value "Paris" and a key "population" with the value of 3.5 million. I\'ll format it correctly, ensuring there are no trailing commas to maintain valid JSON.\n\nI think that covers all the requirements. Now, time to write the response accordingly.\n</think>\n\n```json\n{\n  "capital": "Paris",\n  "population": 3500000\n}\n```', 'output_ids': [71486, 11, 773, 358, 1184, 311, 7071, 700, 1246, 311, 5889, 311, 419, 1681, 13, 576, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1191, 553, 14719, 419, 1495, 382, 5338, 11, 358, 1414, 429, 279, 6722, 315, 9625, 374, 12095, 13, 2938, 748, 5020, 30339, 11, 714, 1101, 311, 387, 2704, 11, 358, 1265, 1990, 15934, 13, 12095, 374, 279, 22707, 6722, 11, 323, 7025, 92999, 646, 2297, 11, 714, 438, 315, 1431, 11, 432, 594, 8491, 12095, 382, 5847, 11, 279, 1196, 6801, 279, 7042, 13, 39529, 5109, 646, 2297, 916, 882, 11, 773, 358, 1184, 311, 1281, 2704, 358, 614, 279, 1429, 3213, 821, 13, 358, 6099, 429, 279, 7042, 12396, 369, 3598, 9720, 13289, 11, 323, 7025, 807, 2299, 44868, 13, 358, 1744, 279, 7042, 315, 12095, 374, 916, 220, 18, 3526, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 6771, 752, 1744, 1112, 16925, 11, 358, 19091, 432, 1660, 2163, 220, 18, 13, 20, 3526, 13, 358, 1265, 7683, 429, 11, 714, 2474, 358, 646, 1405, 1401, 432, 705, 1290, 1431, 11, 358, 3278, 728, 448, 429, 1372, 369, 1431, 382, 7039, 11, 2036, 1677, 279, 2033, 304, 4718, 3561, 374, 2441, 12893, 13, 4718, 7460, 1376, 19083, 13530, 11, 323, 432, 1265, 387, 10277, 23126, 11, 448, 76602, 304, 279, 1290, 7482, 311, 5648, 5975, 13, 358, 1265, 1281, 2704, 279, 19482, 374, 4396, 11, 1667, 54231, 15423, 323, 38929, 34901, 382, 785, 1196, 2578, 387, 264, 5458, 3238, 389, 264, 2390, 476, 8365, 4325, 6243, 705, 264, 3910, 1380, 807, 1184, 13382, 821, 6157, 13, 2379, 3129, 10161, 369, 264, 3151, 3561, 11, 773, 807, 2299, 4658, 7853, 315, 4718, 323, 1366, 279, 1995, 32930, 429, 1616, 369, 4135, 22314, 476, 17590, 1119, 2441, 3766, 382, 40, 1265, 1083, 2908, 421, 807, 1184, 803, 11682, 821, 11, 1075, 279, 1429, 3213, 1042, 279, 7042, 374, 504, 476, 421, 807, 1184, 5107, 1995, 7797, 1101, 279, 7042, 13, 1988, 2474, 862, 1681, 374, 3151, 11, 358, 3278, 9214, 311, 279, 6722, 11, 1181, 829, 11, 323, 279, 7042, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 614, 264, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 264, 1376, 330, 44441, 1, 448, 279, 897, 315, 220, 18, 13, 20, 3526, 13, 358, 3278, 3561, 432, 12440, 11, 22573, 1052, 525, 902, 27748, 76602, 311, 10306, 2697, 4718, 382, 40, 1744, 429, 14521, 678, 279, 8502, 13, 4695, 11, 882, 311, 3270, 279, 2033, 27079, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 18, 20, 15, 15, 15, 15, 15, 198, 532, 73594, 151643], 'meta_info': {'id': 'd3edff868a7e4ee68e9dd967e3070c3b', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 431, 'completion_tokens': 459, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.437822729349136, 'response_sent_to_client_ts': 1781048043.728515}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.41s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.29s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.31s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:44,  4.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:44,  4.99s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:07,  2.28s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:07,  2.28s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:17,  1.40s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:17,  1.40s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:52,  1.04it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:52,  1.04it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:28,  1.83it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:28,  1.83it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:21,  2.34it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:21,  2.34it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:17,  2.83it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:17,  2.83it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:14,  3.47it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:14,  3.47it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:11,  4.16it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:11,  4.16it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:09,  4.85it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:09,  4.85it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:08,  5.59it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:08,  5.59it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:07,  6.39it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:07,  6.39it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:07,  6.39it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:05,  7.76it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:05,  7.76it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:05,  7.76it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:04,  9.27it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:04,  9.27it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:04,  9.27it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:07<00:03, 11.12it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:07<00:03, 11.12it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:08<00:03, 11.12it/s]

    Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:08<00:03, 11.12it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:08<00:02, 14.50it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:08<00:02, 14.50it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:08<00:02, 14.50it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:08<00:02, 14.50it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:08<00:01, 18.04it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:08<00:01, 18.04it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:08<00:01, 18.04it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:08<00:01, 18.04it/s]

    Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:08<00:01, 18.04it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:08<00:01, 22.87it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:08<00:01, 22.87it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:08<00:01, 22.87it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:08<00:01, 22.87it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:08<00:01, 22.87it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:08<00:01, 22.87it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:08<00:00, 28.74it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:08<00:00, 28.74it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:08<00:00, 28.74it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:08<00:00, 28.74it/s]

    Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:08<00:00, 28.74it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:08<00:00, 28.74it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:08<00:00, 28.74it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:08<00:00, 35.39it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:08<00:00, 35.39it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:08<00:00, 35.39it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:08<00:00, 35.39it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:08<00:00, 35.39it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:08<00:00, 35.39it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:08<00:00, 35.39it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:08<00:00, 40.78it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:08<00:00, 40.78it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:08<00:00, 40.78it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:08<00:00, 40.78it/s]

    Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:08<00:00, 40.78it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:08<00:00, 40.78it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:08<00:00, 40.78it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:08<00:00, 40.78it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:08<00:00, 47.21it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:08<00:00, 47.21it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:08<00:00, 47.21it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:08<00:00, 47.21it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:08<00:00, 47.21it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:08<00:00, 47.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.07 GB):   2%|▏         | 1/58 [00:00<00:21,  2.64it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.04 GB):   2%|▏         | 1/58 [00:00<00:21,  2.64it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.04 GB):   3%|▎         | 2/58 [00:00<00:18,  2.96it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.03 GB):   3%|▎         | 2/58 [00:00<00:18,  2.96it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.03 GB):   5%|▌         | 3/58 [00:00<00:16,  3.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.03 GB):   5%|▌         | 3/58 [00:00<00:16,  3.38it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.03 GB):   7%|▋         | 4/58 [00:01<00:14,  3.61it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.00 GB):   7%|▋         | 4/58 [00:01<00:14,  3.61it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.00 GB):   9%|▊         | 5/58 [00:01<00:13,  3.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.00 GB):   9%|▊         | 5/58 [00:01<00:13,  3.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.00 GB):  10%|█         | 6/58 [00:01<00:11,  4.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.00 GB):  10%|█         | 6/58 [00:01<00:11,  4.38it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.00 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.00 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.00 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=25.00 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.89it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=25.00 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=24.94 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=24.94 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.56it/s]Capturing num tokens (num_tokens=3584 avail_mem=24.94 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.56it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=24.94 GB):  19%|█▉        | 11/58 [00:02<00:09,  4.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=24.94 GB):  19%|█▉        | 11/58 [00:02<00:09,  4.84it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=24.94 GB):  21%|██        | 12/58 [00:02<00:10,  4.55it/s]Capturing num tokens (num_tokens=3072 avail_mem=24.93 GB):  21%|██        | 12/58 [00:02<00:10,  4.55it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=24.93 GB):  22%|██▏       | 13/58 [00:03<00:10,  4.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=24.93 GB):  22%|██▏       | 13/58 [00:03<00:10,  4.36it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=24.93 GB):  24%|██▍       | 14/58 [00:03<00:10,  4.38it/s]Capturing num tokens (num_tokens=2560 avail_mem=24.93 GB):  24%|██▍       | 14/58 [00:03<00:10,  4.38it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=24.93 GB):  26%|██▌       | 15/58 [00:03<00:09,  4.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=24.93 GB):  26%|██▌       | 15/58 [00:03<00:09,  4.48it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=24.93 GB):  28%|██▊       | 16/58 [00:03<00:10,  3.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.93 GB):  28%|██▊       | 16/58 [00:03<00:10,  3.94it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=24.93 GB):  29%|██▉       | 17/58 [00:04<00:10,  3.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=24.93 GB):  29%|██▉       | 17/58 [00:04<00:10,  3.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=24.92 GB):  29%|██▉       | 17/58 [00:04<00:10,  3.75it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=24.92 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.55 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.54 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.14it/s]Capturing num tokens (num_tokens=960 avail_mem=42.54 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.14it/s] Capturing num tokens (num_tokens=960 avail_mem=42.54 GB):  38%|███▊      | 22/58 [00:04<00:04,  8.18it/s]Capturing num tokens (num_tokens=896 avail_mem=42.53 GB):  38%|███▊      | 22/58 [00:04<00:04,  8.18it/s]

    Capturing num tokens (num_tokens=832 avail_mem=42.53 GB):  38%|███▊      | 22/58 [00:04<00:04,  8.18it/s]Capturing num tokens (num_tokens=832 avail_mem=42.53 GB):  41%|████▏     | 24/58 [00:04<00:03, 10.03it/s]Capturing num tokens (num_tokens=768 avail_mem=42.52 GB):  41%|████▏     | 24/58 [00:04<00:03, 10.03it/s]Capturing num tokens (num_tokens=704 avail_mem=42.52 GB):  41%|████▏     | 24/58 [00:04<00:03, 10.03it/s]Capturing num tokens (num_tokens=640 avail_mem=42.52 GB):  41%|████▏     | 24/58 [00:04<00:03, 10.03it/s]Capturing num tokens (num_tokens=640 avail_mem=42.52 GB):  47%|████▋     | 27/58 [00:04<00:02, 13.00it/s]Capturing num tokens (num_tokens=576 avail_mem=42.51 GB):  47%|████▋     | 27/58 [00:04<00:02, 13.00it/s]

    Capturing num tokens (num_tokens=512 avail_mem=42.51 GB):  47%|████▋     | 27/58 [00:04<00:02, 13.00it/s]Capturing num tokens (num_tokens=512 avail_mem=42.51 GB):  50%|█████     | 29/58 [00:04<00:02, 14.19it/s]Capturing num tokens (num_tokens=480 avail_mem=42.50 GB):  50%|█████     | 29/58 [00:04<00:02, 14.19it/s]Capturing num tokens (num_tokens=448 avail_mem=42.50 GB):  50%|█████     | 29/58 [00:04<00:02, 14.19it/s]Capturing num tokens (num_tokens=448 avail_mem=42.50 GB):  53%|█████▎    | 31/58 [00:04<00:01, 15.16it/s]Capturing num tokens (num_tokens=416 avail_mem=42.50 GB):  53%|█████▎    | 31/58 [00:04<00:01, 15.16it/s]

    Capturing num tokens (num_tokens=384 avail_mem=42.50 GB):  53%|█████▎    | 31/58 [00:04<00:01, 15.16it/s]Capturing num tokens (num_tokens=352 avail_mem=42.49 GB):  53%|█████▎    | 31/58 [00:05<00:01, 15.16it/s]Capturing num tokens (num_tokens=352 avail_mem=42.49 GB):  59%|█████▊    | 34/58 [00:05<00:01, 16.83it/s]Capturing num tokens (num_tokens=320 avail_mem=42.49 GB):  59%|█████▊    | 34/58 [00:05<00:01, 16.83it/s]Capturing num tokens (num_tokens=288 avail_mem=42.49 GB):  59%|█████▊    | 34/58 [00:05<00:01, 16.83it/s]Capturing num tokens (num_tokens=256 avail_mem=42.49 GB):  59%|█████▊    | 34/58 [00:05<00:01, 16.83it/s]

    Capturing num tokens (num_tokens=256 avail_mem=42.49 GB):  64%|██████▍   | 37/58 [00:05<00:01, 18.65it/s]Capturing num tokens (num_tokens=240 avail_mem=42.48 GB):  64%|██████▍   | 37/58 [00:05<00:01, 18.65it/s]Capturing num tokens (num_tokens=224 avail_mem=42.48 GB):  64%|██████▍   | 37/58 [00:05<00:01, 18.65it/s]Capturing num tokens (num_tokens=208 avail_mem=42.47 GB):  64%|██████▍   | 37/58 [00:05<00:01, 18.65it/s]Capturing num tokens (num_tokens=208 avail_mem=42.47 GB):  69%|██████▉   | 40/58 [00:05<00:00, 20.98it/s]Capturing num tokens (num_tokens=192 avail_mem=42.47 GB):  69%|██████▉   | 40/58 [00:05<00:00, 20.98it/s]Capturing num tokens (num_tokens=176 avail_mem=42.47 GB):  69%|██████▉   | 40/58 [00:05<00:00, 20.98it/s]Capturing num tokens (num_tokens=160 avail_mem=42.47 GB):  69%|██████▉   | 40/58 [00:05<00:00, 20.98it/s]

    Capturing num tokens (num_tokens=160 avail_mem=42.47 GB):  74%|███████▍  | 43/58 [00:05<00:00, 21.98it/s]Capturing num tokens (num_tokens=144 avail_mem=42.46 GB):  74%|███████▍  | 43/58 [00:05<00:00, 21.98it/s]Capturing num tokens (num_tokens=128 avail_mem=42.46 GB):  74%|███████▍  | 43/58 [00:05<00:00, 21.98it/s]Capturing num tokens (num_tokens=112 avail_mem=42.46 GB):  74%|███████▍  | 43/58 [00:05<00:00, 21.98it/s]Capturing num tokens (num_tokens=96 avail_mem=42.45 GB):  74%|███████▍  | 43/58 [00:05<00:00, 21.98it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=42.45 GB):  81%|████████  | 47/58 [00:05<00:00, 20.56it/s]Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  81%|████████  | 47/58 [00:05<00:00, 20.56it/s]Capturing num tokens (num_tokens=64 avail_mem=61.52 GB):  81%|████████  | 47/58 [00:05<00:00, 20.56it/s]Capturing num tokens (num_tokens=48 avail_mem=61.51 GB):  81%|████████  | 47/58 [00:05<00:00, 20.56it/s]Capturing num tokens (num_tokens=32 avail_mem=61.51 GB):  81%|████████  | 47/58 [00:05<00:00, 20.56it/s]Capturing num tokens (num_tokens=32 avail_mem=61.51 GB):  88%|████████▊ | 51/58 [00:05<00:00, 24.72it/s]Capturing num tokens (num_tokens=28 avail_mem=61.51 GB):  88%|████████▊ | 51/58 [00:05<00:00, 24.72it/s]Capturing num tokens (num_tokens=24 avail_mem=61.51 GB):  88%|████████▊ | 51/58 [00:05<00:00, 24.72it/s]Capturing num tokens (num_tokens=20 avail_mem=61.50 GB):  88%|████████▊ | 51/58 [00:05<00:00, 24.72it/s]Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  88%|████████▊ | 51/58 [00:05<00:00, 24.72it/s]

    Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  95%|█████████▍| 55/58 [00:05<00:00, 28.08it/s]Capturing num tokens (num_tokens=12 avail_mem=61.50 GB):  95%|█████████▍| 55/58 [00:05<00:00, 28.08it/s]Capturing num tokens (num_tokens=8 avail_mem=61.49 GB):  95%|█████████▍| 55/58 [00:05<00:00, 28.08it/s] Capturing num tokens (num_tokens=4 avail_mem=61.49 GB):  95%|█████████▍| 55/58 [00:05<00:00, 28.08it/s]Capturing num tokens (num_tokens=4 avail_mem=61.49 GB): 100%|██████████| 58/58 [00:05<00:00,  9.79it/s]


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
    
    Generated text: Okay, so the user is asking for the information and population of the capital of France in JSON format. Hmm, let me start by identifying what they need. The capital of France is Paris, so that's the key piece of information. They also want the population, so I need to find the latest data on that.
    
    Wait, I remember that population figures can change over time. I think as of 2023, the population of Paris is around 2.2 million people. But I should double-check that to make sure it's accurate. Maybe I can recall that the population has been growing steadily, so 2.2 million seems about right.
    
    Now, the user wants this in JSON format. I know JSON stands for JavaScript Object Notation, and it's commonly used for data interchange. So I'll structure it with a key-value pair where the key is "capital" and the value is another object containing "city" and "population". 
    
    I should make sure the JSON syntax is correct, with proper commas and quotation marks. Also, the population number should be an integer to keep it clean. Putting it all together, the JSON object will look neat and be easy for the user to use if they're working with data in a programming context.
    
    I think that's all. The user probably just needs this information quickly, so providing it in a clear and concise JSON format should meet their needs without any extra fluff.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "capital": {
        "city": "Paris",
        "population": 2215000
      }
    }
    ```



```python
llm.shutdown()
```
