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

    Multi-thread loading shards:  50% Completed | 1/2 [00:04<00:04,  4.23s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:06<00:00,  3.14s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:06<00:00,  3.31s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:10,  5.45s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:10,  5.45s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:12,  2.37s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:12,  2.37s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:15,  1.37s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:15,  1.37s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:48,  1.11it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:48,  1.11it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:33,  1.58it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:24,  2.13it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:24,  2.13it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.79it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.79it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.54it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.54it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.40it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.40it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.40it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.12it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.12it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.12it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:05,  7.69it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:05,  7.69it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:05,  7.69it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.31it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.31it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.31it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.28it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.28it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.28it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.28it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.53it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.53it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.53it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.53it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.53it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.50it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.50it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.50it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.50it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.50it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.50it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 19.50it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.23it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.23it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.23it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.23it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.23it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.23it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.23it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 35.42it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 35.42it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 35.42it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 35.42it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 35.42it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 35.42it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 35.42it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 35.42it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 43.97it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 43.97it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:07<00:00, 43.97it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:07<00:00, 43.97it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:07<00:00, 43.97it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:07<00:00, 43.97it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:07<00:00, 43.97it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:07<00:00, 43.97it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:07<00:00, 50.46it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:07<00:00, 50.46it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 50.46it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 50.46it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:07<00:00, 50.46it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:07<00:00, 50.46it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:08<00:00, 50.46it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:08<00:00, 50.46it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:08<00:00, 50.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=57.50 GB):   2%|▏         | 1/58 [00:00<00:16,  3.37it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.47 GB):   2%|▏         | 1/58 [00:00<00:16,  3.37it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=57.47 GB):   3%|▎         | 2/58 [00:00<00:15,  3.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.46 GB):   3%|▎         | 2/58 [00:00<00:15,  3.55it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=57.46 GB):   5%|▌         | 3/58 [00:00<00:14,  3.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.46 GB):   5%|▌         | 3/58 [00:00<00:14,  3.82it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.46 GB):   7%|▋         | 4/58 [00:01<00:13,  4.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.46 GB):   7%|▋         | 4/58 [00:01<00:13,  4.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.46 GB):   9%|▊         | 5/58 [00:01<00:12,  4.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.46 GB):   9%|▊         | 5/58 [00:01<00:12,  4.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.46 GB):  10%|█         | 6/58 [00:01<00:11,  4.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.64 GB):  10%|█         | 6/58 [00:01<00:11,  4.72it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=56.64 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.64 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.64 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.64 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.59it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=56.64 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.64 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.64 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.64 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.59it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=56.64 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.64 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.64 GB):  21%|██        | 12/58 [00:02<00:05,  7.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.64 GB):  21%|██        | 12/58 [00:02<00:05,  7.77it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=56.64 GB):  21%|██        | 12/58 [00:02<00:05,  7.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.64 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.76it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.63 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.63 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.76it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=56.63 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.63 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.63 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.63 GB):  31%|███       | 18/58 [00:02<00:03, 11.63it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.62 GB):  31%|███       | 18/58 [00:02<00:03, 11.63it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.62 GB):  31%|███       | 18/58 [00:02<00:03, 11.63it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=56.62 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.61 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.55it/s]Capturing num tokens (num_tokens=960 avail_mem=56.60 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.55it/s] Capturing num tokens (num_tokens=896 avail_mem=56.60 GB):  34%|███▍      | 20/58 [00:02<00:02, 13.55it/s]Capturing num tokens (num_tokens=896 avail_mem=56.60 GB):  40%|███▉      | 23/58 [00:02<00:02, 16.86it/s]Capturing num tokens (num_tokens=832 avail_mem=56.60 GB):  40%|███▉      | 23/58 [00:02<00:02, 16.86it/s]Capturing num tokens (num_tokens=768 avail_mem=56.59 GB):  40%|███▉      | 23/58 [00:02<00:02, 16.86it/s]Capturing num tokens (num_tokens=704 avail_mem=56.59 GB):  40%|███▉      | 23/58 [00:02<00:02, 16.86it/s]

    Capturing num tokens (num_tokens=704 avail_mem=56.59 GB):  45%|████▍     | 26/58 [00:02<00:01, 19.74it/s]Capturing num tokens (num_tokens=640 avail_mem=56.59 GB):  45%|████▍     | 26/58 [00:02<00:01, 19.74it/s]Capturing num tokens (num_tokens=576 avail_mem=56.58 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.74it/s]Capturing num tokens (num_tokens=512 avail_mem=56.58 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.74it/s]Capturing num tokens (num_tokens=480 avail_mem=56.58 GB):  45%|████▍     | 26/58 [00:03<00:01, 19.74it/s]Capturing num tokens (num_tokens=480 avail_mem=56.58 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.52it/s]Capturing num tokens (num_tokens=448 avail_mem=56.57 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.52it/s]Capturing num tokens (num_tokens=416 avail_mem=56.57 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.52it/s]Capturing num tokens (num_tokens=384 avail_mem=56.57 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.52it/s]

    Capturing num tokens (num_tokens=352 avail_mem=56.56 GB):  52%|█████▏    | 30/58 [00:03<00:01, 23.52it/s]Capturing num tokens (num_tokens=352 avail_mem=56.56 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.73it/s]Capturing num tokens (num_tokens=320 avail_mem=56.56 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.73it/s]Capturing num tokens (num_tokens=288 avail_mem=56.56 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.73it/s]Capturing num tokens (num_tokens=256 avail_mem=56.56 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.73it/s]Capturing num tokens (num_tokens=240 avail_mem=56.55 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.73it/s]Capturing num tokens (num_tokens=240 avail_mem=56.55 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.78it/s]Capturing num tokens (num_tokens=224 avail_mem=56.55 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.78it/s]Capturing num tokens (num_tokens=208 avail_mem=56.54 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.78it/s]Capturing num tokens (num_tokens=192 avail_mem=56.54 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.78it/s]

    Capturing num tokens (num_tokens=176 avail_mem=56.54 GB):  66%|██████▌   | 38/58 [00:03<00:00, 28.78it/s]Capturing num tokens (num_tokens=176 avail_mem=56.54 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.03it/s]Capturing num tokens (num_tokens=160 avail_mem=56.54 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.03it/s]Capturing num tokens (num_tokens=144 avail_mem=56.53 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.03it/s]Capturing num tokens (num_tokens=128 avail_mem=56.53 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.03it/s]Capturing num tokens (num_tokens=112 avail_mem=56.53 GB):  72%|███████▏  | 42/58 [00:03<00:00, 31.03it/s]

    Capturing num tokens (num_tokens=112 avail_mem=56.53 GB):  79%|███████▉  | 46/58 [00:03<00:00, 27.57it/s]Capturing num tokens (num_tokens=96 avail_mem=56.52 GB):  79%|███████▉  | 46/58 [00:03<00:00, 27.57it/s] Capturing num tokens (num_tokens=80 avail_mem=56.52 GB):  79%|███████▉  | 46/58 [00:03<00:00, 27.57it/s]Capturing num tokens (num_tokens=64 avail_mem=56.52 GB):  79%|███████▉  | 46/58 [00:03<00:00, 27.57it/s]Capturing num tokens (num_tokens=64 avail_mem=56.52 GB):  84%|████████▍ | 49/58 [00:03<00:00, 28.14it/s]Capturing num tokens (num_tokens=48 avail_mem=56.51 GB):  84%|████████▍ | 49/58 [00:03<00:00, 28.14it/s]Capturing num tokens (num_tokens=32 avail_mem=56.51 GB):  84%|████████▍ | 49/58 [00:03<00:00, 28.14it/s]Capturing num tokens (num_tokens=28 avail_mem=56.51 GB):  84%|████████▍ | 49/58 [00:03<00:00, 28.14it/s]Capturing num tokens (num_tokens=24 avail_mem=56.50 GB):  84%|████████▍ | 49/58 [00:03<00:00, 28.14it/s]

    Capturing num tokens (num_tokens=24 avail_mem=56.50 GB):  91%|█████████▏| 53/58 [00:03<00:00, 31.06it/s]Capturing num tokens (num_tokens=20 avail_mem=56.50 GB):  91%|█████████▏| 53/58 [00:03<00:00, 31.06it/s]Capturing num tokens (num_tokens=16 avail_mem=56.50 GB):  91%|█████████▏| 53/58 [00:03<00:00, 31.06it/s]Capturing num tokens (num_tokens=12 avail_mem=56.49 GB):  91%|█████████▏| 53/58 [00:03<00:00, 31.06it/s]Capturing num tokens (num_tokens=8 avail_mem=56.49 GB):  91%|█████████▏| 53/58 [00:03<00:00, 31.06it/s] Capturing num tokens (num_tokens=8 avail_mem=56.49 GB):  98%|█████████▊| 57/58 [00:03<00:00, 33.07it/s]Capturing num tokens (num_tokens=4 avail_mem=56.49 GB):  98%|█████████▊| 57/58 [00:03<00:00, 33.07it/s]Capturing num tokens (num_tokens=4 avail_mem=56.49 GB): 100%|██████████| 58/58 [00:03<00:00, 14.61it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out how to get the current date and time in New York and the weather there using the functions provided. Let me start by understanding what each function does.<br><br>First, there's get_current_weather. It requires city, state, and unit. I know the city is New York, but I'm not sure about the state abbreviation. I think New York is NY. The unit could be either Celsius or Fahrenheit. I'm not sure which one to choose, but maybe it's better to check both or just pick one. Wait, the user didn't specify, so maybe I can use a default, like Fahrenheit since many people are more familiar with it.<br><br>Next, the get_current_date function requires a timezone. The user is in New York, so the timezone is probably 'America/New_York'. I'll need to make sure the timezone parameter is correctly formatted.<br><br>Now, I need to decide which function to call first. Since the user is asking for both date and weather, I might need to call both functions. But I can't call them at the same time, so I have to choose one function to call in each message. So first, I'll call get_current_date with the timezone parameter. Then, I'll call get_current_weather with New York, NY, and Fahrenheit.<br><br>Wait, but the user wants both pieces of information, so maybe I should structure it so that each function call is a separate message. But the user's message is one, so perhaps I can include both in the same response, but each function call in its own message. However, the instructions say to only call one function at a time, so I should handle each function call separately in separate messages.<br><br>But the user's query is asking for both, so perhaps I can structure it by first getting the date, then the weather. Alternatively, maybe I can include both in the same response but separate the function calls.<br><br>Wait, no, each function call is a separate message. So I need to first call get_current_date with the timezone, then call get_current_weather with the city, state, and unit.<br><br>But in the initial response, the user is in New York and wants both. So perhaps I should first get the date, then the weather, or vice versa. But the example given in the instructions shows a single function call, so perhaps I should structure it as two separate messages: one for the date and one for the weather.<br><br>Wait, but the user is asking for both in one message, so perhaps I can include both function calls in the same response but separate them with line breaks. However, the instructions say to put the entire function call on one line, so maybe I can structure it as two separate lines.<br><br>But looking back, the example given was a single function call, so perhaps each function call should be in its own message. So I'll first call get_current_date with the timezone, then call get_current_weather with the necessary parameters.<br><br>Wait, but the user's message is a single request, so perhaps I can structure it as a single message with both function calls, but that's not possible because each function call is separate. So I think the correct approach is to first call get_current_date and then get_current_weather, each in their own message.<br><br>But the user's message is one, so maybe I can combine both function calls into one message by using a logical OR or something, but that might complicate things. Alternatively, perhaps I can call both functions in the same message, but each function call is on a separate line within the same message.<br><br>Wait, the instructions say that the entire function call should be on one line, but perhaps if I need to call two functions, I can do it in two separate messages. So first, I'll call get_current_date with the timezone, then in the next message, call get_current_weather with the city, state, and unit.<br><br>But the user's query is a single message, so maybe I can structure it as a single response with both function calls, each on their own line. Let me check the example given in the instructions. It shows a single function call, but perhaps I can format it with each function on a separate line within the same message.<br><br>Alternatively, perhaps I can combine the two function calls into a single JSON object, but that might not be straightforward. So maybe it's better to handle each function call separately.<br><br>Wait, no, each function call is separate, so I'll need to send two separate messages. But the user's query is a single message, so perhaps I can structure it as a single response with both function calls, each on their own line.<br><br>But according to the instructions, each function call should be on its own line, so I'll need to send two separate messages. But the user's query is one, so perhaps I can structure it as a single message with both function calls, each on a separate line.<br><br>Wait, but the instructions say to put the entire function call on one line, so perhaps I can structure it as a single function call that includes both, but that's not possible because each function requires its own parameters.<br><br>Hmm, maybe I should just proceed with calling each function separately in two messages. So first, I'll send the date function, then the weather function.<br><br>But the user's query is a single message, so perhaps I can structure it as two separate function calls within the same message, each on their own line.<br><br>Wait, but the instructions say to put the entire function call on one line, so perhaps each function call is a separate line within the same message. So I can format it like this:<br><br><function=get_current_date>{{<br>  "timezone": "America/New_York"<br>}}</function><br><function=get_current_weather>{{<br>  "city": "New York",<br>  "state": "NY",<br>  "unit": "Fahrenheit"<br>}}</function><br><br>But I'm not sure if that's allowed. Alternatively, perhaps I can list them as separate function calls in the same message, each on a new line.<br><br>Wait, the example given in the instructions was a single function call, so perhaps each function call is a separate message. So I'll need to send two messages: one for the date and one for the weather.<br><br>But the user's query is one message, so perhaps I can include both function calls in the same message, each on their own line.<br><br>Alternatively, perhaps I can structure it as a single message with both function calls, each on a separate line, but within the same function call. But that's not possible because each function requires its own parameters.<br><br>Wait, perhaps I can use a logical OR or something, but that's not how function calls work. So I think the correct approach is to send two separate function calls, each on their own line, within the same message.<br><br>So, in the response, I'll write:<br><br><function=get_current_date>{{<br>  "timezone": "America/New_York"<br>}}</function><br><function=get_current_weather>{{<br>  "city": "New York",<br>  "state": "NY",<br>  "unit": "Fahrenheit"<br>}}</function><br><br>But I'm not sure if that's the correct format. Alternatively, perhaps each function call should be on its own line within the same message, but I'm not sure if that's allowed.<br><br>Wait, the instructions say to put the entire function call on one line, so perhaps each function call is a separate line. So I'll structure it as two separate messages, each with a function call.<br><br>But the user's query is a single message, so perhaps I can include both function calls in the same message, each on a separate line.<br><br>Wait, perhaps the user expects both pieces of information in the response, so perhaps I can structure it as a single message with both function calls, each on their own line.<br><br>So, in the response, I'll write:<br><br><function=get_current_date>{{<br>  "timezone": "America/New_York"<br>}}</function><br><function=get_current_weather>{{<br>  "city": "New York",<br>  "state": "NY",<br>  "unit": "Fahrenheit"<br>}}</function><br><br>Yes, that should work. Each function call is on its own line, and the entire message includes both function calls. So the user will get the current date and time in New York, and the current weather in Fahrenheit.<br><br>I think that's the correct approach. So I'll proceed with that.<br><br><br>content: <br><br>To get the current date and time in New York, and the weather, you can call the `get_current_date` and `get_current_weather` functions separately. Here are the function calls:<br><br><function=get_current_date>{  <br>  "timezone": "America/New_York"  <br>}</function><br><br><function=get_current_weather>{  <br>  "city": "New York",  <br>  "state": "NY",  <br>  "unit": "fahrenheit"  <br>}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '6f0da0df762a4c33bfc200f4a766861c', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 6.007787048816681, 'response_sent_to_client_ts': 1780365991.8577523}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'ad02fbb2bdec4c70a9e6f1c939fe9fda', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.190279982052743, 'response_sent_to_client_ts': 1780365997.0585456}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'af498608c6144da898e634bdc5715279', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.17451747320592403, 'response_sent_to_client_ts': 1780365997.2572396}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'c2cb3791438946ec87b2ec2b59be9b1a', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.17445349041372538, 'response_sent_to_client_ts': 1780365997.2572536}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '48a7304c8e4f42908ead656c817b1802', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.17426985269412398, 'response_sent_to_client_ts': 1780365997.2572577}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'df1c8135d54443138b06ba8e88065fb8', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 17.283935534767807, 'response_sent_to_client_ts': 1780366014.548018}}


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


<strong style='color: #00008B;'>{'text': 'Alright, so I need to figure out how to respond to this user\'s query. They asked for information and the population of the capital of France in JSON format. I know that the capital city is Paris, that\'s a given. Now, the population part is a bit tricky because I don\'t have the exact current numbers on hand. Population data changes every year and it\'s not something I can calculate on the spot.\n\nFirst, I should acknowledge that while I don\'t have the latest population figure, I can provide the most recent data I have along with some additional relevant information about Paris. Gives the user a sense that I\'m helpful and that I can offer alternative information. I should include key facts about Paris to add value beyond the population figure they specifically asked for.\n\nThinking about the structure, JSON is an object format with key-value pairs. So I\'ll need to create a JSON object that includes sections like "description", "population", and "key_facts". Since the population is uncertain, I\'ll set it to null and suggest the user can update it with the latest data from official sources. For the key facts, I\'ll include the main landmarks, historical aspects, whether it\'s a cultural melting pot, and some notable attractions to give a comprehensive overview.\n\nI also need to make sure my response is clear and informative, explaining that while the population figure can\'t be provided definitively, the rest of the information should be helpful. I\'ll structure it in a way that\'s easy to read, perhaps using subheadings but since the user requested JSON, I\'ll format it accordingly without markdown.\n\nI should double-check the spelling of Paris and the key facts to ensure accuracy. Maybe elaborate a bit on each key fact to make it more engaging for the user. Once all that\'s organized, I can put it all together in a coherent JSON structure that\'s both informative and user-friendly.\n\nFinally, I\'ll make sure to present the information in a polite and helpful manner, inviting the user to provide more details if they need further assistance. That way, the interaction is smooth, and the user feels supported despite the missing data point.\n</think>\n\nHere is the information and population of the capital of France in JSON format:\n\n```json\n{\n  "capital": {\n    "name": "Paris",\n    "description": "The capital and largest city in France, known for its historical landmarks, culinary scene, and creative industries.",\n    "population": null, // Population data may vary and should be confirmed from reliable sources.\n    "tags": ["opolis", "history", "arts", "food"],\n    "fact1": "Paris is one of the most visited cities in the world due to its rich history and cultural significance.",\n    "fact2": "The city is renowned for landmarks like the Eiffel Tower, the Louvre Museum, and协a-Père Auguste Renoir Gardens.",\n    "fact3": "Paris has a diverse cultural melting pot, attracting people from various countries and ethnicities.",\n    "fact4": "The city is a hub for fashion, design, and intellectual activity."\n  }\n}\n```\n\nLet me know if you need further details!', 'output_ids': [71486, 11, 773, 358, 1184, 311, 7071, 700, 1246, 311, 5889, 311, 419, 1196, 594, 3239, 13, 2379, 4588, 369, 1995, 323, 279, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 358, 1414, 429, 279, 6722, 3283, 374, 12095, 11, 429, 594, 264, 2661, 13, 4695, 11, 279, 7042, 949, 374, 264, 2699, 33453, 1576, 358, 1513, 944, 614, 279, 4734, 1482, 5109, 389, 1424, 13, 39529, 821, 4344, 1449, 1042, 323, 432, 594, 537, 2494, 358, 646, 11047, 389, 279, 7702, 382, 5338, 11, 358, 1265, 24645, 429, 1393, 358, 1513, 944, 614, 279, 5535, 7042, 7071, 11, 358, 646, 3410, 279, 1429, 3213, 821, 358, 614, 3156, 448, 1045, 5107, 9760, 1995, 911, 12095, 13, 68913, 279, 1196, 264, 5530, 429, 358, 2776, 10950, 323, 429, 358, 646, 3010, 10555, 1995, 13, 358, 1265, 2924, 1376, 13064, 911, 12095, 311, 912, 897, 7797, 279, 7042, 7071, 807, 11689, 4588, 369, 382, 93945, 911, 279, 5944, 11, 4718, 374, 458, 1633, 3561, 448, 1376, 19083, 13530, 13, 2055, 358, 3278, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 14158, 1075, 330, 4684, 497, 330, 44441, 497, 323, 330, 792, 761, 11359, 3263, 8704, 279, 7042, 374, 35118, 11, 358, 3278, 738, 432, 311, 845, 323, 4190, 279, 1196, 646, 2647, 432, 448, 279, 5535, 821, 504, 3946, 8173, 13, 1752, 279, 1376, 13064, 11, 358, 3278, 2924, 279, 1887, 59924, 11, 13656, 13566, 11, 3425, 432, 594, 264, 12752, 49584, 3338, 11, 323, 1045, 27190, 38491, 311, 2968, 264, 15817, 23251, 382, 40, 1083, 1184, 311, 1281, 2704, 847, 2033, 374, 2797, 323, 38219, 11, 25021, 429, 1393, 279, 7042, 7071, 646, 944, 387, 3897, 30288, 3132, 11, 279, 2732, 315, 279, 1995, 1265, 387, 10950, 13, 358, 3278, 5944, 432, 304, 264, 1616, 429, 594, 4135, 311, 1349, 11, 8365, 1667, 1186, 1983, 819, 714, 2474, 279, 1196, 11223, 4718, 11, 358, 3278, 3561, 432, 27079, 2041, 50494, 382, 40, 1265, 1990, 15934, 279, 42429, 315, 12095, 323, 279, 1376, 13064, 311, 5978, 13403, 13, 10696, 35967, 264, 2699, 389, 1817, 1376, 2097, 311, 1281, 432, 803, 22570, 369, 279, 1196, 13, 9646, 678, 429, 594, 16645, 11, 358, 646, 2182, 432, 678, 3786, 304, 264, 55787, 4718, 5944, 429, 594, 2176, 38219, 323, 1196, 21896, 382, 23949, 11, 358, 3278, 1281, 2704, 311, 3042, 279, 1995, 304, 264, 47787, 323, 10950, 11566, 11, 41192, 279, 1196, 311, 3410, 803, 3565, 421, 807, 1184, 4623, 12994, 13, 2938, 1616, 11, 279, 16230, 374, 10876, 11, 323, 279, 1196, 11074, 7248, 8818, 279, 7402, 821, 1459, 624, 151649, 271, 8420, 374, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 341, 262, 330, 606, 788, 330, 59604, 756, 262, 330, 4684, 788, 330, 785, 6722, 323, 7772, 3283, 304, 9625, 11, 3881, 369, 1181, 13656, 59924, 11, 57341, 6109, 11, 323, 11521, 19102, 10346, 262, 330, 44441, 788, 845, 11, 442, 39529, 821, 1231, 13289, 323, 1265, 387, 10774, 504, 14720, 8173, 624, 262, 330, 14082, 788, 4383, 27665, 285, 497, 330, 18844, 497, 330, 7038, 497, 330, 13915, 8097, 262, 330, 33110, 16, 788, 330, 59604, 374, 825, 315, 279, 1429, 11994, 9720, 304, 279, 1879, 4152, 311, 1181, 9080, 3840, 323, 12752, 25361, 10346, 262, 330, 33110, 17, 788, 330, 785, 3283, 374, 35948, 369, 59924, 1075, 279, 468, 3092, 301, 21938, 11, 279, 9729, 48506, 16328, 11, 323, 99339, 64, 9299, 12068, 6156, 68, 63418, 404, 42043, 10346, 262, 330, 33110, 18, 788, 330, 59604, 702, 264, 16807, 12752, 49584, 3338, 11, 50547, 1251, 504, 5257, 5837, 323, 21551, 1361, 10346, 262, 330, 33110, 19, 788, 330, 785, 3283, 374, 264, 18719, 369, 11153, 11, 2884, 11, 323, 19634, 5702, 10040, 220, 456, 532, 13874, 19324, 10061, 752, 1414, 421, 498, 1184, 4623, 3565, 0, 151643], 'meta_info': {'id': '3129c7ccc4db45e6812929888c271549', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 431, 'completion_tokens': 644, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.875090579967946, 'response_sent_to_client_ts': 1780366020.4334877}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.13s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.05s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.06s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:02,  5.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:02,  5.31s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:17,  2.45s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:17,  2.45s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:23,  1.53s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:23,  1.53s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:57,  1.07s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:57,  1.07s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:43,  1.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:43,  1.23it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:33,  1.56it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:33,  1.56it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:26,  1.92it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:26,  1.92it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:21,  2.28it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:21,  2.28it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:17,  2.78it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:17,  2.78it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:15,  3.15it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:15,  3.15it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:12,  3.74it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:12,  3.74it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:11,  4.09it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:11,  4.09it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:09,  4.76it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:09,  4.76it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:08,  5.21it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:08,  5.21it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:07,  5.82it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:07,  5.82it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:09<00:07,  5.82it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:05,  7.02it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:05,  7.02it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:09<00:05,  7.02it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:09<00:04,  8.66it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:09<00:04,  8.66it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:09<00:04,  8.66it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:09<00:03,  9.64it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:09<00:03,  9.64it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:09<00:03,  9.64it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:09<00:03,  9.64it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:09<00:02, 12.53it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:09<00:02, 12.53it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:09<00:02, 12.53it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:09<00:02, 12.53it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:09<00:02, 15.49it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:09<00:02, 15.49it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:09<00:02, 15.49it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:09<00:02, 15.49it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:09<00:01, 17.96it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:09<00:01, 17.96it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:09<00:01, 17.96it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:09<00:01, 17.96it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:09<00:01, 17.96it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:09<00:01, 21.64it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:09<00:01, 21.64it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:10<00:01, 21.64it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:10<00:01, 21.64it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:10<00:01, 21.64it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:10<00:00, 25.57it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:10<00:00, 25.57it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:10<00:00, 25.57it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:10<00:00, 25.57it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:10<00:00, 25.57it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:10<00:00, 28.10it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:10<00:00, 28.10it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:10<00:00, 28.10it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:10<00:00, 28.10it/s]

    Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:10<00:00, 28.10it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:10<00:00, 30.95it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:10<00:00, 30.95it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:10<00:00, 30.95it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:10<00:00, 30.95it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:10<00:00, 30.95it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:10<00:00, 32.92it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:10<00:00, 32.92it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:10<00:00, 32.92it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:10<00:00, 32.92it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:10<00:00, 32.92it/s]

    Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:10<00:00, 32.92it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:10<00:00, 37.58it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:10<00:00, 37.58it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:10<00:00, 37.58it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:10<00:00, 37.58it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=42.14 GB):   2%|▏         | 1/58 [00:00<00:23,  2.42it/s]Capturing num tokens (num_tokens=7680 avail_mem=41.44 GB):   2%|▏         | 1/58 [00:00<00:23,  2.42it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=41.44 GB):   3%|▎         | 2/58 [00:00<00:21,  2.66it/s]Capturing num tokens (num_tokens=7168 avail_mem=41.42 GB):   3%|▎         | 2/58 [00:00<00:21,  2.66it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=41.42 GB):   5%|▌         | 3/58 [00:01<00:19,  2.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=41.41 GB):   5%|▌         | 3/58 [00:01<00:19,  2.86it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=41.41 GB):   7%|▋         | 4/58 [00:01<00:16,  3.19it/s]Capturing num tokens (num_tokens=6144 avail_mem=41.40 GB):   7%|▋         | 4/58 [00:01<00:16,  3.19it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=41.40 GB):   9%|▊         | 5/58 [00:01<00:15,  3.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.37 GB):   9%|▊         | 5/58 [00:01<00:15,  3.40it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=41.37 GB):  10%|█         | 6/58 [00:01<00:14,  3.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.39 GB):  10%|█         | 6/58 [00:01<00:14,  3.68it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=41.39 GB):  12%|█▏        | 7/58 [00:02<00:12,  3.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.38 GB):  12%|█▏        | 7/58 [00:02<00:12,  3.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.38 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.37 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.36it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=41.37 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.66it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.36 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.66it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.36 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.35 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.05it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=41.35 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.34 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.34 GB):  21%|██        | 12/58 [00:02<00:08,  5.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.33 GB):  21%|██        | 12/58 [00:02<00:08,  5.73it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=41.33 GB):  22%|██▏       | 13/58 [00:03<00:07,  6.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.32 GB):  22%|██▏       | 13/58 [00:03<00:07,  6.15it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.32 GB):  22%|██▏       | 13/58 [00:03<00:07,  6.15it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=41.32 GB):  26%|██▌       | 15/58 [00:03<00:05,  7.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.32 GB):  26%|██▌       | 15/58 [00:03<00:05,  7.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.32 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.32 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.32 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.88it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=41.32 GB):  31%|███       | 18/58 [00:03<00:04,  9.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=40.25 GB):  31%|███       | 18/58 [00:03<00:04,  9.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.25 GB):  31%|███       | 18/58 [00:03<00:04,  9.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.25 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.24 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.68it/s]Capturing num tokens (num_tokens=960 avail_mem=30.61 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.68it/s] Capturing num tokens (num_tokens=896 avail_mem=27.25 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.68it/s]

    Capturing num tokens (num_tokens=896 avail_mem=27.25 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.07it/s]Capturing num tokens (num_tokens=832 avail_mem=27.25 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.07it/s]Capturing num tokens (num_tokens=768 avail_mem=27.24 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.07it/s]Capturing num tokens (num_tokens=704 avail_mem=27.24 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.07it/s]Capturing num tokens (num_tokens=704 avail_mem=27.24 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.41it/s]Capturing num tokens (num_tokens=640 avail_mem=27.23 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.41it/s]Capturing num tokens (num_tokens=576 avail_mem=27.23 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.41it/s]Capturing num tokens (num_tokens=512 avail_mem=27.23 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.41it/s]

    Capturing num tokens (num_tokens=480 avail_mem=27.22 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.41it/s]Capturing num tokens (num_tokens=480 avail_mem=27.22 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.49it/s]Capturing num tokens (num_tokens=448 avail_mem=27.22 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.49it/s]Capturing num tokens (num_tokens=416 avail_mem=27.22 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.49it/s]Capturing num tokens (num_tokens=384 avail_mem=27.21 GB):  52%|█████▏    | 30/58 [00:04<00:01, 22.49it/s]Capturing num tokens (num_tokens=384 avail_mem=27.21 GB):  57%|█████▋    | 33/58 [00:04<00:01, 22.61it/s]Capturing num tokens (num_tokens=352 avail_mem=27.21 GB):  57%|█████▋    | 33/58 [00:04<00:01, 22.61it/s]Capturing num tokens (num_tokens=320 avail_mem=27.20 GB):  57%|█████▋    | 33/58 [00:04<00:01, 22.61it/s]

    Capturing num tokens (num_tokens=288 avail_mem=27.21 GB):  57%|█████▋    | 33/58 [00:04<00:01, 22.61it/s]Capturing num tokens (num_tokens=256 avail_mem=27.20 GB):  57%|█████▋    | 33/58 [00:04<00:01, 22.61it/s]Capturing num tokens (num_tokens=256 avail_mem=27.20 GB):  64%|██████▍   | 37/58 [00:04<00:00, 26.20it/s]Capturing num tokens (num_tokens=240 avail_mem=27.20 GB):  64%|██████▍   | 37/58 [00:04<00:00, 26.20it/s]Capturing num tokens (num_tokens=224 avail_mem=27.20 GB):  64%|██████▍   | 37/58 [00:04<00:00, 26.20it/s]Capturing num tokens (num_tokens=208 avail_mem=27.19 GB):  64%|██████▍   | 37/58 [00:04<00:00, 26.20it/s]Capturing num tokens (num_tokens=192 avail_mem=27.19 GB):  64%|██████▍   | 37/58 [00:04<00:00, 26.20it/s]Capturing num tokens (num_tokens=192 avail_mem=27.19 GB):  71%|███████   | 41/58 [00:04<00:00, 29.07it/s]Capturing num tokens (num_tokens=176 avail_mem=27.19 GB):  71%|███████   | 41/58 [00:04<00:00, 29.07it/s]Capturing num tokens (num_tokens=160 avail_mem=27.18 GB):  71%|███████   | 41/58 [00:04<00:00, 29.07it/s]

    Capturing num tokens (num_tokens=144 avail_mem=27.18 GB):  71%|███████   | 41/58 [00:04<00:00, 29.07it/s]Capturing num tokens (num_tokens=128 avail_mem=27.18 GB):  71%|███████   | 41/58 [00:04<00:00, 29.07it/s]Capturing num tokens (num_tokens=128 avail_mem=27.18 GB):  78%|███████▊  | 45/58 [00:04<00:00, 31.05it/s]Capturing num tokens (num_tokens=112 avail_mem=27.18 GB):  78%|███████▊  | 45/58 [00:04<00:00, 31.05it/s]Capturing num tokens (num_tokens=96 avail_mem=27.17 GB):  78%|███████▊  | 45/58 [00:04<00:00, 31.05it/s] Capturing num tokens (num_tokens=80 avail_mem=27.17 GB):  78%|███████▊  | 45/58 [00:04<00:00, 31.05it/s]Capturing num tokens (num_tokens=64 avail_mem=27.16 GB):  78%|███████▊  | 45/58 [00:04<00:00, 31.05it/s]Capturing num tokens (num_tokens=64 avail_mem=27.16 GB):  84%|████████▍ | 49/58 [00:04<00:00, 32.95it/s]Capturing num tokens (num_tokens=48 avail_mem=27.16 GB):  84%|████████▍ | 49/58 [00:04<00:00, 32.95it/s]Capturing num tokens (num_tokens=32 avail_mem=27.16 GB):  84%|████████▍ | 49/58 [00:04<00:00, 32.95it/s]

    Capturing num tokens (num_tokens=28 avail_mem=27.16 GB):  84%|████████▍ | 49/58 [00:04<00:00, 32.95it/s]Capturing num tokens (num_tokens=24 avail_mem=27.15 GB):  84%|████████▍ | 49/58 [00:04<00:00, 32.95it/s]Capturing num tokens (num_tokens=24 avail_mem=27.15 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.11it/s]Capturing num tokens (num_tokens=20 avail_mem=27.15 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.11it/s]Capturing num tokens (num_tokens=16 avail_mem=27.15 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.11it/s]Capturing num tokens (num_tokens=12 avail_mem=27.14 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.11it/s]Capturing num tokens (num_tokens=8 avail_mem=27.14 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.11it/s] Capturing num tokens (num_tokens=8 avail_mem=27.14 GB):  98%|█████████▊| 57/58 [00:04<00:00, 34.89it/s]Capturing num tokens (num_tokens=4 avail_mem=27.14 GB):  98%|█████████▊| 57/58 [00:04<00:00, 34.89it/s]Capturing num tokens (num_tokens=4 avail_mem=27.14 GB): 100%|██████████| 58/58 [00:04<00:00, 12.25it/s]


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
    
    Generated text: Alright, the user asked for the information and population of the capital of France in JSON format. So, the first thing I need to figure out is who the capital is. I remember that Paris is the capital of France, so that's straightforward.
    
    Next, I have to find the population of Paris. I'm pretty sure it's a large city, maybe over 3 million people. Let me double-check that. Yes, according to recent data, the population is around 3,500,000. I should note that the number might change a bit over time.
    
    Now, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, and it's a way to structure data. I'll need to create a JSON object that includes the key details. The main points are the name of the city, its status as the capital, and the population figure.
    
    I should make sure the JSON syntax is correct. That means using quotation marks around the keys and string values, and braces for the object. Also, the numbers shouldn't have quotes around them unless they're strings. So the population will be an integer without quotes.
    
    Putting it all together, the JSON should look something like this: a key "capital" with the name "Paris", a key "capital_status" set to "Yes" since it's indeed the capital, and a key "population" with the number 3500000. 
    
    I think that covers everything the user asked for. They probably needed this data for a project, report, or maybe just general knowledge. Either way, providing the information in a clear and structured format like JSON should be helpful for their use.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "capital": "Paris",
      "capital_status": "Yes",
      "population": 3500000
    }
    ```



```python
llm.shutdown()
```
