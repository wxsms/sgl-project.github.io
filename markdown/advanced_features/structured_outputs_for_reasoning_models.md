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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.27s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.16s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.18s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:16,  5.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:16,  5.56s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:29,  2.67s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:29,  2.67s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:32,  1.69s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:32,  1.69s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:03,  1.18s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:03,  1.18s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:46,  1.14it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:46,  1.14it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:28,  1.78it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:28,  1.78it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:23,  2.13it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:23,  2.13it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:19,  2.53it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:19,  2.53it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:16,  2.96it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:16,  2.96it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:13,  3.38it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:13,  3.38it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:12,  3.79it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:12,  3.79it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:10,  4.28it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:10,  4.28it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:09,  4.77it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:09,  4.77it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:08,  5.35it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:08,  5.35it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:07,  5.87it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:07,  5.87it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:06,  6.58it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:06,  6.58it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:09<00:06,  6.58it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:10<00:04,  8.03it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:10<00:04,  8.03it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:10<00:04,  8.03it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:10<00:03,  9.62it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:10<00:03,  9.62it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:10<00:03,  9.62it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:10<00:03, 11.54it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:10<00:03, 11.54it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:10<00:03, 11.54it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:10<00:02, 13.45it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:10<00:02, 13.45it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:10<00:02, 13.45it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:10<00:02, 13.45it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:10<00:01, 16.16it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:10<00:01, 16.16it/s]

    Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:10<00:01, 16.16it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:10<00:01, 16.16it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:10<00:01, 19.23it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:10<00:01, 19.23it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:10<00:01, 19.23it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:10<00:01, 19.23it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:10<00:01, 19.23it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:10<00:01, 22.84it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:10<00:01, 22.84it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:10<00:01, 22.84it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:10<00:01, 22.84it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:10<00:01, 22.84it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:10<00:01, 22.84it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:10<00:00, 27.95it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:10<00:00, 27.95it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:10<00:00, 27.95it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:10<00:00, 27.95it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:11<00:00, 27.95it/s]

    Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:11<00:00, 30.84it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:11<00:00, 30.84it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:11<00:00, 30.84it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:11<00:00, 30.84it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:11<00:00, 30.84it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:11<00:00, 30.84it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:11<00:00, 35.31it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:11<00:00, 35.31it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:11<00:00, 35.31it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:11<00:00, 35.31it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:11<00:00, 35.31it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:11<00:00, 35.31it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:11<00:00, 35.31it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:11<00:00, 40.80it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:11<00:00, 40.80it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:11<00:00, 40.80it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:11<00:00, 40.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  5.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=44.28 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=44.28 GB):   2%|▏         | 1/58 [00:00<00:32,  1.73it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.22 GB):   2%|▏         | 1/58 [00:00<00:32,  1.73it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.22 GB):   3%|▎         | 2/58 [00:01<00:31,  1.79it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.23 GB):   3%|▎         | 2/58 [00:01<00:31,  1.79it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.23 GB):   5%|▌         | 3/58 [00:01<00:29,  1.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.29 GB):   5%|▌         | 3/58 [00:01<00:29,  1.87it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.29 GB):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.36 GB):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.36 GB):   9%|▊         | 5/58 [00:02<00:25,  2.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.43 GB):   9%|▊         | 5/58 [00:02<00:25,  2.10it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.43 GB):  10%|█         | 6/58 [00:02<00:22,  2.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.49 GB):  10%|█         | 6/58 [00:02<00:22,  2.28it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.49 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.46it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.55 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.46it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.55 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.62 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.70it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.62 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.69 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.95it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.69 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.18it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.71 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.18it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.71 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.74 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.31it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=43.74 GB):  21%|██        | 12/58 [00:04<00:12,  3.55it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.77 GB):  21%|██        | 12/58 [00:04<00:12,  3.55it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.77 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.80 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.90it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.80 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.55it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.93 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.55it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.93 GB):  26%|██▌       | 15/58 [00:05<00:08,  4.87it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.19 GB):  26%|██▌       | 15/58 [00:05<00:08,  4.87it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=44.19 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.18 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.18 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.18 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.82it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=44.18 GB):  31%|███       | 18/58 [00:05<00:06,  6.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.17 GB):  31%|███       | 18/58 [00:05<00:06,  6.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.17 GB):  33%|███▎      | 19/58 [00:05<00:06,  6.50it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.17 GB):  33%|███▎      | 19/58 [00:05<00:06,  6.50it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=44.17 GB):  34%|███▍      | 20/58 [00:05<00:05,  7.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.16 GB):  34%|███▍      | 20/58 [00:05<00:05,  7.01it/s]Capturing num tokens (num_tokens=960 avail_mem=44.14 GB):  34%|███▍      | 20/58 [00:05<00:05,  7.01it/s] Capturing num tokens (num_tokens=960 avail_mem=44.14 GB):  38%|███▊      | 22/58 [00:05<00:04,  8.73it/s]Capturing num tokens (num_tokens=896 avail_mem=44.13 GB):  38%|███▊      | 22/58 [00:05<00:04,  8.73it/s]

    Capturing num tokens (num_tokens=832 avail_mem=44.13 GB):  38%|███▊      | 22/58 [00:05<00:04,  8.73it/s]Capturing num tokens (num_tokens=832 avail_mem=44.13 GB):  41%|████▏     | 24/58 [00:06<00:03, 10.37it/s]Capturing num tokens (num_tokens=768 avail_mem=44.12 GB):  41%|████▏     | 24/58 [00:06<00:03, 10.37it/s]Capturing num tokens (num_tokens=704 avail_mem=44.11 GB):  41%|████▏     | 24/58 [00:06<00:03, 10.37it/s]Capturing num tokens (num_tokens=704 avail_mem=44.11 GB):  45%|████▍     | 26/58 [00:06<00:02, 11.96it/s]Capturing num tokens (num_tokens=640 avail_mem=44.11 GB):  45%|████▍     | 26/58 [00:06<00:02, 11.96it/s]

    Capturing num tokens (num_tokens=576 avail_mem=44.10 GB):  45%|████▍     | 26/58 [00:06<00:02, 11.96it/s]Capturing num tokens (num_tokens=576 avail_mem=44.10 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.37it/s]Capturing num tokens (num_tokens=512 avail_mem=44.09 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.37it/s]Capturing num tokens (num_tokens=480 avail_mem=44.08 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.37it/s]Capturing num tokens (num_tokens=480 avail_mem=44.08 GB):  52%|█████▏    | 30/58 [00:06<00:01, 14.88it/s]Capturing num tokens (num_tokens=448 avail_mem=44.08 GB):  52%|█████▏    | 30/58 [00:06<00:01, 14.88it/s]

    Capturing num tokens (num_tokens=416 avail_mem=44.07 GB):  52%|█████▏    | 30/58 [00:06<00:01, 14.88it/s]Capturing num tokens (num_tokens=416 avail_mem=44.07 GB):  55%|█████▌    | 32/58 [00:06<00:01, 15.79it/s]Capturing num tokens (num_tokens=384 avail_mem=44.07 GB):  55%|█████▌    | 32/58 [00:06<00:01, 15.79it/s]Capturing num tokens (num_tokens=352 avail_mem=44.05 GB):  55%|█████▌    | 32/58 [00:06<00:01, 15.79it/s]Capturing num tokens (num_tokens=320 avail_mem=44.04 GB):  55%|█████▌    | 32/58 [00:06<00:01, 15.79it/s]Capturing num tokens (num_tokens=320 avail_mem=44.04 GB):  60%|██████    | 35/58 [00:06<00:01, 18.02it/s]Capturing num tokens (num_tokens=288 avail_mem=44.05 GB):  60%|██████    | 35/58 [00:06<00:01, 18.02it/s]

    Capturing num tokens (num_tokens=256 avail_mem=44.05 GB):  60%|██████    | 35/58 [00:06<00:01, 18.02it/s]Capturing num tokens (num_tokens=240 avail_mem=44.04 GB):  60%|██████    | 35/58 [00:06<00:01, 18.02it/s]Capturing num tokens (num_tokens=240 avail_mem=44.04 GB):  66%|██████▌   | 38/58 [00:06<00:01, 19.98it/s]Capturing num tokens (num_tokens=224 avail_mem=44.03 GB):  66%|██████▌   | 38/58 [00:06<00:01, 19.98it/s]Capturing num tokens (num_tokens=208 avail_mem=44.02 GB):  66%|██████▌   | 38/58 [00:06<00:01, 19.98it/s]Capturing num tokens (num_tokens=192 avail_mem=44.02 GB):  66%|██████▌   | 38/58 [00:06<00:01, 19.98it/s]Capturing num tokens (num_tokens=192 avail_mem=44.02 GB):  71%|███████   | 41/58 [00:06<00:00, 21.61it/s]Capturing num tokens (num_tokens=176 avail_mem=44.01 GB):  71%|███████   | 41/58 [00:06<00:00, 21.61it/s]

    Capturing num tokens (num_tokens=160 avail_mem=44.00 GB):  71%|███████   | 41/58 [00:06<00:00, 21.61it/s]Capturing num tokens (num_tokens=144 avail_mem=43.99 GB):  71%|███████   | 41/58 [00:06<00:00, 21.61it/s]Capturing num tokens (num_tokens=144 avail_mem=43.99 GB):  76%|███████▌  | 44/58 [00:06<00:00, 22.87it/s]Capturing num tokens (num_tokens=128 avail_mem=43.99 GB):  76%|███████▌  | 44/58 [00:06<00:00, 22.87it/s]Capturing num tokens (num_tokens=112 avail_mem=43.99 GB):  76%|███████▌  | 44/58 [00:07<00:00, 22.87it/s]Capturing num tokens (num_tokens=96 avail_mem=43.98 GB):  76%|███████▌  | 44/58 [00:07<00:00, 22.87it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=43.98 GB):  81%|████████  | 47/58 [00:07<00:00, 23.24it/s]Capturing num tokens (num_tokens=80 avail_mem=43.97 GB):  81%|████████  | 47/58 [00:07<00:00, 23.24it/s]Capturing num tokens (num_tokens=64 avail_mem=43.96 GB):  81%|████████  | 47/58 [00:07<00:00, 23.24it/s]Capturing num tokens (num_tokens=48 avail_mem=43.96 GB):  81%|████████  | 47/58 [00:07<00:00, 23.24it/s]Capturing num tokens (num_tokens=48 avail_mem=43.96 GB):  86%|████████▌ | 50/58 [00:07<00:00, 24.38it/s]Capturing num tokens (num_tokens=32 avail_mem=43.95 GB):  86%|████████▌ | 50/58 [00:07<00:00, 24.38it/s]Capturing num tokens (num_tokens=28 avail_mem=43.95 GB):  86%|████████▌ | 50/58 [00:07<00:00, 24.38it/s]Capturing num tokens (num_tokens=24 avail_mem=43.94 GB):  86%|████████▌ | 50/58 [00:07<00:00, 24.38it/s]

    Capturing num tokens (num_tokens=24 avail_mem=43.94 GB):  91%|█████████▏| 53/58 [00:07<00:00, 25.79it/s]Capturing num tokens (num_tokens=20 avail_mem=43.93 GB):  91%|█████████▏| 53/58 [00:07<00:00, 25.79it/s]Capturing num tokens (num_tokens=16 avail_mem=43.93 GB):  91%|█████████▏| 53/58 [00:07<00:00, 25.79it/s]Capturing num tokens (num_tokens=12 avail_mem=43.92 GB):  91%|█████████▏| 53/58 [00:07<00:00, 25.79it/s]Capturing num tokens (num_tokens=12 avail_mem=43.92 GB):  97%|█████████▋| 56/58 [00:07<00:00, 26.62it/s]Capturing num tokens (num_tokens=8 avail_mem=43.91 GB):  97%|█████████▋| 56/58 [00:07<00:00, 26.62it/s] Capturing num tokens (num_tokens=4 avail_mem=43.91 GB):  97%|█████████▋| 56/58 [00:07<00:00, 26.62it/s]Capturing num tokens (num_tokens=4 avail_mem=43.91 GB): 100%|██████████| 58/58 [00:07<00:00,  7.76it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, the user is in New York and wants the current date and time, along with the weather. Let me figure out how to approach this.<br><br>First, I need to determine which functions to use. There's 'get_current_date' for the date and time, and 'get_current_weather' for the weather. Both require specific parameters.<br><br>For the date and time, the function needs the timezone. Since the user is in New York, I'll use 'America/New_York' as the timezone parameter.<br><br>Next, for the weather, I need the city, state, and unit. The city is New York, the state is NY, and the unit should be Fahrenheit since the user didn't specify otherwise.<br><br>I should structure the response using the specified format, calling each function with the appropriate parameters. I'll make sure to include the sources from the functions' documentation.<br><br>Putting it all together, I'll send two function calls: one for the date and time with the timezone, and another for the weather with the necessary details. This should provide the user with both pieces of information they're asking for.<br><br><br>content: <br><br><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'ae9cd7696f2e49f29ad2ae6d2043e0a4', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.258485889062285, 'response_sent_to_client_ts': 1780313941.5349505}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': 'd3afe671be7b408a888f788d5ef434b4', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 22.08604156691581, 'response_sent_to_client_ts': 1780313963.6299632}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'e3452f2aef0947d9a2d9619dc154ffb0', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.22842664550989866, 'response_sent_to_client_ts': 1780313963.8878422}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'eec19f6776004dfa9a307a181c04404f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.22833948582410812, 'response_sent_to_client_ts': 1780313963.887858}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '43281919862c4706a05f8a4b6ff3840c', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.22829117253422737, 'response_sent_to_client_ts': 1780313963.887863}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '8578009a9b3144d4a7bb00fddefc2b93', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 21.15519357752055, 'response_sent_to_client_ts': 1780313985.0520751}}


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


<strong style='color: #00008B;'>{'text': 'Alright, the user asked for the information and population of the capital of France in JSON format. Well, I know that the capital is Paris, so that\'s straightforward. Now, they might need precise details, so I should focus on accurate and up-to-date information.\n\n Population can be a bit tricky because it changes yearly. I\'ll estimate based on the latest data I have. I think around 2.2 million is right, but I should double-check the exact figure to make sure it\'s correct.\n\nThey want it in JSON, so I\'ll structure it with "name" as "Paris", "administrative Region" as " Île-de-France", and "Population" with both the estimated number and wording in French for clarity. That way, it\'s clear and user-friendly.\n\nAlso, since they mentioned "the information and population", including the region is helpful for context. I need to make sure all the fields are correctly labeled and the JSON syntax is accurate.\n\nI should offer further assistance in case there\'s something else they need, keeping the response friendly and open-ended.\n</think>\n\nHere is the information and population of the capital of France (Paris) in JSON format:\n\n```json\n{\n  "name": "Paris",\n  "administrativeRegion": "Île-de-France",\n  "Population": {\n    "estimated": 2291000,\n    "text": "environ 2,291 000 habitants"\n  }\n}\n```\n\nThis JSON includes the estimated population of Paris, which is approximately 2.291 million people.', 'output_ids': [71486, 11, 279, 1196, 4588, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 8325, 11, 358, 1414, 429, 279, 6722, 374, 12095, 11, 773, 429, 594, 30339, 13, 4695, 11, 807, 2578, 1184, 23560, 3565, 11, 773, 358, 1265, 5244, 389, 13382, 323, 705, 4686, 18413, 1995, 382, 39529, 646, 387, 264, 2699, 33453, 1576, 432, 4344, 44270, 13, 358, 3278, 16045, 3118, 389, 279, 5535, 821, 358, 614, 13, 358, 1744, 2163, 220, 17, 13, 17, 3526, 374, 1290, 11, 714, 358, 1265, 1990, 15934, 279, 4734, 7071, 311, 1281, 2704, 432, 594, 4396, 382, 6865, 1366, 432, 304, 4718, 11, 773, 358, 3278, 5944, 432, 448, 330, 606, 1, 438, 330, 59604, 497, 330, 68849, 1388, 17152, 1, 438, 330, 59108, 273, 6810, 7276, 34106, 497, 323, 330, 53371, 1, 448, 2176, 279, 12943, 1372, 323, 60227, 304, 8585, 369, 31273, 13, 2938, 1616, 11, 432, 594, 2797, 323, 1196, 21896, 382, 13394, 11, 2474, 807, 9733, 330, 1782, 1995, 323, 7042, 497, 2670, 279, 5537, 374, 10950, 369, 2266, 13, 358, 1184, 311, 1281, 2704, 678, 279, 5043, 525, 12440, 29829, 323, 279, 4718, 19482, 374, 13382, 382, 40, 1265, 3010, 4623, 12994, 304, 1142, 1052, 594, 2494, 770, 807, 1184, 11, 10282, 279, 2033, 11657, 323, 1787, 83075, 624, 151649, 271, 8420, 374, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 320, 59604, 8, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 68849, 1388, 14091, 788, 330, 71807, 273, 6810, 7276, 34106, 756, 220, 330, 53371, 788, 341, 262, 330, 44229, 788, 220, 17, 17, 24, 16, 15, 15, 15, 345, 262, 330, 1318, 788, 330, 268, 2772, 220, 17, 11, 17, 24, 16, 220, 15, 15, 15, 14132, 1783, 698, 220, 456, 532, 13874, 19324, 1986, 4718, 5646, 279, 12943, 7042, 315, 12095, 11, 892, 374, 13187, 220, 17, 13, 17, 24, 16, 3526, 1251, 13, 151643], 'meta_info': {'id': '69aa8fecfba2433796f7d7bef600ba34', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 220, 'completion_tokens': 328, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.407170284539461, 'response_sent_to_client_ts': 1780313988.473273}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.26s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.13s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.15s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:55,  5.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:55,  5.18s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:06,  2.26s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:06,  2.26s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.15it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.19it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.19it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.85it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.85it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.59it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.59it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.45it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.45it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:09,  5.30it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:09,  5.30it/s]

    Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:06<00:09,  5.30it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:06,  7.05it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:06,  7.05it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:06<00:06,  7.05it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:05,  8.37it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:05,  8.37it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:06<00:05,  8.37it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:04,  9.82it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:04,  9.82it/s]

    Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:07<00:04,  9.82it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:07<00:03, 11.56it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:07<00:03, 11.56it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:07<00:03, 11.56it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:07<00:03, 11.56it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:07<00:02, 14.46it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:07<00:02, 14.46it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:07<00:02, 14.46it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:07<00:02, 14.46it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:07<00:02, 14.46it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:07<00:01, 19.40it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:07<00:01, 19.40it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:07<00:01, 19.40it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:07<00:01, 19.40it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:07<00:01, 19.40it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:07<00:01, 19.40it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:01, 24.94it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:01, 24.94it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:01, 24.94it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:01, 24.94it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:07<00:00, 25.50it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:07<00:00, 25.50it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:07<00:00, 25.50it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:07<00:00, 25.50it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:07<00:00, 25.50it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:07<00:00, 28.77it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:07<00:00, 28.77it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:07<00:00, 28.77it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:07<00:00, 28.77it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:07<00:00, 28.77it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:07<00:00, 31.67it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:07<00:00, 31.67it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:07<00:00, 31.67it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:07<00:00, 31.67it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:07<00:00, 31.67it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:07<00:00, 33.07it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:07<00:00, 33.07it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:07<00:00, 33.07it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:08<00:00, 33.07it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:08<00:00, 33.07it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 34.28it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 34.28it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 34.28it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 34.28it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 34.28it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:08<00:00, 34.28it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:08<00:00, 34.28it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:08<00:00, 34.28it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:08<00:00, 43.70it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:08<00:00, 43.70it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:08<00:00, 43.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.07it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=45.99 GB):   2%|▏         | 1/58 [00:00<00:17,  3.30it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.95 GB):   2%|▏         | 1/58 [00:00<00:17,  3.30it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=45.95 GB):   3%|▎         | 2/58 [00:00<00:17,  3.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.95 GB):   3%|▎         | 2/58 [00:00<00:17,  3.27it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=45.95 GB):   5%|▌         | 3/58 [00:00<00:16,  3.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=45.95 GB):   5%|▌         | 3/58 [00:00<00:16,  3.38it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=45.95 GB):   7%|▋         | 4/58 [00:01<00:14,  3.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.95 GB):   7%|▋         | 4/58 [00:01<00:14,  3.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=45.95 GB):   9%|▊         | 5/58 [00:01<00:13,  3.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.89 GB):   9%|▊         | 5/58 [00:01<00:13,  3.97it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=45.89 GB):  10%|█         | 6/58 [00:01<00:12,  4.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.89 GB):  10%|█         | 6/58 [00:01<00:12,  4.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.89 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.63it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.89 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.63it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=45.89 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.89 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.89 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.89 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.76it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=45.89 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.89 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.89 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.86it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.88 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.86it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=45.88 GB):  21%|██        | 12/58 [00:02<00:06,  7.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=45.88 GB):  21%|██        | 12/58 [00:02<00:06,  7.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.88 GB):  21%|██        | 12/58 [00:02<00:06,  7.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.88 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.88 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.67it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=45.88 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.88 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=45.87 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.87 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.87 GB):  31%|███       | 18/58 [00:02<00:03, 11.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.87 GB):  31%|███       | 18/58 [00:02<00:03, 11.48it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=45.87 GB):  31%|███       | 18/58 [00:02<00:03, 11.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=45.85 GB):  31%|███       | 18/58 [00:02<00:03, 11.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=45.85 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.42it/s]Capturing num tokens (num_tokens=960 avail_mem=45.85 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.42it/s] Capturing num tokens (num_tokens=896 avail_mem=45.85 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.42it/s]Capturing num tokens (num_tokens=832 avail_mem=45.00 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.42it/s]

    Capturing num tokens (num_tokens=832 avail_mem=45.00 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.61it/s]Capturing num tokens (num_tokens=768 avail_mem=39.89 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.61it/s]Capturing num tokens (num_tokens=704 avail_mem=38.48 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.61it/s]Capturing num tokens (num_tokens=640 avail_mem=38.47 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.61it/s]Capturing num tokens (num_tokens=640 avail_mem=38.47 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.23it/s]Capturing num tokens (num_tokens=576 avail_mem=38.47 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.23it/s]Capturing num tokens (num_tokens=512 avail_mem=38.46 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.23it/s]Capturing num tokens (num_tokens=480 avail_mem=38.46 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.23it/s]

    Capturing num tokens (num_tokens=448 avail_mem=38.46 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.23it/s]Capturing num tokens (num_tokens=448 avail_mem=38.46 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.63it/s]Capturing num tokens (num_tokens=416 avail_mem=38.46 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.63it/s]Capturing num tokens (num_tokens=384 avail_mem=38.45 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.63it/s]Capturing num tokens (num_tokens=352 avail_mem=38.45 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.63it/s]Capturing num tokens (num_tokens=320 avail_mem=38.44 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.63it/s]Capturing num tokens (num_tokens=320 avail_mem=38.44 GB):  60%|██████    | 35/58 [00:03<00:00, 25.72it/s]Capturing num tokens (num_tokens=288 avail_mem=38.45 GB):  60%|██████    | 35/58 [00:03<00:00, 25.72it/s]Capturing num tokens (num_tokens=256 avail_mem=38.44 GB):  60%|██████    | 35/58 [00:03<00:00, 25.72it/s]

    Capturing num tokens (num_tokens=240 avail_mem=38.44 GB):  60%|██████    | 35/58 [00:03<00:00, 25.72it/s]Capturing num tokens (num_tokens=224 avail_mem=38.44 GB):  60%|██████    | 35/58 [00:03<00:00, 25.72it/s]Capturing num tokens (num_tokens=224 avail_mem=38.44 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.48it/s]Capturing num tokens (num_tokens=208 avail_mem=38.43 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.48it/s]Capturing num tokens (num_tokens=192 avail_mem=38.43 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.48it/s]Capturing num tokens (num_tokens=176 avail_mem=38.42 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.48it/s]Capturing num tokens (num_tokens=160 avail_mem=38.42 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.48it/s]Capturing num tokens (num_tokens=160 avail_mem=38.42 GB):  74%|███████▍  | 43/58 [00:03<00:00, 30.79it/s]Capturing num tokens (num_tokens=144 avail_mem=38.42 GB):  74%|███████▍  | 43/58 [00:03<00:00, 30.79it/s]Capturing num tokens (num_tokens=128 avail_mem=38.42 GB):  74%|███████▍  | 43/58 [00:03<00:00, 30.79it/s]

    Capturing num tokens (num_tokens=112 avail_mem=38.42 GB):  74%|███████▍  | 43/58 [00:03<00:00, 30.79it/s]Capturing num tokens (num_tokens=96 avail_mem=38.41 GB):  74%|███████▍  | 43/58 [00:03<00:00, 30.79it/s] Capturing num tokens (num_tokens=96 avail_mem=38.41 GB):  81%|████████  | 47/58 [00:03<00:00, 32.05it/s]Capturing num tokens (num_tokens=80 avail_mem=38.41 GB):  81%|████████  | 47/58 [00:03<00:00, 32.05it/s]Capturing num tokens (num_tokens=64 avail_mem=38.40 GB):  81%|████████  | 47/58 [00:03<00:00, 32.05it/s]Capturing num tokens (num_tokens=48 avail_mem=38.40 GB):  81%|████████  | 47/58 [00:03<00:00, 32.05it/s]Capturing num tokens (num_tokens=32 avail_mem=38.40 GB):  81%|████████  | 47/58 [00:03<00:00, 32.05it/s]Capturing num tokens (num_tokens=32 avail_mem=38.40 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.78it/s]Capturing num tokens (num_tokens=28 avail_mem=38.40 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.78it/s]Capturing num tokens (num_tokens=24 avail_mem=38.39 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.78it/s]

    Capturing num tokens (num_tokens=20 avail_mem=38.39 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.78it/s]Capturing num tokens (num_tokens=16 avail_mem=38.38 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.78it/s]Capturing num tokens (num_tokens=16 avail_mem=38.38 GB):  95%|█████████▍| 55/58 [00:04<00:00, 34.55it/s]Capturing num tokens (num_tokens=12 avail_mem=38.38 GB):  95%|█████████▍| 55/58 [00:04<00:00, 34.55it/s]Capturing num tokens (num_tokens=8 avail_mem=38.38 GB):  95%|█████████▍| 55/58 [00:04<00:00, 34.55it/s] Capturing num tokens (num_tokens=4 avail_mem=38.37 GB):  95%|█████████▍| 55/58 [00:04<00:00, 34.55it/s]Capturing num tokens (num_tokens=4 avail_mem=38.37 GB): 100%|██████████| 58/58 [00:04<00:00, 14.18it/s]


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
    
    Generated text: Alright, so I need to figure out the information and population of the capital of France. First, I should identify what the capital of France is. I remember that Paris is the capital of France, but I'm not 100% sure. Let me think... yes, Paris is definitely the capital. So, the capital's name is Paris.
    
    Next, I need to find out the population of Paris. I think it's a large city, but I'm not sure of the exact number. I know that France's population is around 40 million, but Paris is the largest city there. I've heard it's over 20 million, maybe around 21 or 22 million? I'm not certain, though.
    
    Wait, I should check if there's a more precise figure. I recall that in recent years, populations have grown, so maybe it's a bit higher now. I think it's about 21 million, but I'm not sure if it's exactly 21. Maybe it's 20.5 million? I should try to get a more accurate number.
    
    I also need to make sure I'm providing this information in JSON format as the user requested. JSON is a data interchange format, so I'll structure it with a key for "capital" and another for "population." I'll include the population as an integer, probably with commas for clarity, like 21,000,000.
    
    Putting it all together, the JSON should look something like:
    {
      "capital": "Paris",
      "population": 21000000
    }
    
    I think that covers the basics. Paris is the capital, and the population is roughly 21 million. I should double-check these details to make sure they're correct, but given my current knowledge, this should be accurate enough.
    </think>
    
    ```json
    {
      "capital": "Paris",
      "population": 21000000
    }
    ```



```python
llm.shutdown()
```
