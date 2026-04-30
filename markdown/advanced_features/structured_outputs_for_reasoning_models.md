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


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.49s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.52s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.51s/it]


    2026-04-30 22:42:30,458 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 22:42:30] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:17,  5.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:17,  5.58s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:23,  2.57s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:23,  2.57s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:27,  1.59s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:27,  1.59s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<01:02,  1.16s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<01:02,  1.16s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:49,  1.08it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:49,  1.08it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:37,  1.38it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:37,  1.38it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:29,  1.72it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:29,  1.72it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:24,  2.07it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:24,  2.07it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:19,  2.47it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:19,  2.47it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:16,  2.90it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:16,  2.90it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:14,  3.32it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:14,  3.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:12,  3.77it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:12,  3.77it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:11,  4.07it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:11,  4.07it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:10,  4.06it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:10,  4.06it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:10,  4.25it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:10,  4.25it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:10<00:09,  4.55it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:10<00:09,  4.55it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:10<00:08,  4.86it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:10<00:08,  4.86it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:10<00:07,  5.26it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:10<00:07,  5.26it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:10<00:06,  5.70it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:10<00:06,  5.70it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:10<00:06,  6.26it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:10<00:06,  6.26it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:10<00:06,  6.26it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:10<00:04,  7.86it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:10<00:04,  7.86it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:10<00:04,  7.86it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:10<00:03,  9.01it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:10<00:03,  9.01it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:11<00:03,  9.01it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:11<00:03, 10.17it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:11<00:03, 10.17it/s]

    Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:11<00:03, 10.17it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:11<00:02, 11.68it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:11<00:02, 11.68it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:11<00:02, 11.68it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:11<00:02, 13.15it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:11<00:02, 13.15it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:11<00:02, 13.15it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:11<00:01, 13.77it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:11<00:01, 13.77it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:11<00:01, 13.77it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:11<00:01, 15.05it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:11<00:01, 15.05it/s]

    Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:11<00:01, 15.05it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:11<00:01, 15.05it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:11<00:01, 16.91it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:11<00:01, 16.91it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:11<00:01, 16.91it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:11<00:01, 16.91it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:11<00:00, 19.05it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:11<00:00, 19.05it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:11<00:00, 19.05it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:11<00:00, 19.05it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:12<00:00, 20.02it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:12<00:00, 20.02it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:12<00:00, 20.02it/s]

    Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:12<00:00, 20.02it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:12<00:00, 21.15it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:12<00:00, 21.15it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:12<00:00, 21.15it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:12<00:00, 21.15it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:12<00:00, 21.60it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:12<00:00, 21.60it/s]

    Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:12<00:00, 21.60it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:12<00:00, 21.60it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:12<00:00, 22.64it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:12<00:00, 22.64it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:12<00:00, 22.64it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:12<00:00, 22.64it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:12<00:00, 24.24it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:12<00:00, 24.24it/s]

    Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:12<00:00, 24.24it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:12<00:00, 24.24it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:12<00:00,  4.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.86 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.86 GB):   2%|▏         | 1/58 [00:00<00:54,  1.05it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.80 GB):   2%|▏         | 1/58 [00:00<00:54,  1.05it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.80 GB):   3%|▎         | 2/58 [00:01<00:48,  1.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=24.79 GB):   3%|▎         | 2/58 [00:01<00:48,  1.17it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=24.79 GB):   5%|▌         | 3/58 [00:02<00:43,  1.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=24.79 GB):   5%|▌         | 3/58 [00:02<00:43,  1.27it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=24.79 GB):   7%|▋         | 4/58 [00:03<00:37,  1.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=24.79 GB):   7%|▋         | 4/58 [00:03<00:37,  1.43it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=24.79 GB):   9%|▊         | 5/58 [00:03<00:33,  1.58it/s]Capturing num tokens (num_tokens=5632 avail_mem=24.79 GB):   9%|▊         | 5/58 [00:03<00:33,  1.58it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=24.79 GB):  10%|█         | 6/58 [00:03<00:29,  1.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=24.78 GB):  10%|█         | 6/58 [00:03<00:29,  1.77it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=24.78 GB):  12%|█▏        | 7/58 [00:04<00:25,  2.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=24.79 GB):  12%|█▏        | 7/58 [00:04<00:25,  2.03it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=24.79 GB):  14%|█▍        | 8/58 [00:04<00:23,  2.16it/s]Capturing num tokens (num_tokens=4096 avail_mem=23.82 GB):  14%|█▍        | 8/58 [00:04<00:23,  2.16it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=23.82 GB):  16%|█▌        | 9/58 [00:05<00:22,  2.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=24.76 GB):  16%|█▌        | 9/58 [00:05<00:22,  2.20it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=24.76 GB):  17%|█▋        | 10/58 [00:05<00:20,  2.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=23.93 GB):  17%|█▋        | 10/58 [00:05<00:20,  2.33it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=23.93 GB):  19%|█▉        | 11/58 [00:05<00:19,  2.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=23.93 GB):  19%|█▉        | 11/58 [00:05<00:19,  2.38it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=23.93 GB):  21%|██        | 12/58 [00:06<00:17,  2.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=24.74 GB):  21%|██        | 12/58 [00:06<00:17,  2.58it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=24.74 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=23.98 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.70it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=23.98 GB):  24%|██▍       | 14/58 [00:06<00:15,  2.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=24.74 GB):  24%|██▍       | 14/58 [00:06<00:15,  2.86it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=24.74 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=24.02 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.07it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=24.02 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.73 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.25it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=24.73 GB):  29%|██▉       | 17/58 [00:07<00:11,  3.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=24.07 GB):  29%|██▉       | 17/58 [00:07<00:11,  3.60it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=24.07 GB):  31%|███       | 18/58 [00:07<00:10,  3.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=24.06 GB):  31%|███       | 18/58 [00:07<00:10,  3.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=24.06 GB):  33%|███▎      | 19/58 [00:08<00:09,  4.18it/s]Capturing num tokens (num_tokens=1280 avail_mem=24.11 GB):  33%|███▎      | 19/58 [00:08<00:09,  4.18it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=24.11 GB):  34%|███▍      | 20/58 [00:08<00:09,  4.04it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.69 GB):  34%|███▍      | 20/58 [00:08<00:09,  4.04it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.69 GB):  36%|███▌      | 21/58 [00:08<00:08,  4.54it/s]Capturing num tokens (num_tokens=960 avail_mem=24.14 GB):  36%|███▌      | 21/58 [00:08<00:08,  4.54it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=24.14 GB):  38%|███▊      | 22/58 [00:08<00:07,  5.07it/s]Capturing num tokens (num_tokens=896 avail_mem=24.68 GB):  38%|███▊      | 22/58 [00:08<00:07,  5.07it/s]Capturing num tokens (num_tokens=896 avail_mem=24.68 GB):  40%|███▉      | 23/58 [00:08<00:06,  5.63it/s]Capturing num tokens (num_tokens=832 avail_mem=24.18 GB):  40%|███▉      | 23/58 [00:08<00:06,  5.63it/s]

    Capturing num tokens (num_tokens=832 avail_mem=24.18 GB):  41%|████▏     | 24/58 [00:08<00:05,  5.89it/s]Capturing num tokens (num_tokens=768 avail_mem=24.66 GB):  41%|████▏     | 24/58 [00:08<00:05,  5.89it/s]Capturing num tokens (num_tokens=704 avail_mem=24.21 GB):  41%|████▏     | 24/58 [00:08<00:05,  5.89it/s]

    Capturing num tokens (num_tokens=704 avail_mem=24.21 GB):  45%|████▍     | 26/58 [00:09<00:04,  6.73it/s]Capturing num tokens (num_tokens=640 avail_mem=24.21 GB):  45%|████▍     | 26/58 [00:09<00:04,  6.73it/s]Capturing num tokens (num_tokens=576 avail_mem=24.64 GB):  45%|████▍     | 26/58 [00:09<00:04,  6.73it/s]

    Capturing num tokens (num_tokens=576 avail_mem=24.64 GB):  48%|████▊     | 28/58 [00:09<00:04,  7.47it/s]Capturing num tokens (num_tokens=512 avail_mem=24.21 GB):  48%|████▊     | 28/58 [00:09<00:04,  7.47it/s]Capturing num tokens (num_tokens=480 avail_mem=24.62 GB):  48%|████▊     | 28/58 [00:09<00:04,  7.47it/s]

    Capturing num tokens (num_tokens=480 avail_mem=24.62 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.17it/s]Capturing num tokens (num_tokens=448 avail_mem=24.23 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.17it/s]Capturing num tokens (num_tokens=448 avail_mem=24.23 GB):  53%|█████▎    | 31/58 [00:09<00:03,  8.39it/s]Capturing num tokens (num_tokens=416 avail_mem=24.61 GB):  53%|█████▎    | 31/58 [00:09<00:03,  8.39it/s]

    Capturing num tokens (num_tokens=416 avail_mem=24.61 GB):  55%|█████▌    | 32/58 [00:09<00:03,  8.57it/s]Capturing num tokens (num_tokens=384 avail_mem=24.24 GB):  55%|█████▌    | 32/58 [00:09<00:03,  8.57it/s]Capturing num tokens (num_tokens=352 avail_mem=24.58 GB):  55%|█████▌    | 32/58 [00:09<00:03,  8.57it/s]

    Capturing num tokens (num_tokens=352 avail_mem=24.58 GB):  59%|█████▊    | 34/58 [00:09<00:02,  8.89it/s]Capturing num tokens (num_tokens=320 avail_mem=24.25 GB):  59%|█████▊    | 34/58 [00:09<00:02,  8.89it/s]Capturing num tokens (num_tokens=288 avail_mem=24.58 GB):  59%|█████▊    | 34/58 [00:10<00:02,  8.89it/s]Capturing num tokens (num_tokens=288 avail_mem=24.58 GB):  62%|██████▏   | 36/58 [00:10<00:02,  9.56it/s]Capturing num tokens (num_tokens=256 avail_mem=24.57 GB):  62%|██████▏   | 36/58 [00:10<00:02,  9.56it/s]

    Capturing num tokens (num_tokens=240 avail_mem=24.29 GB):  62%|██████▏   | 36/58 [00:10<00:02,  9.56it/s]

    Capturing num tokens (num_tokens=240 avail_mem=24.29 GB):  66%|██████▌   | 38/58 [00:10<00:02,  8.64it/s]Capturing num tokens (num_tokens=224 avail_mem=24.31 GB):  66%|██████▌   | 38/58 [00:10<00:02,  8.64it/s]Capturing num tokens (num_tokens=208 avail_mem=24.55 GB):  66%|██████▌   | 38/58 [00:10<00:02,  8.64it/s]Capturing num tokens (num_tokens=208 avail_mem=24.55 GB):  69%|██████▉   | 40/58 [00:10<00:01,  9.72it/s]Capturing num tokens (num_tokens=192 avail_mem=24.32 GB):  69%|██████▉   | 40/58 [00:10<00:01,  9.72it/s]Capturing num tokens (num_tokens=176 avail_mem=24.53 GB):  69%|██████▉   | 40/58 [00:10<00:01,  9.72it/s]

    Capturing num tokens (num_tokens=176 avail_mem=24.53 GB):  72%|███████▏  | 42/58 [00:10<00:01, 11.18it/s]Capturing num tokens (num_tokens=160 avail_mem=24.52 GB):  72%|███████▏  | 42/58 [00:10<00:01, 11.18it/s]Capturing num tokens (num_tokens=144 avail_mem=24.34 GB):  72%|███████▏  | 42/58 [00:10<00:01, 11.18it/s]Capturing num tokens (num_tokens=144 avail_mem=24.34 GB):  76%|███████▌  | 44/58 [00:10<00:01, 12.48it/s]Capturing num tokens (num_tokens=128 avail_mem=24.50 GB):  76%|███████▌  | 44/58 [00:10<00:01, 12.48it/s]Capturing num tokens (num_tokens=112 avail_mem=24.51 GB):  76%|███████▌  | 44/58 [00:10<00:01, 12.48it/s]

    Capturing num tokens (num_tokens=112 avail_mem=24.51 GB):  79%|███████▉  | 46/58 [00:10<00:00, 13.43it/s]Capturing num tokens (num_tokens=96 avail_mem=24.48 GB):  79%|███████▉  | 46/58 [00:10<00:00, 13.43it/s] Capturing num tokens (num_tokens=80 avail_mem=24.36 GB):  79%|███████▉  | 46/58 [00:10<00:00, 13.43it/s]Capturing num tokens (num_tokens=64 avail_mem=24.46 GB):  79%|███████▉  | 46/58 [00:11<00:00, 13.43it/s]Capturing num tokens (num_tokens=64 avail_mem=24.46 GB):  84%|████████▍ | 49/58 [00:11<00:00, 15.28it/s]Capturing num tokens (num_tokens=48 avail_mem=24.45 GB):  84%|████████▍ | 49/58 [00:11<00:00, 15.28it/s]

    Capturing num tokens (num_tokens=32 avail_mem=24.44 GB):  84%|████████▍ | 49/58 [00:11<00:00, 15.28it/s]Capturing num tokens (num_tokens=32 avail_mem=24.44 GB):  88%|████████▊ | 51/58 [00:11<00:00, 15.81it/s]Capturing num tokens (num_tokens=28 avail_mem=24.44 GB):  88%|████████▊ | 51/58 [00:11<00:00, 15.81it/s]Capturing num tokens (num_tokens=24 avail_mem=24.40 GB):  88%|████████▊ | 51/58 [00:11<00:00, 15.81it/s]Capturing num tokens (num_tokens=20 avail_mem=24.41 GB):  88%|████████▊ | 51/58 [00:11<00:00, 15.81it/s]Capturing num tokens (num_tokens=20 avail_mem=24.41 GB):  93%|█████████▎| 54/58 [00:11<00:00, 17.61it/s]Capturing num tokens (num_tokens=16 avail_mem=24.40 GB):  93%|█████████▎| 54/58 [00:11<00:00, 17.61it/s]

    Capturing num tokens (num_tokens=12 avail_mem=24.39 GB):  93%|█████████▎| 54/58 [00:11<00:00, 17.61it/s]Capturing num tokens (num_tokens=12 avail_mem=24.39 GB):  97%|█████████▋| 56/58 [00:11<00:00, 18.10it/s]Capturing num tokens (num_tokens=8 avail_mem=24.38 GB):  97%|█████████▋| 56/58 [00:11<00:00, 18.10it/s] Capturing num tokens (num_tokens=4 avail_mem=24.33 GB):  97%|█████████▋| 56/58 [00:11<00:00, 18.10it/s]Capturing num tokens (num_tokens=4 avail_mem=24.33 GB): 100%|██████████| 58/58 [00:11<00:00,  5.02it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should check if that's correct.<br><br>Wait, I think the population might have changed a bit over the years. I recall reading somewhere that Paris has grown a lot, especially with the influx of people moving there for work. But I'm not sure if it's exactly 21 million or maybe a bit more. I should look up the latest data to confirm.<br><br>I also wonder if the population figure includes just the city proper or the entire metropolitan area. Sometimes, people talk about the metro area, which can be much larger. But the question specifically asks for the population of the capital, so I think it refers to the city limits. Still, I should make sure.<br><br>Another thing to consider is that population figures can vary depending on the source. Some might cite estimates from government agencies, while others might use more recent data from censuses or surveys. I should find a reliable source to get the most accurate number.<br><br>I think the population of Paris is around 21 million, but I'm not 100% sure. Maybe I should think about other major cities in France to compare. For example, Lyon is another big city, but it's much smaller. I believe its population is around 1.2 million. That gives me a sense that Paris is significantly larger.<br><br>Also, considering the economic activities in Paris, like the fashion industry and the entertainment sector, it makes sense that it's the capital and has a large population. The city hosts a lot of events, conventions, and businesses, which would attract a diverse population.<br><br>I should also think about the historical growth of Paris. It's been a major city for centuries, so its population has been increasing steadily. I think it's safe to say that it's over 20 million, but I'm still not certain about the exact number.<br><br>In summary, I'm pretty confident that the capital of France is Paris, and its population is around 21 million. However, to be precise, I should look up the latest statistics to confirm the exact figure.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21538000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should check if that's correct.<br><br>Wait, I think the population might have changed a bit over the years. I recall reading somewhere that Paris has grown a lot, especially with the influx of people moving there for work. But I'm not sure if it's exactly 21 million or maybe a bit more. I should look up the latest data to confirm.<br><br>I also wonder if the population figure includes just the city proper or the entire metropolitan area. Sometimes, people talk about the metro area, which can be much larger. But the question specifically asks for the population of the capital, so I think it refers to the city limits. Still, I should make sure.<br><br>Another thing to consider is that population figures can vary depending on the source. Some might cite estimates from government agencies, while others might use more recent data from censuses or surveys. I should find a reliable source to get the most accurate number.<br><br>I think the population of Paris is around 21 million, but I'm not 100% sure. Maybe I should think about other major cities in France to compare. For example, Lyon is another big city, but it's much smaller. I believe its population is around 1.2 million. That gives me a sense that Paris is significantly larger.<br><br>Also, considering the economic activities in Paris, like the fashion industry and the entertainment sector, it makes sense that it's the capital and has a large population. The city hosts a lot of events, conventions, and businesses, which would attract a diverse population.<br><br>I should also think about the historical growth of Paris. It's been a major city for centuries, so its population has been increasing steadily. I think it's safe to say that it's over 20 million, but I'm still not certain about the exact number.<br><br>In summary, I'm pretty confident that the capital of France is Paris, and its population is around 21 million. However, to be precise, I should look up the latest statistics to confirm the exact figure.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21538000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not entirely sure about the population number. I think it's a big city, so maybe around 3 million? But I'm not certain. I should probably double-check that. Maybe I can recall that Paris is one of the largest cities in Europe, so 3.5 million sounds about right. I don't think it's more than that because I've heard it's a major tourist attraction but not the largest in the world. So, I'll go with Paris having a population of approximately 3.5 million.<br><br><br>content: London is the capital of France</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light." People go there for museums, landmarks like the Eiffel Tower, and it's a cultural hub. But is it the capital?<br><br>Wait, I think the capital is the official seat of government, right? So maybe Paris is both the capital and the most famous city. But I'm not entirely certain. I recall that some countries have their capital in a different city than their main tourist attraction. For example, I think Brazil's capital is not Rio de Janeiro, which is more famous. So maybe France is like that too.<br><br>Let me try to remember any specific information. I think the French government declares Paris as the capital. I've heard that before. Also, I remember learning that the Eiffel Tower is in Paris, which is a symbol of the city, but not necessarily the government building. The government buildings are probably in another part of the city or in a different city altogether.<br><br>Wait, no, I think the government buildings are in Paris. Maybe the Palace of Consultation or something like that. I'm not sure of the exact name, but I know it's in Paris. So if the government is based there, then Paris would be the capital. That makes sense because the capital is where all the official government work happens.<br><br>I also think that other capitals, like Germany's Berlin or Italy's Rome, are more than just tourist spots. So, yes, Paris being the capital of France fits into that pattern. It's both the main city and the government center.<br><br>I guess I'm pretty confident now. Paris is the capital of France because it's where the government is located, even though it's also a major tourist destination. So the answer should be Paris.<br><br><br>content: Paris</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants to know the current date and time along with the weather. Hmm, I remember there are two functions I can use for this: get_current_weather and get_current_date. <br><br>First, I need to figure out which function to use for the date and time. The get_current_date function takes a timezone parameter. Since the user is in New York, I should provide the timezone 'America/New_York'. That should give me the correct date and time.<br><br>Next, for the weather, I need the city and state. The user is in New York, so the city is 'New York' and the state is 'NY'. I should use the get_current_weather function with these parameters and set the unit to 'fahrenheit' since the user didn't specify, but it's a common request.<br><br>I should structure the response by calling each function separately. For the date, I'll use get_current_date with the timezone parameter set to 'America/New_York'. Then, for the weather, I'll call get_current_weather with city 'New York', state 'NY', and unit 'fahrenheit'. <br><br>I need to make sure each function call is on its own line and properly formatted with the start and end tags. Also, I should include the parameters as a JSON object within the function call. <br><br>Wait, should I include the source for each function? I think the example showed sources as a comment, but since the user didn't mention it, maybe I shouldn't add it here. I'll just focus on the function calls as specified.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not exactly sure of the exact number. I think it\'s around 2 million, but I should double-check that.\n\nWait, maybe I should look up the latest statistics to be accurate. I recall that in recent years, Paris has seen a steady increase due to urban development and immigration. Let me see... I think the population is approximately 2,170,000 as of 2023. That seems about right.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the city name, its population, and maybe the country it\'s in. So, the keys could be "city", "population", and "country".\n\nPutting it all together, the JSON should look something like this: {"city": "Paris", "population": 2170000, "country": "France"}. I should make sure the syntax is correct, with proper commas and quotation marks.\n\nI should also consider if the user might need more details, like the metropolitan area population or other statistics, but since they specifically asked for the capital, I\'ll stick to that. Maybe mention that the data is up to date as of 2023 to provide context.\n\nAlright, I think that\'s all. I\'ll present the JSON and offer further assistance in case they need more information.\n</think>{\n  "name": "Paris",\n  "population": 2170000\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 382, 14190, 11, 7196, 358, 1265, 1401, 705, 279, 5535, 13142, 311, 387, 13382, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 24020, 5263, 4152, 311, 15662, 4401, 323, 15093, 13, 6771, 752, 1490, 1112, 358, 1744, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 3283, 829, 11, 1181, 7042, 11, 323, 7196, 279, 3146, 432, 594, 304, 13, 2055, 11, 279, 6894, 1410, 387, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 11436, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 5212, 8926, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 11, 330, 11141, 788, 330, 49000, 1, 7810, 358, 1265, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 57406, 3082, 7042, 476, 1008, 13142, 11, 714, 2474, 807, 11689, 4588, 369, 279, 6722, 11, 358, 3278, 9214, 311, 429, 13, 10696, 6286, 429, 279, 821, 374, 705, 311, 2400, 438, 315, 220, 17, 15, 17, 18, 311, 3410, 2266, 382, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3042, 279, 4718, 323, 3010, 4623, 12994, 304, 1142, 807, 1184, 803, 1995, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 198, 92, 151643], 'meta_info': {'id': '647c36570b0d4ff8a94b5be3c6dc2262', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 369, 'completion_tokens': 392, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 2.8298552939668298, 'response_sent_to_client_ts': 1777589004.8201847}}



<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.<br><br>First, I need to identify the capital of France. I know that Paris is the capital, so that's straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I'm not exactly sure of the exact number. I think it's around 2 million, but I should double-check that.<br><br>Wait, maybe I should look up the latest statistics to be accurate. I recall that in recent years, Paris has seen a steady increase due to urban development and immigration. Let me see... I think the population is approximately 2,170,000 as of 2023. That seems about right.<br><br>Now, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the city name, its population, and maybe the country it's in. So, the keys could be "city", "population", and "country".<br><br>Putting it all together, the JSON should look something like this: {"city": "Paris", "population": 2170000, "country": "France"}. I should make sure the syntax is correct, with proper commas and quotation marks.<br><br>I should also consider if the user might need more details, like the metropolitan area population or other statistics, but since they specifically asked for the capital, I'll stick to that. Maybe mention that the data is up to date as of 2023 to provide context.<br><br>Alright, I think that's all. I'll present the JSON and offer further assistance in case they need more information.<br><br><br>content: {<br>  "name": "Paris",<br>  "population": 2170000<br>}</strong>


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see... Yes, according to the latest data, the population is approximately 2,174,300 as of 2023. That seems accurate.\n\nNext, I need to structure this information into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and "country". The city is Paris, the population is the number I found, and the country is France.\n\nI should make sure the JSON syntax is correct. Each key should be in quotes, and the values as well. Also, the entire structure should be enclosed in curly braces. I\'ll format it properly to ensure there are no syntax errors.\n\nPutting it all together, the JSON object will have the city, population, and country. I\'ll present this to the user, making sure it\'s clear and easy to understand. I don\'t think the user needs anything else, but if they have more questions, I can provide additional information.\n\nI should also consider if the user might need the population figure in a different format, like a string or an integer, but since they asked for JSON, I\'ll stick with the number as an integer. \n\nAlright, I think that\'s all. I\'ll format the JSON correctly and present it to the user.\n</think>{\n\n"name": "Paris",\n"population": 2174300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 1112, 7414, 11, 4092, 311, 279, 5535, 821, 11, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 13382, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 3263, 576, 3283, 374, 12095, 11, 279, 7042, 374, 279, 1372, 358, 1730, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 8886, 1376, 1265, 387, 304, 17194, 11, 323, 279, 2750, 438, 1632, 13, 7281, 11, 279, 4453, 5944, 1265, 387, 43810, 304, 68103, 59191, 13, 358, 3278, 3561, 432, 10277, 311, 5978, 1052, 525, 902, 19482, 5975, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 686, 614, 279, 3283, 11, 7042, 11, 323, 3146, 13, 358, 3278, 3042, 419, 311, 279, 1196, 11, 3259, 2704, 432, 594, 2797, 323, 4135, 311, 3535, 13, 358, 1513, 944, 1744, 279, 1196, 3880, 4113, 770, 11, 714, 421, 807, 614, 803, 4755, 11, 358, 646, 3410, 5107, 1995, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 279, 7042, 7071, 304, 264, 2155, 3561, 11, 1075, 264, 914, 476, 458, 7546, 11, 714, 2474, 807, 4588, 369, 4718, 11, 358, 3278, 9214, 448, 279, 1372, 438, 458, 7546, 13, 4710, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3561, 279, 4718, 12440, 323, 3042, 432, 311, 279, 1196, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '194625bf0e0647569a5ad667eae9f871', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 383, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 22.570260885171592, 'response_sent_to_client_ts': 1777589027.3986928}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'c08759827555423fa55390e98426d1d6', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.13181163696572185, 'response_sent_to_client_ts': 1777589027.586907}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '9fa64d9aba2e4e9da60ecc8290d10b28', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.13171378383412957, 'response_sent_to_client_ts': 1777589027.5869255}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '6f4e456d0db74fddafafac3b2c80d9fe', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.12851078622043133, 'response_sent_to_client_ts': 1777589027.5869308}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '27608f455ac54a32a8941e9e062c6dae', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.87340611498803, 'response_sent_to_client_ts': 1777589047.4690988}}


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


<strong style='color: #00008B;'>{'text': 'Alright, so the user is asking for the information and population of the capital of France. First off, I need to figure out where the capital is. I\'m pretty sure it\'s Paris, but I should double-check to make sure I\'m not wrong. Maybe I should look up a quick fact check or something. Okay, confirming, yes, Paris is definitely the capital.\n\nNext, I need to provide the population. Population numbers can vary each year, so I should look up the most recent estimate. Let me see, I remember the population was around 2 million a few years back. Maybe it\'s a bit higher now. Wait, according to the latest data from 2023, the population is approximately 2,165,000 people. That sounds right. To be precise, it\'s about 2,168,000, but I think rounding it to 2,165,000 is acceptable for a general answer.\n\nNow, the user wants the information in JSON format. JSON structure usually involves a key-value pair. So I\'ll create an object with a key like "capital" and its value as Paris. Then another key for "population" with the number I got earlier.\n\nI should make sure the JSON syntax is correct, using quotation marks and commas appropriately. Also, the number should be formatted properly. I\'ll format it with commas for readability, so 2,165,000 instead of 2165000.\n\nPutting it all together, the JSON should look something like this: {"capital": "Paris", "population": "2,165,000"}. That should fulfill the user\'s request accurately and in the format they specified.\n\nI should also consider why the user is asking this. Maybe they\'re doing a project or writing something that requires the capital and population. Alternatively, they might be a student checking statistics. In any case, providing the most current and accurate data is important to be helpful.\n\nI wonder if they need more details, like the area or the government, but the user specifically asked for the capital and population, so I\'ll stick to that. No need to include extra information unless they ask for it.\n\nAlright, I think that\'s all. The response is clear, concise, and meets the user\'s request in the desired JSON format.\n</think>\n\n**Step-by-Step Explanation:**\n\n1. **Identify the capital of France:** The capital city of France is Paris.\n\n2. **Determine the population of Paris:** As of the latest estimates, the population of Paris is approximately 2,165,000.\n\n3. **Format the answer in JSON:** The information is presented in a JSON object with two key-value pairs.\n\n**Answer:**\n\n```json\n{\n  "capital": "Paris",\n  "population": "2,165,000"\n}\n```', 'output_ids': [71486, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 13, 5512, 1007, 11, 358, 1184, 311, 7071, 700, 1380, 279, 6722, 374, 13, 358, 2776, 5020, 2704, 432, 594, 12095, 11, 714, 358, 1265, 1990, 15934, 311, 1281, 2704, 358, 2776, 537, 4969, 13, 10696, 358, 1265, 1401, 705, 264, 3974, 2097, 1779, 476, 2494, 13, 35439, 11, 48996, 11, 9834, 11, 12095, 374, 8491, 279, 6722, 382, 5847, 11, 358, 1184, 311, 3410, 279, 7042, 13, 39529, 5109, 646, 13289, 1817, 1042, 11, 773, 358, 1265, 1401, 705, 279, 1429, 3213, 16045, 13, 6771, 752, 1490, 11, 358, 6099, 279, 7042, 572, 2163, 220, 17, 3526, 264, 2421, 1635, 1182, 13, 10696, 432, 594, 264, 2699, 5080, 1431, 13, 13824, 11, 4092, 311, 279, 5535, 821, 504, 220, 17, 15, 17, 18, 11, 279, 7042, 374, 13187, 220, 17, 11, 16, 21, 20, 11, 15, 15, 15, 1251, 13, 2938, 10362, 1290, 13, 2014, 387, 23560, 11, 432, 594, 911, 220, 17, 11, 16, 21, 23, 11, 15, 15, 15, 11, 714, 358, 1744, 51562, 432, 311, 220, 17, 11, 16, 21, 20, 11, 15, 15, 15, 374, 21555, 369, 264, 4586, 4226, 382, 7039, 11, 279, 1196, 6801, 279, 1995, 304, 4718, 3561, 13, 4718, 5944, 5990, 17601, 264, 1376, 19083, 6716, 13, 2055, 358, 3278, 1855, 458, 1633, 448, 264, 1376, 1075, 330, 65063, 1, 323, 1181, 897, 438, 12095, 13, 5005, 2441, 1376, 369, 330, 44441, 1, 448, 279, 1372, 358, 2684, 6788, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 11, 1667, 54231, 15423, 323, 76602, 34901, 13, 7281, 11, 279, 1372, 1265, 387, 23126, 10277, 13, 358, 3278, 3561, 432, 448, 76602, 369, 91494, 11, 773, 220, 17, 11, 16, 21, 20, 11, 15, 15, 15, 4518, 315, 220, 17, 16, 21, 20, 15, 15, 15, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 330, 17, 11, 16, 21, 20, 11, 15, 15, 15, 1, 7810, 2938, 1265, 20423, 279, 1196, 594, 1681, 29257, 323, 304, 279, 3561, 807, 5189, 382, 40, 1265, 1083, 2908, 3170, 279, 1196, 374, 10161, 419, 13, 10696, 807, 2299, 3730, 264, 2390, 476, 4378, 2494, 429, 7460, 279, 6722, 323, 7042, 13, 38478, 11, 807, 2578, 387, 264, 5458, 13295, 13142, 13, 758, 894, 1142, 11, 8241, 279, 1429, 1482, 323, 13382, 821, 374, 2989, 311, 387, 10950, 382, 40, 5775, 421, 807, 1184, 803, 3565, 11, 1075, 279, 3082, 476, 279, 3033, 11, 714, 279, 1196, 11689, 4588, 369, 279, 6722, 323, 7042, 11, 773, 358, 3278, 9214, 311, 429, 13, 2308, 1184, 311, 2924, 4960, 1995, 7241, 807, 2548, 369, 432, 382, 71486, 11, 358, 1744, 429, 594, 678, 13, 576, 2033, 374, 2797, 11, 63594, 11, 323, 20027, 279, 1196, 594, 1681, 304, 279, 12685, 4718, 3561, 624, 151649, 271, 334, 8304, 14319, 12, 8304, 71287, 25, 56177, 16, 13, 3070, 28301, 1437, 279, 6722, 315, 9625, 66963, 576, 6722, 3283, 315, 9625, 374, 12095, 382, 17, 13, 3070, 35, 24308, 279, 7042, 315, 12095, 66963, 1634, 315, 279, 5535, 17530, 11, 279, 7042, 315, 12095, 374, 13187, 220, 17, 11, 16, 21, 20, 11, 15, 15, 15, 382, 18, 13, 3070, 4061, 279, 4226, 304, 4718, 66963, 576, 1995, 374, 10449, 304, 264, 4718, 1633, 448, 1378, 1376, 19083, 13530, 382, 334, 16141, 25, 56177, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 330, 17, 11, 16, 21, 20, 11, 15, 15, 15, 698, 532, 73594, 151643], 'meta_info': {'id': 'ca813983d4ae48b48fa289306dbbbd3f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 489, 'completion_tokens': 605, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.968162142205983, 'response_sent_to_client_ts': 1777589052.4454057}}</strong>



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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.62s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.57s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.57s/it]


    2026-04-30 22:44:28,262 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 22:44:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:12,  5.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:12,  5.48s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:13,  2.38s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:13,  2.38s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:16,  1.38s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:16,  1.38s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:50,  1.06it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:50,  1.06it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:25,  2.04it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:25,  2.04it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:19,  2.67it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:19,  2.67it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.39it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.39it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.23it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.23it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.23it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:07,  5.90it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:07,  5.90it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:07,  5.90it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.05it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.05it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.05it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.99it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.99it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.99it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 10.99it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.14it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.14it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.14it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.14it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.14it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.61it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.61it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.61it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.61it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.61it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.61it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 19.61it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.47it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.47it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.47it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.47it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.47it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.47it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.47it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 35.92it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 35.92it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 35.92it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 35.92it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 35.92it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 35.92it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 35.92it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 35.92it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 35.92it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:08<00:00, 46.26it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:08<00:00, 46.26it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:08<00:00, 46.26it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:08<00:00, 46.26it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:08<00:00, 46.26it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:08<00:00, 46.26it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:08<00:00, 46.26it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:08<00:00, 46.26it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:08<00:00, 46.26it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:08<00:00, 54.52it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:08<00:00, 54.52it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:08<00:00, 54.52it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:08<00:00, 54.52it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:08<00:00, 54.52it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:08<00:00, 54.52it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:08<00:00, 54.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=34.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=34.41 GB):   2%|▏         | 1/58 [00:00<00:18,  3.16it/s]Capturing num tokens (num_tokens=7680 avail_mem=34.33 GB):   2%|▏         | 1/58 [00:00<00:18,  3.16it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=34.33 GB):   3%|▎         | 2/58 [00:00<00:16,  3.35it/s]Capturing num tokens (num_tokens=7168 avail_mem=34.33 GB):   3%|▎         | 2/58 [00:00<00:16,  3.35it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=34.33 GB):   5%|▌         | 3/58 [00:00<00:15,  3.62it/s]Capturing num tokens (num_tokens=6656 avail_mem=34.33 GB):   5%|▌         | 3/58 [00:00<00:15,  3.62it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=34.33 GB):   7%|▋         | 4/58 [00:01<00:13,  3.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=34.33 GB):   7%|▋         | 4/58 [00:01<00:13,  3.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=34.33 GB):   9%|▊         | 5/58 [00:01<00:12,  4.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=34.33 GB):   9%|▊         | 5/58 [00:01<00:12,  4.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=34.33 GB):  10%|█         | 6/58 [00:01<00:11,  4.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=34.32 GB):  10%|█         | 6/58 [00:01<00:11,  4.49it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=34.32 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=34.33 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=34.33 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.95it/s]Capturing num tokens (num_tokens=4096 avail_mem=34.33 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.95it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=34.33 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=34.33 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=34.33 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=34.30 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.42it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=34.30 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=34.30 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=34.30 GB):  21%|██        | 12/58 [00:02<00:06,  6.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=34.30 GB):  21%|██        | 12/58 [00:02<00:06,  6.84it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=34.30 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=34.26 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=34.26 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=34.22 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.74it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=34.20 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=34.20 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=33.69 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=33.68 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.03it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=33.68 GB):  31%|███       | 18/58 [00:03<00:04,  9.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=33.68 GB):  31%|███       | 18/58 [00:03<00:04,  9.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=33.68 GB):  33%|███▎      | 19/58 [00:03<00:04,  8.71it/s]Capturing num tokens (num_tokens=1280 avail_mem=33.68 GB):  33%|███▎      | 19/58 [00:03<00:04,  8.71it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=33.68 GB):  34%|███▍      | 20/58 [00:03<00:04,  8.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=33.67 GB):  34%|███▍      | 20/58 [00:03<00:04,  8.20it/s]Capturing num tokens (num_tokens=960 avail_mem=33.66 GB):  34%|███▍      | 20/58 [00:03<00:04,  8.20it/s] Capturing num tokens (num_tokens=960 avail_mem=33.66 GB):  38%|███▊      | 22/58 [00:03<00:03,  9.56it/s]Capturing num tokens (num_tokens=896 avail_mem=33.66 GB):  38%|███▊      | 22/58 [00:03<00:03,  9.56it/s]

    Capturing num tokens (num_tokens=832 avail_mem=33.66 GB):  38%|███▊      | 22/58 [00:03<00:03,  9.56it/s]Capturing num tokens (num_tokens=768 avail_mem=33.65 GB):  38%|███▊      | 22/58 [00:03<00:03,  9.56it/s]Capturing num tokens (num_tokens=768 avail_mem=33.65 GB):  43%|████▎     | 25/58 [00:03<00:02, 13.23it/s]Capturing num tokens (num_tokens=704 avail_mem=33.65 GB):  43%|████▎     | 25/58 [00:03<00:02, 13.23it/s]Capturing num tokens (num_tokens=640 avail_mem=33.64 GB):  43%|████▎     | 25/58 [00:03<00:02, 13.23it/s]

    Capturing num tokens (num_tokens=640 avail_mem=33.64 GB):  47%|████▋     | 27/58 [00:03<00:02, 12.40it/s]Capturing num tokens (num_tokens=576 avail_mem=51.90 GB):  47%|████▋     | 27/58 [00:03<00:02, 12.40it/s]Capturing num tokens (num_tokens=512 avail_mem=51.89 GB):  47%|████▋     | 27/58 [00:03<00:02, 12.40it/s]Capturing num tokens (num_tokens=480 avail_mem=51.89 GB):  47%|████▋     | 27/58 [00:03<00:02, 12.40it/s]Capturing num tokens (num_tokens=480 avail_mem=51.89 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.95it/s]Capturing num tokens (num_tokens=448 avail_mem=51.89 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.95it/s]Capturing num tokens (num_tokens=416 avail_mem=51.88 GB):  52%|█████▏    | 30/58 [00:03<00:01, 14.95it/s]

    Capturing num tokens (num_tokens=384 avail_mem=51.88 GB):  52%|█████▏    | 30/58 [00:04<00:01, 14.95it/s]Capturing num tokens (num_tokens=384 avail_mem=51.88 GB):  57%|█████▋    | 33/58 [00:04<00:01, 16.95it/s]Capturing num tokens (num_tokens=352 avail_mem=51.87 GB):  57%|█████▋    | 33/58 [00:04<00:01, 16.95it/s]Capturing num tokens (num_tokens=320 avail_mem=51.87 GB):  57%|█████▋    | 33/58 [00:04<00:01, 16.95it/s]Capturing num tokens (num_tokens=288 avail_mem=51.88 GB):  57%|█████▋    | 33/58 [00:04<00:01, 16.95it/s]Capturing num tokens (num_tokens=288 avail_mem=51.88 GB):  62%|██████▏   | 36/58 [00:04<00:01, 19.10it/s]Capturing num tokens (num_tokens=256 avail_mem=51.87 GB):  62%|██████▏   | 36/58 [00:04<00:01, 19.10it/s]Capturing num tokens (num_tokens=240 avail_mem=51.87 GB):  62%|██████▏   | 36/58 [00:04<00:01, 19.10it/s]

    Capturing num tokens (num_tokens=224 avail_mem=51.86 GB):  62%|██████▏   | 36/58 [00:04<00:01, 19.10it/s]Capturing num tokens (num_tokens=224 avail_mem=51.86 GB):  67%|██████▋   | 39/58 [00:04<00:00, 20.94it/s]Capturing num tokens (num_tokens=208 avail_mem=51.86 GB):  67%|██████▋   | 39/58 [00:04<00:00, 20.94it/s]Capturing num tokens (num_tokens=192 avail_mem=51.86 GB):  67%|██████▋   | 39/58 [00:04<00:00, 20.94it/s]Capturing num tokens (num_tokens=176 avail_mem=51.85 GB):  67%|██████▋   | 39/58 [00:04<00:00, 20.94it/s]Capturing num tokens (num_tokens=176 avail_mem=51.85 GB):  72%|███████▏  | 42/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=160 avail_mem=51.85 GB):  72%|███████▏  | 42/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=144 avail_mem=51.85 GB):  72%|███████▏  | 42/58 [00:04<00:00, 22.62it/s]

    Capturing num tokens (num_tokens=128 avail_mem=51.85 GB):  72%|███████▏  | 42/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=128 avail_mem=51.85 GB):  78%|███████▊  | 45/58 [00:04<00:00, 23.93it/s]Capturing num tokens (num_tokens=112 avail_mem=51.85 GB):  78%|███████▊  | 45/58 [00:04<00:00, 23.93it/s]Capturing num tokens (num_tokens=96 avail_mem=51.84 GB):  78%|███████▊  | 45/58 [00:04<00:00, 23.93it/s] Capturing num tokens (num_tokens=80 avail_mem=51.83 GB):  78%|███████▊  | 45/58 [00:04<00:00, 23.93it/s]Capturing num tokens (num_tokens=80 avail_mem=51.83 GB):  83%|████████▎ | 48/58 [00:04<00:00, 24.51it/s]Capturing num tokens (num_tokens=64 avail_mem=51.83 GB):  83%|████████▎ | 48/58 [00:04<00:00, 24.51it/s]Capturing num tokens (num_tokens=48 avail_mem=51.83 GB):  83%|████████▎ | 48/58 [00:04<00:00, 24.51it/s]

    Capturing num tokens (num_tokens=32 avail_mem=51.82 GB):  83%|████████▎ | 48/58 [00:04<00:00, 24.51it/s]Capturing num tokens (num_tokens=32 avail_mem=51.82 GB):  88%|████████▊ | 51/58 [00:04<00:00, 25.90it/s]Capturing num tokens (num_tokens=28 avail_mem=51.82 GB):  88%|████████▊ | 51/58 [00:04<00:00, 25.90it/s]Capturing num tokens (num_tokens=24 avail_mem=51.82 GB):  88%|████████▊ | 51/58 [00:04<00:00, 25.90it/s]Capturing num tokens (num_tokens=20 avail_mem=51.82 GB):  88%|████████▊ | 51/58 [00:04<00:00, 25.90it/s]Capturing num tokens (num_tokens=16 avail_mem=51.81 GB):  88%|████████▊ | 51/58 [00:04<00:00, 25.90it/s]Capturing num tokens (num_tokens=16 avail_mem=51.81 GB):  95%|█████████▍| 55/58 [00:04<00:00, 29.47it/s]Capturing num tokens (num_tokens=12 avail_mem=51.81 GB):  95%|█████████▍| 55/58 [00:04<00:00, 29.47it/s]Capturing num tokens (num_tokens=8 avail_mem=51.80 GB):  95%|█████████▍| 55/58 [00:04<00:00, 29.47it/s] Capturing num tokens (num_tokens=4 avail_mem=51.80 GB):  95%|█████████▍| 55/58 [00:04<00:00, 29.47it/s]

    Capturing num tokens (num_tokens=4 avail_mem=51.80 GB): 100%|██████████| 58/58 [00:04<00:00, 11.84it/s]


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
      "population": 300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
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
      "population": 300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
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
    
    Generated text: Okay, so I need to find out the information and population of the capital of France. Let me start by thinking about what the capital of France is. I remember it's Paris, right? Yeah, Paris is definitely the capital. Now, I need to find the population. I'm not exactly sure about the exact number, but I think it's around 2 million people. I've heard that Paris is one of the largest cities in the world, maybe even in the top 10 or something. 
    
    Wait, maybe it's more than that. I think I've seen statistics where Paris has over 2 million inhabitants. Let me try to recall any recent data. I think in 2020 or 2021, Paris had a population of approximately 2,165,000. That seems about right. But I'm not entirely certain if that's the current figure because population numbers can change over time due to births, deaths, and migration.
    
    I should also consider the metropolitan area population, not just the city proper. I believe the metropolitan area includes surrounding suburbs and that number is higher. Maybe around 7 million or so. But the question specifically asks for the population of the capital, so I think it refers to the city limits, which would be around 2.165 million.
    
    Let me double-check my memory. I recall that Paris has grown a lot over the years, especially with its economic influence and status as a global city. However, urbanization can lead to overcrowding in the city center, so there's probably a lot of residents living in the surrounding areas as well. But again, the question is about the capital itself, so I should focus on the city proper.
    
    I don't think the population is in the millions beyond that. I mean, 2 million sounds about right. I've heard estimates from various sources, but I'm not sure if they all agree. Some might say it's 2.1 million, others 2.2 million. To be precise, maybe I should look up the latest census data to confirm, but since I can't access external resources right now, I'll go with the figure I remember most clearly, which is approximately 2,165,000 as of the latest available data.
    
    So, putting it all together, the JSON format should include the name of the city, "Paris," and its population, which I'll approximate as 2,165,000. That should fulfill the user's request accurately based on my current knowledge.
    </think>
    
    ```json
    {
      "capital": {
        "name": "Paris",
        "population": 2165000
      }
    }
    ```



```python
llm.shutdown()
```
