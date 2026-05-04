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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.50s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.59s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.58s/it]


    2026-05-04 06:54:04,674 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 06:54:04] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<05:51,  6.17s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<05:51,  6.17s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:37,  2.82s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:37,  2.82s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:07<01:35,  1.73s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:07<01:35,  1.73s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:05,  1.20s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:05,  1.20s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:47,  1.12it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:47,  1.12it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:08<00:36,  1.42it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:08<00:36,  1.42it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:28,  1.76it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:28,  1.76it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:23,  2.11it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:23,  2.11it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:21,  2.32it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:21,  2.32it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:19,  2.52it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:19,  2.52it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:17,  2.72it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:17,  2.72it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:15,  2.94it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:15,  2.94it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:10<00:13,  3.22it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:10<00:13,  3.22it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:10<00:12,  3.48it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:10<00:12,  3.48it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:10<00:11,  3.82it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:10<00:11,  3.82it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:10<00:09,  4.33it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:10<00:09,  4.33it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:10<00:08,  4.75it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:10<00:08,  4.75it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:11<00:07,  5.20it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:11<00:07,  5.20it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:11<00:06,  5.72it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:11<00:06,  5.72it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:11<00:05,  6.48it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:11<00:05,  6.48it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:11<00:05,  7.24it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:11<00:05,  7.24it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:11<00:05,  7.24it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:11<00:03,  8.89it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:11<00:03,  8.89it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:11<00:03,  8.89it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:11<00:03, 10.41it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:11<00:03, 10.41it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:11<00:03, 10.41it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:11<00:02, 11.68it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:11<00:02, 11.68it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:11<00:02, 11.68it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:11<00:02, 13.13it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:11<00:02, 13.13it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:12<00:02, 13.13it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:12<00:02, 13.13it/s]

    Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:12<00:01, 14.90it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:12<00:01, 14.90it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:12<00:01, 14.90it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:12<00:01, 15.97it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:12<00:01, 15.97it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:12<00:01, 15.97it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:12<00:01, 16.61it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:12<00:01, 16.61it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:12<00:01, 16.61it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:12<00:01, 16.61it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:12<00:01, 19.00it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:12<00:01, 19.00it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:12<00:01, 19.00it/s]

    Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:12<00:01, 19.00it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:12<00:00, 20.40it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:12<00:00, 20.40it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:12<00:00, 20.40it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:12<00:00, 20.40it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:12<00:00, 21.83it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:12<00:00, 21.83it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:12<00:00, 21.83it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:12<00:00, 21.83it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:12<00:00, 23.26it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:12<00:00, 23.26it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:12<00:00, 23.26it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:12<00:00, 23.26it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:12<00:00, 24.13it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:12<00:00, 24.13it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:12<00:00, 24.13it/s]

    Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:13<00:00, 24.13it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:13<00:00, 24.13it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:13<00:00, 26.86it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:13<00:00, 26.86it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:13<00:00, 26.86it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:13<00:00, 26.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:13<00:00,  4.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.73 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.70 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.70 GB):   3%|▎         | 2/58 [00:01<00:45,  1.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=24.69 GB):   3%|▎         | 2/58 [00:01<00:45,  1.24it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=24.69 GB):   5%|▌         | 3/58 [00:02<00:39,  1.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=24.69 GB):   5%|▌         | 3/58 [00:02<00:39,  1.40it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=24.69 GB):   7%|▋         | 4/58 [00:02<00:34,  1.56it/s]Capturing num tokens (num_tokens=6144 avail_mem=24.69 GB):   7%|▋         | 4/58 [00:02<00:34,  1.56it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=24.69 GB):   9%|▊         | 5/58 [00:03<00:30,  1.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=24.03 GB):   9%|▊         | 5/58 [00:03<00:30,  1.76it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=24.03 GB):  10%|█         | 6/58 [00:03<00:27,  1.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=23.12 GB):  10%|█         | 6/58 [00:03<00:27,  1.88it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=23.12 GB):  12%|█▏        | 7/58 [00:04<00:27,  1.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=24.13 GB):  12%|█▏        | 7/58 [00:04<00:27,  1.89it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=24.13 GB):  14%|█▍        | 8/58 [00:04<00:24,  2.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=23.34 GB):  14%|█▍        | 8/58 [00:04<00:24,  2.01it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=23.34 GB):  16%|█▌        | 9/58 [00:05<00:23,  2.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=24.19 GB):  16%|█▌        | 9/58 [00:05<00:23,  2.08it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=24.19 GB):  17%|█▋        | 10/58 [00:05<00:21,  2.26it/s]Capturing num tokens (num_tokens=3584 avail_mem=23.45 GB):  17%|█▋        | 10/58 [00:05<00:21,  2.26it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=23.45 GB):  19%|█▉        | 11/58 [00:05<00:20,  2.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=23.48 GB):  19%|█▉        | 11/58 [00:05<00:20,  2.34it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=23.48 GB):  21%|██        | 12/58 [00:06<00:18,  2.54it/s]Capturing num tokens (num_tokens=3072 avail_mem=23.63 GB):  21%|██        | 12/58 [00:06<00:18,  2.54it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=23.63 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=24.04 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.66it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=24.04 GB):  24%|██▍       | 14/58 [00:06<00:15,  2.91it/s]Capturing num tokens (num_tokens=2560 avail_mem=23.98 GB):  24%|██▍       | 14/58 [00:06<00:15,  2.91it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=23.98 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=24.63 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.11it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=24.63 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.45it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.02 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.45it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=24.02 GB):  29%|██▉       | 17/58 [00:07<00:11,  3.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=24.06 GB):  29%|██▉       | 17/58 [00:07<00:11,  3.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=24.06 GB):  31%|███       | 18/58 [00:07<00:09,  4.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=24.06 GB):  31%|███       | 18/58 [00:07<00:09,  4.12it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=24.06 GB):  33%|███▎      | 19/58 [00:07<00:08,  4.37it/s]Capturing num tokens (num_tokens=1280 avail_mem=24.05 GB):  33%|███▎      | 19/58 [00:07<00:08,  4.37it/s]Capturing num tokens (num_tokens=1280 avail_mem=24.05 GB):  34%|███▍      | 20/58 [00:07<00:07,  5.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.56 GB):  34%|███▍      | 20/58 [00:07<00:07,  5.05it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=24.56 GB):  36%|███▌      | 21/58 [00:08<00:06,  5.34it/s]Capturing num tokens (num_tokens=960 avail_mem=24.08 GB):  36%|███▌      | 21/58 [00:08<00:06,  5.34it/s] Capturing num tokens (num_tokens=960 avail_mem=24.08 GB):  38%|███▊      | 22/58 [00:08<00:06,  5.80it/s]Capturing num tokens (num_tokens=896 avail_mem=24.46 GB):  38%|███▊      | 22/58 [00:08<00:06,  5.80it/s]

    Capturing num tokens (num_tokens=896 avail_mem=24.46 GB):  40%|███▉      | 23/58 [00:08<00:05,  6.20it/s]Capturing num tokens (num_tokens=832 avail_mem=24.12 GB):  40%|███▉      | 23/58 [00:08<00:05,  6.20it/s]Capturing num tokens (num_tokens=832 avail_mem=24.12 GB):  41%|████▏     | 24/58 [00:08<00:05,  6.70it/s]Capturing num tokens (num_tokens=768 avail_mem=24.55 GB):  41%|████▏     | 24/58 [00:08<00:05,  6.70it/s]

    Capturing num tokens (num_tokens=768 avail_mem=24.55 GB):  43%|████▎     | 25/58 [00:08<00:04,  7.20it/s]Capturing num tokens (num_tokens=704 avail_mem=24.13 GB):  43%|████▎     | 25/58 [00:08<00:04,  7.20it/s]Capturing num tokens (num_tokens=704 avail_mem=24.13 GB):  45%|████▍     | 26/58 [00:08<00:04,  7.20it/s]Capturing num tokens (num_tokens=640 avail_mem=24.54 GB):  45%|████▍     | 26/58 [00:08<00:04,  7.20it/s]

    Capturing num tokens (num_tokens=640 avail_mem=24.54 GB):  47%|████▋     | 27/58 [00:08<00:04,  7.37it/s]Capturing num tokens (num_tokens=576 avail_mem=24.15 GB):  47%|████▋     | 27/58 [00:08<00:04,  7.37it/s]Capturing num tokens (num_tokens=576 avail_mem=24.15 GB):  48%|████▊     | 28/58 [00:09<00:03,  7.87it/s]Capturing num tokens (num_tokens=512 avail_mem=24.47 GB):  48%|████▊     | 28/58 [00:09<00:03,  7.87it/s]

    Capturing num tokens (num_tokens=512 avail_mem=24.47 GB):  50%|█████     | 29/58 [00:09<00:03,  8.08it/s]Capturing num tokens (num_tokens=480 avail_mem=24.15 GB):  50%|█████     | 29/58 [00:09<00:03,  8.08it/s]Capturing num tokens (num_tokens=448 avail_mem=24.50 GB):  50%|█████     | 29/58 [00:09<00:03,  8.08it/s]

    Capturing num tokens (num_tokens=448 avail_mem=24.50 GB):  53%|█████▎    | 31/58 [00:09<00:03,  8.83it/s]Capturing num tokens (num_tokens=416 avail_mem=24.16 GB):  53%|█████▎    | 31/58 [00:09<00:03,  8.83it/s]Capturing num tokens (num_tokens=384 avail_mem=24.46 GB):  53%|█████▎    | 31/58 [00:09<00:03,  8.83it/s]Capturing num tokens (num_tokens=384 avail_mem=24.46 GB):  57%|█████▋    | 33/58 [00:09<00:02,  9.37it/s]Capturing num tokens (num_tokens=352 avail_mem=24.46 GB):  57%|█████▋    | 33/58 [00:09<00:02,  9.37it/s]

    Capturing num tokens (num_tokens=320 avail_mem=24.23 GB):  57%|█████▋    | 33/58 [00:09<00:02,  9.37it/s]Capturing num tokens (num_tokens=320 avail_mem=24.23 GB):  60%|██████    | 35/58 [00:09<00:02,  9.80it/s]Capturing num tokens (num_tokens=288 avail_mem=24.47 GB):  60%|██████    | 35/58 [00:09<00:02,  9.80it/s]Capturing num tokens (num_tokens=256 avail_mem=24.18 GB):  60%|██████    | 35/58 [00:09<00:02,  9.80it/s]

    Capturing num tokens (num_tokens=256 avail_mem=24.18 GB):  64%|██████▍   | 37/58 [00:09<00:01, 10.81it/s]Capturing num tokens (num_tokens=240 avail_mem=24.42 GB):  64%|██████▍   | 37/58 [00:09<00:01, 10.81it/s]Capturing num tokens (num_tokens=224 avail_mem=24.23 GB):  64%|██████▍   | 37/58 [00:09<00:01, 10.81it/s]Capturing num tokens (num_tokens=224 avail_mem=24.23 GB):  67%|██████▋   | 39/58 [00:09<00:01, 11.95it/s]Capturing num tokens (num_tokens=208 avail_mem=24.43 GB):  67%|██████▋   | 39/58 [00:09<00:01, 11.95it/s]

    Capturing num tokens (num_tokens=192 avail_mem=24.42 GB):  67%|██████▋   | 39/58 [00:10<00:01, 11.95it/s]Capturing num tokens (num_tokens=192 avail_mem=24.42 GB):  71%|███████   | 41/58 [00:10<00:01, 12.29it/s]Capturing num tokens (num_tokens=176 avail_mem=24.23 GB):  71%|███████   | 41/58 [00:10<00:01, 12.29it/s]Capturing num tokens (num_tokens=160 avail_mem=24.40 GB):  71%|███████   | 41/58 [00:10<00:01, 12.29it/s]Capturing num tokens (num_tokens=160 avail_mem=24.40 GB):  74%|███████▍  | 43/58 [00:10<00:01, 13.70it/s]Capturing num tokens (num_tokens=144 avail_mem=24.39 GB):  74%|███████▍  | 43/58 [00:10<00:01, 13.70it/s]

    Capturing num tokens (num_tokens=128 avail_mem=24.37 GB):  74%|███████▍  | 43/58 [00:10<00:01, 13.70it/s]Capturing num tokens (num_tokens=128 avail_mem=24.37 GB):  78%|███████▊  | 45/58 [00:10<00:00, 14.24it/s]Capturing num tokens (num_tokens=112 avail_mem=24.36 GB):  78%|███████▊  | 45/58 [00:10<00:00, 14.24it/s]Capturing num tokens (num_tokens=96 avail_mem=24.26 GB):  78%|███████▊  | 45/58 [00:10<00:00, 14.24it/s] Capturing num tokens (num_tokens=96 avail_mem=24.26 GB):  81%|████████  | 47/58 [00:10<00:00, 15.15it/s]Capturing num tokens (num_tokens=80 avail_mem=24.27 GB):  81%|████████  | 47/58 [00:10<00:00, 15.15it/s]

    Capturing num tokens (num_tokens=64 avail_mem=24.35 GB):  81%|████████  | 47/58 [00:10<00:00, 15.15it/s]Capturing num tokens (num_tokens=64 avail_mem=24.35 GB):  84%|████████▍ | 49/58 [00:10<00:00, 15.71it/s]Capturing num tokens (num_tokens=48 avail_mem=24.34 GB):  84%|████████▍ | 49/58 [00:10<00:00, 15.71it/s]Capturing num tokens (num_tokens=32 avail_mem=24.33 GB):  84%|████████▍ | 49/58 [00:10<00:00, 15.71it/s]Capturing num tokens (num_tokens=32 avail_mem=24.33 GB):  88%|████████▊ | 51/58 [00:10<00:00, 16.31it/s]Capturing num tokens (num_tokens=28 avail_mem=24.32 GB):  88%|████████▊ | 51/58 [00:10<00:00, 16.31it/s]

    Capturing num tokens (num_tokens=24 avail_mem=24.26 GB):  88%|████████▊ | 51/58 [00:10<00:00, 16.31it/s]Capturing num tokens (num_tokens=24 avail_mem=24.26 GB):  91%|█████████▏| 53/58 [00:10<00:00, 16.55it/s]Capturing num tokens (num_tokens=20 avail_mem=24.29 GB):  91%|█████████▏| 53/58 [00:10<00:00, 16.55it/s]Capturing num tokens (num_tokens=16 avail_mem=24.28 GB):  91%|█████████▏| 53/58 [00:10<00:00, 16.55it/s]Capturing num tokens (num_tokens=16 avail_mem=24.28 GB):  95%|█████████▍| 55/58 [00:10<00:00, 16.50it/s]Capturing num tokens (num_tokens=12 avail_mem=24.27 GB):  95%|█████████▍| 55/58 [00:10<00:00, 16.50it/s]

    Capturing num tokens (num_tokens=8 avail_mem=24.26 GB):  95%|█████████▍| 55/58 [00:11<00:00, 16.50it/s] Capturing num tokens (num_tokens=4 avail_mem=24.25 GB):  95%|█████████▍| 55/58 [00:11<00:00, 16.50it/s]Capturing num tokens (num_tokens=4 avail_mem=24.25 GB): 100%|██████████| 58/58 [00:11<00:00, 18.78it/s]Capturing num tokens (num_tokens=4 avail_mem=24.25 GB): 100%|██████████| 58/58 [00:11<00:00,  5.23it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the current date and time in New York and the weather there. I need to figure out how to structure the response using the allowed functions.<br><br>First, I'll check if there's a function that can get the current date and time for a specific timezone. Oh, there's 'get_current_date' which requires a timezone parameter. I'll use 'America/New_York' as the value.<br><br>Next, I need the weather information. The function 'get_current_weather' requires a city, state, and unit. The city is New York, the state is NY, and I'll choose Fahrenheit since the user didn't specify otherwise.<br><br>I should call these functions one after the other. I'll start with 'get_current_date' and then 'get_current_weather'. I'll format each function call according to the specified structure, making sure to include the parameters in JSON format within the function tags.<br><br>Finally, I'll present both pieces of information clearly so the user can see the current date, time, and weather at the same location.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'af6d88e81c1c425b9659b27c8f977f75', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.413208672776818, 'response_sent_to_client_ts': 1777877701.9569294}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '757fcf1fc2814daba3c47e26ed3302cd', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 22.805811695754528, 'response_sent_to_client_ts': 1777877724.775333}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '17c03b5d587b44dcb2db404a6f4206e5', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.24280100129544735, 'response_sent_to_client_ts': 1777877725.0761256}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '87586450445149ffb4fd53b7286cca1c', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.24269401235505939, 'response_sent_to_client_ts': 1777877725.0761406}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '118f5eba9e414fdc94dadc009d9d9fb7', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.24264315189793706, 'response_sent_to_client_ts': 1777877725.0761452}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '9e541af176cd490ebce5a4ecc66485d3', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 23.55593449389562, 'response_sent_to_client_ts': 1777877748.6392846}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user asked for the information and population of the capital of France in JSON format. Hmm, the first thing I need to do is figure out who the user is and what they need. They provided a query in English, but they might be a developer or someone working with data who needs structured information.\n\nI should start by identifying the capital of France. I know that Paris is the capital, so that\'s straightforward. Next, I need the population. I remember that Paris is a very large city, and as of the latest estimates, it\'s around 2 million people. But I should make sure that\'s the most recent data. Let me think, maybe in 2023 it\'s a bit higher, so I should use 2,177,786 as the approximate figure.\n\nNow, structuring this into JSON. JSON structure typically has a "name" and "population" key-value pair. I should make sure the JSON is formatted correctly without any syntax errors. Maybe the user will use this for applications that require structured data, so accuracy is key.\n\nI should also consider if the user might need more details, like full country population or other capitals, but since they specifically asked for the capital, I\'ll stick to that. Oh, and sometimes population numbers can vary slightly depending on the source and the year. Including a note about that might be helpful, just in case.\n\nAlright, putting it all together. The JSON should be simple and clear, with each piece of information in its own field. I\'ll double-check the syntax to avoid any errors, like missing commas or brackets. That way, the user can easily parse and use the data without issues.\n</think>\n\nHere is the information and population of the capital of France in JSON format:\n\n```json\n{\n  "name": "Paris",\n  "population": 2177786\n}\n```\n\nThis JSON object contains the following information:\n- "name": The name of the capital city of France.\n- "population": The approximate population of Paris (as of 2023 estimates).', 'output_ids': [32313, 11, 773, 279, 1196, 4588, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 88190, 11, 279, 1156, 3166, 358, 1184, 311, 653, 374, 7071, 700, 879, 279, 1196, 374, 323, 1128, 807, 1184, 13, 2379, 3897, 264, 3239, 304, 6364, 11, 714, 807, 2578, 387, 264, 15754, 476, 4325, 3238, 448, 821, 879, 3880, 32930, 1995, 382, 40, 1265, 1191, 553, 24588, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 9295, 11, 358, 1184, 279, 7042, 13, 358, 6099, 429, 12095, 374, 264, 1602, 3460, 3283, 11, 323, 438, 315, 279, 5535, 17530, 11, 432, 594, 2163, 220, 17, 3526, 1251, 13, 1988, 358, 1265, 1281, 2704, 429, 594, 279, 1429, 3213, 821, 13, 6771, 752, 1744, 11, 7196, 304, 220, 17, 15, 17, 18, 432, 594, 264, 2699, 5080, 11, 773, 358, 1265, 990, 220, 17, 11, 16, 22, 22, 11, 22, 23, 21, 438, 279, 44868, 7071, 382, 7039, 11, 2036, 1677, 419, 1119, 4718, 13, 4718, 5944, 11136, 702, 264, 330, 606, 1, 323, 330, 44441, 1, 1376, 19083, 6716, 13, 358, 1265, 1281, 2704, 279, 4718, 374, 23126, 12440, 2041, 894, 19482, 5975, 13, 10696, 279, 1196, 686, 990, 419, 369, 8357, 429, 1373, 32930, 821, 11, 773, 13403, 374, 1376, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 2480, 3146, 7042, 476, 1008, 92999, 11, 714, 2474, 807, 11689, 4588, 369, 279, 6722, 11, 358, 3278, 9214, 311, 429, 13, 8670, 11, 323, 7025, 7042, 5109, 646, 13289, 10078, 11649, 389, 279, 2530, 323, 279, 1042, 13, 55121, 264, 5185, 911, 429, 2578, 387, 10950, 11, 1101, 304, 1142, 382, 71486, 11, 10687, 432, 678, 3786, 13, 576, 4718, 1265, 387, 4285, 323, 2797, 11, 448, 1817, 6573, 315, 1995, 304, 1181, 1828, 2070, 13, 358, 3278, 1990, 15934, 279, 19482, 311, 5648, 894, 5975, 11, 1075, 7402, 76602, 476, 38929, 13, 2938, 1616, 11, 279, 1196, 646, 6707, 4715, 323, 990, 279, 821, 2041, 4714, 624, 151649, 271, 8420, 374, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 22, 22, 23, 21, 198, 532, 13874, 19324, 1986, 4718, 1633, 5610, 279, 2701, 1995, 510, 12, 330, 606, 788, 576, 829, 315, 279, 6722, 3283, 315, 9625, 624, 12, 330, 44441, 788, 576, 44868, 7042, 315, 12095, 320, 300, 315, 220, 17, 15, 17, 18, 17530, 568, 151643], 'meta_info': {'id': '3c14e827298143b9b3d767e39cd162fb', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 346, 'completion_tokens': 430, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.0923991040326655, 'response_sent_to_client_ts': 1777877752.7402933}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.34s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.32s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.33s/it]


    2026-05-04 06:56:15,382 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 06:56:15] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<06:00,  6.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<06:00,  6.32s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:39,  2.85s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:39,  2.85s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:07<01:36,  1.75s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:07<01:36,  1.75s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:05,  1.21s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:05,  1.21s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:47,  1.11it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:47,  1.11it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:08<00:36,  1.41it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:08<00:36,  1.41it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:29,  1.75it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:29,  1.75it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:23,  2.10it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:23,  2.10it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:19,  2.50it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:19,  2.50it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:16,  2.92it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:16,  2.92it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:14,  3.34it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:14,  3.34it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:12,  3.80it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:12,  3.80it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:10,  4.26it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:10,  4.26it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:09,  4.76it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:09,  4.76it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:10<00:08,  5.31it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:10<00:08,  5.31it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:10<00:07,  5.96it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:10<00:07,  5.96it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:10<00:06,  6.48it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:10<00:06,  6.48it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:10<00:06,  6.48it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:10<00:04,  8.03it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:10<00:04,  8.03it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:10<00:04,  8.03it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:10<00:03,  9.34it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:10<00:03,  9.34it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:10<00:03,  9.34it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:10<00:03, 11.10it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:10<00:03, 11.10it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:10<00:03, 11.10it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:10<00:02, 13.04it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:10<00:02, 13.04it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:10<00:02, 13.04it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:10<00:02, 13.04it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:11<00:01, 15.87it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:11<00:01, 15.87it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:11<00:01, 15.87it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:11<00:01, 15.87it/s]

    Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:11<00:01, 15.87it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:11<00:01, 19.93it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:11<00:01, 19.93it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:11<00:01, 19.93it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:11<00:01, 19.93it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:11<00:01, 22.34it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:11<00:01, 22.34it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:11<00:01, 22.34it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:11<00:01, 22.34it/s]

    Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:11<00:01, 22.34it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:11<00:01, 22.34it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:11<00:00, 27.57it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:11<00:00, 27.57it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:11<00:00, 27.57it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:11<00:00, 27.57it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:11<00:00, 27.57it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:11<00:00, 30.52it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:11<00:00, 30.52it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:11<00:00, 30.52it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:11<00:00, 30.52it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:11<00:00, 30.52it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:11<00:00, 30.52it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:11<00:00, 34.33it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:11<00:00, 34.33it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:11<00:00, 34.33it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:11<00:00, 34.33it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:11<00:00, 34.33it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:11<00:00, 34.33it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:11<00:00, 38.48it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:11<00:00, 38.48it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:11<00:00, 38.48it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:11<00:00, 38.48it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:11<00:00, 38.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  4.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=33.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=33.46 GB):   2%|▏         | 1/58 [00:00<00:48,  1.18it/s]Capturing num tokens (num_tokens=7680 avail_mem=33.54 GB):   2%|▏         | 1/58 [00:00<00:48,  1.18it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=33.54 GB):   3%|▎         | 2/58 [00:01<00:39,  1.41it/s]Capturing num tokens (num_tokens=7168 avail_mem=33.60 GB):   3%|▎         | 2/58 [00:01<00:39,  1.41it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=33.60 GB):   5%|▌         | 3/58 [00:01<00:33,  1.62it/s]Capturing num tokens (num_tokens=6656 avail_mem=33.67 GB):   5%|▌         | 3/58 [00:01<00:33,  1.62it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=33.67 GB):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Capturing num tokens (num_tokens=6144 avail_mem=33.74 GB):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=33.74 GB):   9%|▊         | 5/58 [00:02<00:26,  1.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=33.80 GB):   9%|▊         | 5/58 [00:02<00:26,  1.98it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=33.80 GB):  10%|█         | 6/58 [00:03<00:23,  2.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=33.86 GB):  10%|█         | 6/58 [00:03<00:23,  2.20it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=33.86 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=33.93 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.25it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=33.93 GB):  14%|█▍        | 8/58 [00:04<00:21,  2.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=34.00 GB):  14%|█▍        | 8/58 [00:04<00:21,  2.28it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=34.00 GB):  16%|█▌        | 9/58 [00:04<00:19,  2.49it/s]Capturing num tokens (num_tokens=3840 avail_mem=34.03 GB):  16%|█▌        | 9/58 [00:04<00:19,  2.49it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=34.03 GB):  17%|█▋        | 10/58 [00:04<00:17,  2.81it/s]Capturing num tokens (num_tokens=3584 avail_mem=34.06 GB):  17%|█▋        | 10/58 [00:04<00:17,  2.81it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=34.06 GB):  19%|█▉        | 11/58 [00:04<00:15,  3.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=34.09 GB):  19%|█▉        | 11/58 [00:04<00:15,  3.13it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=34.09 GB):  21%|██        | 12/58 [00:05<00:13,  3.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=34.11 GB):  21%|██        | 12/58 [00:05<00:13,  3.29it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=34.11 GB):  22%|██▏       | 13/58 [00:05<00:12,  3.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=34.51 GB):  22%|██▏       | 13/58 [00:05<00:12,  3.58it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=34.51 GB):  24%|██▍       | 14/58 [00:05<00:12,  3.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=34.50 GB):  24%|██▍       | 14/58 [00:05<00:12,  3.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=34.50 GB):  26%|██▌       | 15/58 [00:05<00:10,  4.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=34.50 GB):  26%|██▌       | 15/58 [00:05<00:10,  4.24it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=34.50 GB):  28%|██▊       | 16/58 [00:05<00:08,  4.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=34.50 GB):  28%|██▊       | 16/58 [00:05<00:08,  4.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=34.50 GB):  29%|██▉       | 17/58 [00:06<00:07,  5.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=34.49 GB):  29%|██▉       | 17/58 [00:06<00:07,  5.62it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=34.49 GB):  31%|███       | 18/58 [00:06<00:06,  6.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=34.48 GB):  31%|███       | 18/58 [00:06<00:06,  6.42it/s]Capturing num tokens (num_tokens=1280 avail_mem=34.47 GB):  31%|███       | 18/58 [00:06<00:06,  6.42it/s]Capturing num tokens (num_tokens=1280 avail_mem=34.47 GB):  34%|███▍      | 20/58 [00:06<00:04,  8.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=34.45 GB):  34%|███▍      | 20/58 [00:06<00:04,  8.09it/s]

    Capturing num tokens (num_tokens=960 avail_mem=34.45 GB):  34%|███▍      | 20/58 [00:06<00:04,  8.09it/s] Capturing num tokens (num_tokens=960 avail_mem=34.45 GB):  38%|███▊      | 22/58 [00:06<00:03,  9.75it/s]Capturing num tokens (num_tokens=896 avail_mem=34.34 GB):  38%|███▊      | 22/58 [00:06<00:03,  9.75it/s]Capturing num tokens (num_tokens=832 avail_mem=34.16 GB):  38%|███▊      | 22/58 [00:06<00:03,  9.75it/s]

    Capturing num tokens (num_tokens=832 avail_mem=34.16 GB):  41%|████▏     | 24/58 [00:06<00:03, 11.12it/s]Capturing num tokens (num_tokens=768 avail_mem=33.96 GB):  41%|████▏     | 24/58 [00:06<00:03, 11.12it/s]Capturing num tokens (num_tokens=704 avail_mem=33.79 GB):  41%|████▏     | 24/58 [00:06<00:03, 11.12it/s]Capturing num tokens (num_tokens=704 avail_mem=33.79 GB):  45%|████▍     | 26/58 [00:06<00:02, 12.48it/s]Capturing num tokens (num_tokens=640 avail_mem=33.61 GB):  45%|████▍     | 26/58 [00:06<00:02, 12.48it/s]Capturing num tokens (num_tokens=576 avail_mem=33.43 GB):  45%|████▍     | 26/58 [00:06<00:02, 12.48it/s]

    Capturing num tokens (num_tokens=576 avail_mem=33.43 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.73it/s]Capturing num tokens (num_tokens=512 avail_mem=33.24 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.73it/s]Capturing num tokens (num_tokens=480 avail_mem=33.09 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.73it/s]Capturing num tokens (num_tokens=480 avail_mem=33.09 GB):  52%|█████▏    | 30/58 [00:06<00:01, 15.12it/s]Capturing num tokens (num_tokens=448 avail_mem=33.09 GB):  52%|█████▏    | 30/58 [00:06<00:01, 15.12it/s]Capturing num tokens (num_tokens=416 avail_mem=33.07 GB):  52%|█████▏    | 30/58 [00:06<00:01, 15.12it/s]

    Capturing num tokens (num_tokens=384 avail_mem=33.06 GB):  52%|█████▏    | 30/58 [00:06<00:01, 15.12it/s]Capturing num tokens (num_tokens=384 avail_mem=33.06 GB):  57%|█████▋    | 33/58 [00:07<00:01, 16.87it/s]Capturing num tokens (num_tokens=352 avail_mem=33.05 GB):  57%|█████▋    | 33/58 [00:07<00:01, 16.87it/s]Capturing num tokens (num_tokens=320 avail_mem=33.03 GB):  57%|█████▋    | 33/58 [00:07<00:01, 16.87it/s]Capturing num tokens (num_tokens=288 avail_mem=33.03 GB):  57%|█████▋    | 33/58 [00:07<00:01, 16.87it/s]Capturing num tokens (num_tokens=288 avail_mem=33.03 GB):  62%|██████▏   | 36/58 [00:07<00:01, 18.83it/s]Capturing num tokens (num_tokens=256 avail_mem=32.93 GB):  62%|██████▏   | 36/58 [00:07<00:01, 18.83it/s]

    Capturing num tokens (num_tokens=240 avail_mem=32.92 GB):  62%|██████▏   | 36/58 [00:07<00:01, 18.83it/s]Capturing num tokens (num_tokens=224 avail_mem=32.88 GB):  62%|██████▏   | 36/58 [00:07<00:01, 18.83it/s]Capturing num tokens (num_tokens=224 avail_mem=32.88 GB):  67%|██████▋   | 39/58 [00:07<00:00, 20.17it/s]Capturing num tokens (num_tokens=208 avail_mem=32.86 GB):  67%|██████▋   | 39/58 [00:07<00:00, 20.17it/s]Capturing num tokens (num_tokens=192 avail_mem=32.33 GB):  67%|██████▋   | 39/58 [00:07<00:00, 20.17it/s]Capturing num tokens (num_tokens=176 avail_mem=32.33 GB):  67%|██████▋   | 39/58 [00:07<00:00, 20.17it/s]

    Capturing num tokens (num_tokens=176 avail_mem=32.33 GB):  72%|███████▏  | 42/58 [00:07<00:00, 20.83it/s]Capturing num tokens (num_tokens=160 avail_mem=31.27 GB):  72%|███████▏  | 42/58 [00:07<00:00, 20.83it/s]Capturing num tokens (num_tokens=144 avail_mem=30.46 GB):  72%|███████▏  | 42/58 [00:07<00:00, 20.83it/s]Capturing num tokens (num_tokens=128 avail_mem=30.44 GB):  72%|███████▏  | 42/58 [00:07<00:00, 20.83it/s]Capturing num tokens (num_tokens=128 avail_mem=30.44 GB):  78%|███████▊  | 45/58 [00:07<00:00, 21.78it/s]Capturing num tokens (num_tokens=112 avail_mem=30.43 GB):  78%|███████▊  | 45/58 [00:07<00:00, 21.78it/s]Capturing num tokens (num_tokens=96 avail_mem=30.42 GB):  78%|███████▊  | 45/58 [00:07<00:00, 21.78it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=30.41 GB):  78%|███████▊  | 45/58 [00:07<00:00, 21.78it/s]Capturing num tokens (num_tokens=80 avail_mem=30.41 GB):  83%|████████▎ | 48/58 [00:07<00:00, 23.09it/s]Capturing num tokens (num_tokens=64 avail_mem=30.41 GB):  83%|████████▎ | 48/58 [00:07<00:00, 23.09it/s]Capturing num tokens (num_tokens=48 avail_mem=30.40 GB):  83%|████████▎ | 48/58 [00:07<00:00, 23.09it/s]Capturing num tokens (num_tokens=32 avail_mem=30.39 GB):  83%|████████▎ | 48/58 [00:07<00:00, 23.09it/s]Capturing num tokens (num_tokens=32 avail_mem=30.39 GB):  88%|████████▊ | 51/58 [00:07<00:00, 24.37it/s]Capturing num tokens (num_tokens=28 avail_mem=30.39 GB):  88%|████████▊ | 51/58 [00:07<00:00, 24.37it/s]Capturing num tokens (num_tokens=24 avail_mem=30.38 GB):  88%|████████▊ | 51/58 [00:07<00:00, 24.37it/s]

    Capturing num tokens (num_tokens=20 avail_mem=30.37 GB):  88%|████████▊ | 51/58 [00:07<00:00, 24.37it/s]Capturing num tokens (num_tokens=20 avail_mem=30.37 GB):  93%|█████████▎| 54/58 [00:07<00:00, 25.38it/s]Capturing num tokens (num_tokens=16 avail_mem=30.37 GB):  93%|█████████▎| 54/58 [00:07<00:00, 25.38it/s]Capturing num tokens (num_tokens=12 avail_mem=30.36 GB):  93%|█████████▎| 54/58 [00:07<00:00, 25.38it/s]Capturing num tokens (num_tokens=8 avail_mem=30.35 GB):  93%|█████████▎| 54/58 [00:07<00:00, 25.38it/s] Capturing num tokens (num_tokens=8 avail_mem=30.35 GB):  98%|█████████▊| 57/58 [00:07<00:00, 26.14it/s]Capturing num tokens (num_tokens=4 avail_mem=30.34 GB):  98%|█████████▊| 57/58 [00:07<00:00, 26.14it/s]Capturing num tokens (num_tokens=4 avail_mem=30.34 GB): 100%|██████████| 58/58 [00:08<00:00,  7.22it/s]


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
    Generated text: Berlin is the capital of Italy


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
    
    Generated text: Okay, so the user is asking for the information and population of the capital of France. I know that the capital is Paris. I should start by confirming that.
    
    Now, the population figure I remember is around 2 million. But wait, is that exact or an estimate? I think it's an estimate because populations can change. Maybe it's about 2.1 million. I should note that it's an approximate figure.
    
    Also, Paris is a major city, so it's not just the capital but also a very important metropolis. Including that in the response would be helpful for context.
    
    I should structure this information in a JSON format as the user requested. So, I'll create an object with "city" as "Paris", "population" as the number, and a "description" explaining it's a major city.
    
    Let me double-check the population to make sure I'm accurate. Yes, sources say approximately 2.16 million as of recent estimates. So I'll include that.
    
    I think that covers the user's query. I don't need to add anything extra unless they ask for more details later.
    </think>
    
    ```json
    {
      "city": "Paris",
      "population": "Approximately 2.16 million",
      "description": "Paris is the capital city of France and one of the most important cities in the world."
    }
    ```



```python
llm.shutdown()
```
