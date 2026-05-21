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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.31s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.45s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.43s/it]


    2026-05-21 02:05:05,241 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-21 02:05:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:11,  5.47s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:11,  5.47s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:29,  2.66s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:29,  2.66s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:35,  1.74s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:35,  1.74s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:09,  1.29s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:09,  1.29s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:53,  1.01s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:53,  1.01s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:08<00:43,  1.19it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:08<00:43,  1.19it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:35,  1.42it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:35,  1.42it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:09<00:30,  1.64it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:09<00:30,  1.64it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:25,  1.89it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:25,  1.89it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:22,  2.17it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:22,  2.17it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:10<00:19,  2.42it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:10<00:19,  2.42it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:10<00:17,  2.70it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:10<00:17,  2.70it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:10<00:14,  3.01it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:10<00:14,  3.01it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:10<00:13,  3.28it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:10<00:13,  3.28it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:11<00:11,  3.67it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:11<00:11,  3.67it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:11<00:10,  4.10it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:11<00:10,  4.10it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:11<00:09,  4.55it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:11<00:09,  4.55it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:11<00:07,  5.08it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:11<00:07,  5.08it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:11<00:06,  5.66it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:11<00:06,  5.66it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:11<00:06,  6.29it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:11<00:06,  6.29it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:12<00:06,  6.29it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:12<00:04,  8.02it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:12<00:04,  8.02it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:12<00:04,  8.02it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:12<00:03,  9.66it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:12<00:03,  9.66it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:12<00:03,  9.66it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:12<00:02, 11.33it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:12<00:02, 11.33it/s]

    Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:12<00:02, 11.33it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:12<00:02, 12.88it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:12<00:02, 12.88it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:12<00:02, 12.88it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:12<00:02, 12.88it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:12<00:01, 15.52it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:12<00:01, 15.52it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:12<00:01, 15.52it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:12<00:01, 15.52it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:12<00:01, 17.52it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:12<00:01, 17.52it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:12<00:01, 17.52it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:12<00:01, 17.52it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:12<00:01, 19.76it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:12<00:01, 19.76it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:12<00:01, 19.76it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:12<00:01, 19.76it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:12<00:01, 19.76it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:12<00:00, 23.47it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:12<00:00, 23.47it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:13<00:00, 23.47it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:13<00:00, 23.47it/s]

    Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:13<00:00, 24.94it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:13<00:00, 24.94it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:13<00:00, 24.94it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:13<00:00, 24.94it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:13<00:00, 24.94it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:13<00:00, 27.52it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:13<00:00, 27.52it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:13<00:00, 27.52it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:13<00:00, 27.52it/s]

    Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:13<00:00, 27.52it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:13<00:00, 29.81it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:13<00:00, 29.81it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:13<00:00, 29.81it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:13<00:00, 29.81it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:13<00:00, 29.81it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:13<00:00, 32.28it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:13<00:00, 32.28it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:13<00:00, 32.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:13<00:00,  4.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.70 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.70 GB):   3%|▎         | 2/58 [00:01<00:49,  1.13it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   3%|▎         | 2/58 [00:01<00:49,  1.13it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   5%|▌         | 3/58 [00:02<00:45,  1.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.82 GB):   5%|▌         | 3/58 [00:02<00:45,  1.22it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.82 GB):   7%|▋         | 4/58 [00:03<00:41,  1.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.90 GB):   7%|▋         | 4/58 [00:03<00:41,  1.31it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.90 GB):   9%|▊         | 5/58 [00:03<00:38,  1.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.23 GB):   9%|▊         | 5/58 [00:03<00:38,  1.39it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.23 GB):  10%|█         | 6/58 [00:04<00:34,  1.52it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.36 GB):  10%|█         | 6/58 [00:04<00:34,  1.52it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.36 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.49 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.65it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=25.49 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.16 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.82it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.16 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.69 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.00it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=25.69 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.16it/s]Capturing num tokens (num_tokens=3584 avail_mem=25.75 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.16it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=25.75 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=25.81 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.36it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=25.81 GB):  21%|██        | 12/58 [00:06<00:17,  2.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=25.86 GB):  21%|██        | 12/58 [00:06<00:17,  2.58it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=25.86 GB):  22%|██▏       | 13/58 [00:07<00:16,  2.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.59 GB):  22%|██▏       | 13/58 [00:07<00:16,  2.81it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.59 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.41 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.11it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.41 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.43it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.35 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.43it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=26.35 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.37 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.37 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.21it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.40 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.21it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=26.40 GB):  31%|███       | 18/58 [00:08<00:08,  4.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.42 GB):  31%|███       | 18/58 [00:08<00:08,  4.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.42 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.43 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.23it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=26.43 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.45 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.87it/s]Capturing num tokens (num_tokens=960 avail_mem=26.45 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.87it/s] Capturing num tokens (num_tokens=960 avail_mem=26.45 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.41it/s]Capturing num tokens (num_tokens=896 avail_mem=26.44 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.41it/s]

    Capturing num tokens (num_tokens=832 avail_mem=26.44 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.41it/s]Capturing num tokens (num_tokens=832 avail_mem=26.44 GB):  41%|████▏     | 24/58 [00:08<00:03,  8.54it/s]Capturing num tokens (num_tokens=768 avail_mem=26.43 GB):  41%|████▏     | 24/58 [00:08<00:03,  8.54it/s]Capturing num tokens (num_tokens=704 avail_mem=26.43 GB):  41%|████▏     | 24/58 [00:08<00:03,  8.54it/s]

    Capturing num tokens (num_tokens=704 avail_mem=26.43 GB):  45%|████▍     | 26/58 [00:08<00:03,  9.76it/s]Capturing num tokens (num_tokens=640 avail_mem=26.45 GB):  45%|████▍     | 26/58 [00:08<00:03,  9.76it/s]Capturing num tokens (num_tokens=576 avail_mem=26.44 GB):  45%|████▍     | 26/58 [00:08<00:03,  9.76it/s]Capturing num tokens (num_tokens=576 avail_mem=26.44 GB):  48%|████▊     | 28/58 [00:08<00:02, 11.01it/s]Capturing num tokens (num_tokens=512 avail_mem=26.43 GB):  48%|████▊     | 28/58 [00:08<00:02, 11.01it/s]Capturing num tokens (num_tokens=480 avail_mem=26.43 GB):  48%|████▊     | 28/58 [00:08<00:02, 11.01it/s]

    Capturing num tokens (num_tokens=480 avail_mem=26.43 GB):  52%|█████▏    | 30/58 [00:09<00:02, 12.12it/s]Capturing num tokens (num_tokens=448 avail_mem=26.42 GB):  52%|█████▏    | 30/58 [00:09<00:02, 12.12it/s]Capturing num tokens (num_tokens=416 avail_mem=26.41 GB):  52%|█████▏    | 30/58 [00:09<00:02, 12.12it/s]Capturing num tokens (num_tokens=416 avail_mem=26.41 GB):  55%|█████▌    | 32/58 [00:09<00:02, 12.77it/s]Capturing num tokens (num_tokens=384 avail_mem=26.45 GB):  55%|█████▌    | 32/58 [00:09<00:02, 12.77it/s]Capturing num tokens (num_tokens=352 avail_mem=26.43 GB):  55%|█████▌    | 32/58 [00:09<00:02, 12.77it/s]

    Capturing num tokens (num_tokens=352 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.03it/s]Capturing num tokens (num_tokens=320 avail_mem=26.44 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.03it/s]Capturing num tokens (num_tokens=288 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.03it/s]Capturing num tokens (num_tokens=288 avail_mem=26.43 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.21it/s]Capturing num tokens (num_tokens=256 avail_mem=26.42 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.21it/s]Capturing num tokens (num_tokens=240 avail_mem=26.41 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.21it/s]

    Capturing num tokens (num_tokens=240 avail_mem=26.41 GB):  66%|██████▌   | 38/58 [00:09<00:01, 16.17it/s]Capturing num tokens (num_tokens=224 avail_mem=26.37 GB):  66%|██████▌   | 38/58 [00:09<00:01, 16.17it/s]Capturing num tokens (num_tokens=208 avail_mem=26.39 GB):  66%|██████▌   | 38/58 [00:09<00:01, 16.17it/s]Capturing num tokens (num_tokens=192 avail_mem=26.38 GB):  66%|██████▌   | 38/58 [00:09<00:01, 16.17it/s]Capturing num tokens (num_tokens=192 avail_mem=26.38 GB):  71%|███████   | 41/58 [00:09<00:00, 17.61it/s]Capturing num tokens (num_tokens=176 avail_mem=26.37 GB):  71%|███████   | 41/58 [00:09<00:00, 17.61it/s]

    Capturing num tokens (num_tokens=160 avail_mem=26.36 GB):  71%|███████   | 41/58 [00:09<00:00, 17.61it/s]Capturing num tokens (num_tokens=160 avail_mem=26.36 GB):  74%|███████▍  | 43/58 [00:09<00:00, 17.61it/s]Capturing num tokens (num_tokens=144 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:00, 17.61it/s]Capturing num tokens (num_tokens=128 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:00, 17.61it/s]Capturing num tokens (num_tokens=128 avail_mem=26.35 GB):  78%|███████▊  | 45/58 [00:09<00:00, 18.09it/s]Capturing num tokens (num_tokens=112 avail_mem=26.35 GB):  78%|███████▊  | 45/58 [00:09<00:00, 18.09it/s]

    Capturing num tokens (num_tokens=96 avail_mem=26.33 GB):  78%|███████▊  | 45/58 [00:09<00:00, 18.09it/s] Capturing num tokens (num_tokens=96 avail_mem=26.33 GB):  81%|████████  | 47/58 [00:09<00:00, 18.58it/s]Capturing num tokens (num_tokens=80 avail_mem=26.32 GB):  81%|████████  | 47/58 [00:09<00:00, 18.58it/s]Capturing num tokens (num_tokens=64 avail_mem=26.31 GB):  81%|████████  | 47/58 [00:10<00:00, 18.58it/s]Capturing num tokens (num_tokens=64 avail_mem=26.31 GB):  84%|████████▍ | 49/58 [00:10<00:00, 18.75it/s]Capturing num tokens (num_tokens=48 avail_mem=26.30 GB):  84%|████████▍ | 49/58 [00:10<00:00, 18.75it/s]

    Capturing num tokens (num_tokens=32 avail_mem=26.29 GB):  84%|████████▍ | 49/58 [00:10<00:00, 18.75it/s]Capturing num tokens (num_tokens=28 avail_mem=26.28 GB):  84%|████████▍ | 49/58 [00:10<00:00, 18.75it/s]Capturing num tokens (num_tokens=28 avail_mem=26.28 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.12it/s]Capturing num tokens (num_tokens=24 avail_mem=26.26 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.12it/s]Capturing num tokens (num_tokens=20 avail_mem=26.26 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.12it/s]Capturing num tokens (num_tokens=16 avail_mem=26.25 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.12it/s]

    Capturing num tokens (num_tokens=16 avail_mem=26.25 GB):  95%|█████████▍| 55/58 [00:10<00:00, 20.02it/s]Capturing num tokens (num_tokens=12 avail_mem=26.24 GB):  95%|█████████▍| 55/58 [00:10<00:00, 20.02it/s]Capturing num tokens (num_tokens=8 avail_mem=26.23 GB):  95%|█████████▍| 55/58 [00:10<00:00, 20.02it/s] Capturing num tokens (num_tokens=8 avail_mem=26.23 GB):  98%|█████████▊| 57/58 [00:10<00:00, 19.94it/s]Capturing num tokens (num_tokens=4 avail_mem=26.22 GB):  98%|█████████▊| 57/58 [00:10<00:00, 19.94it/s]Capturing num tokens (num_tokens=4 avail_mem=26.22 GB): 100%|██████████| 58/58 [00:10<00:00,  5.51it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, I need to figure out how to get the current date and time in New York and the weather there using the allowed functions. The user mentioned they're in New York, so I should use the functions get_current_weather and get_current_date.<br><br>First, for the date, I'll use get_current_date with the timezone parameter set to 'America/New_York'. That should give me the current date and time in that timezone.<br><br>Next, for the weather, I need to provide the city and state. The city is New York, and the state abbreviation is NY. I'll call get_current_weather with city 'New York', state 'NY', and maybe set the unit to Celsius since that's a common choice, but the function allows Fahrenheit too. I'll include unit: 'celsius' in the parameters.<br><br>I should make sure each function call is on its own line and in the correct format, with the function name and parameters properly enclosed in the start and end tags. Also, I need to remember to add the sources where I got the information from, even if it's just the function descriptions.<br><br>So putting it together, I'll send two separate function calls: one for the date and one for the weather, each with their respective parameters.<br><br><br>content: <br><br><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "celsius"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'c1933aa057614f8888b6f9344497656f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.239222322590649, 'response_sent_to_client_ts': 1779329175.2794585}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': 'e4ee5ba87f7f47f3b69bcc4c657ddb3c', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 21.67355158366263, 'response_sent_to_client_ts': 1779329196.963646}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '20a66e39c9784cd88255b26df3af74e5', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11334054730832577, 'response_sent_to_client_ts': 1779329197.1116054}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'd32f995408c84b9bbbd4045c3c6f14f1', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11313981097191572, 'response_sent_to_client_ts': 1779329197.111621}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '873818aa21924cd68d43203dbd6472a3', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11309737619012594, 'response_sent_to_client_ts': 1779329197.1116254}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'abfcd9d769a6426aab2cce12a6ba367e', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.55420424323529, 'response_sent_to_client_ts': 1779329216.6738975}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user just asked for the population of the capital of France, which they want in JSON format. That seems straightforward. But wait, is that all they\'re looking for? Maybe they need more information about the capital city itself.\n\nHmm, the capital of France is Paris. I should check the latest population numbers. I know Paris is a major city, so its population must be quite large. Let me think, I believe the current estimate is around 2.1 million people. But to be accurate, I should probably look up the most recent data.\n\nI wonder if the exact number changes every year. Yeah, populations can fluctuate based on births, deaths, and migrations. So, it\'s best to provide a recent figure or a range if the exact number isn\'t fixed. Maybe 2.1 to 2.15 million? That sounds about right.\n\nWait, the user asked for information and population. So, I shouldn\'t just stop at the population. What other information should I include? Perhaps the city\'s location, some key landmarks, and maybe some notable facts or services. That would give a more comprehensive answer.\n\nI should structure the JSON properly. Maybe start with "name" and "population", then add sections for "description", "location", "infrastructure", and "factoids". That way, it\'s organized and easy to read. Each section can have relevant details, making the information more useful for the user.\n\nOh, I should also consider if the user might have specific reasons for this query. They could be a student working on a project, a traveler looking for info, or maybe someone planning an event. Providing thorough information might cater to their needs better. Including some interesting facts could spark their interest and make the response more engaging.\n\nAlright, putting it all together: name Paris, population around 2.1 million, location in Île-de-France, some landmarks like the Eiffel Tower and Notre-Dame, Major institutions like the Louvre, climate info, notable people such as Jean Jacques Rousseau, and some fun facts about transportation and museums.\n\nI think that covers it. Now, I\'ll format it all into a JSON structure, making sure each key is clear and the information is accurate. That way, the user gets a well-rounded and informative response in the format they requested.\n</think>\n\nSure! Here\'s the information about the capital of France in JSON format:\n\n```json\n{\n  "name": "Paris",\n  "population": "2,147,325",\n  "description": "Paris is the capital city and the most populous city in France. It is known for its rich history, culture, and landmarks.",\n  "location": "The capital is located in Île-de-France, which is an island in the Seine River.",\n  "infrastructure": "Paris is a major transportation hub with an extensive public transit system, including the Métro and RER networks.",\n  "factoids": {\n    "population": "The population of Paris is approximately 2.1 million (as of 2023).",\n    "landmarks": "The city is famous for landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.",\n    "languages": "The official languages are French and, to a lesser extent, English.",\n    "climate": "Paris has a continental climate, with four distinct seasons and cold winters with short days in winter and long, sunny days in summer.",\n    "renown": "Paris is renowned for its cuisine, fashion, and the arts, including the works of famous artists and writers."\n  }\n}\n```\n\nLet me know if you\'d like more details!', 'output_ids': [32313, 11, 773, 279, 1196, 1101, 4588, 369, 279, 7042, 315, 279, 6722, 315, 9625, 11, 892, 807, 1366, 304, 4718, 3561, 13, 2938, 4977, 30339, 13, 1988, 3783, 11, 374, 429, 678, 807, 2299, 3330, 369, 30, 10696, 807, 1184, 803, 1995, 911, 279, 6722, 3283, 5086, 382, 80022, 11, 279, 6722, 315, 9625, 374, 12095, 13, 358, 1265, 1779, 279, 5535, 7042, 5109, 13, 358, 1414, 12095, 374, 264, 3598, 3283, 11, 773, 1181, 7042, 1969, 387, 5008, 3460, 13, 6771, 752, 1744, 11, 358, 4411, 279, 1482, 16045, 374, 2163, 220, 17, 13, 16, 3526, 1251, 13, 1988, 311, 387, 13382, 11, 358, 1265, 4658, 1401, 705, 279, 1429, 3213, 821, 382, 40, 5775, 421, 279, 4734, 1372, 4344, 1449, 1042, 13, 21607, 11, 21910, 646, 38288, 6292, 3118, 389, 65232, 11, 16375, 11, 323, 17063, 13, 2055, 11, 432, 594, 1850, 311, 3410, 264, 3213, 7071, 476, 264, 2088, 421, 279, 4734, 1372, 4436, 944, 8356, 13, 10696, 220, 17, 13, 16, 311, 220, 17, 13, 16, 20, 3526, 30, 2938, 10362, 911, 1290, 382, 14190, 11, 279, 1196, 4588, 369, 1995, 323, 7042, 13, 2055, 11, 358, 13133, 944, 1101, 2936, 518, 279, 7042, 13, 3555, 1008, 1995, 1265, 358, 2924, 30, 18765, 279, 3283, 594, 3728, 11, 1045, 1376, 59924, 11, 323, 7196, 1045, 27190, 13064, 476, 3516, 13, 2938, 1035, 2968, 264, 803, 15817, 4226, 382, 40, 1265, 5944, 279, 4718, 10277, 13, 10696, 1191, 448, 330, 606, 1, 323, 330, 44441, 497, 1221, 912, 14158, 369, 330, 4684, 497, 330, 2527, 497, 330, 13573, 10314, 497, 323, 330, 33110, 16960, 3263, 2938, 1616, 11, 432, 594, 16645, 323, 4135, 311, 1349, 13, 8886, 3772, 646, 614, 9760, 3565, 11, 3259, 279, 1995, 803, 5390, 369, 279, 1196, 382, 11908, 11, 358, 1265, 1083, 2908, 421, 279, 1196, 2578, 614, 3151, 7966, 369, 419, 3239, 13, 2379, 1410, 387, 264, 5458, 3238, 389, 264, 2390, 11, 264, 62765, 3330, 369, 3546, 11, 476, 7196, 4325, 9115, 458, 1538, 13, 80100, 17423, 1995, 2578, 27968, 311, 862, 3880, 2664, 13, 55121, 1045, 7040, 13064, 1410, 15186, 862, 2734, 323, 1281, 279, 2033, 803, 22570, 382, 71486, 11, 10687, 432, 678, 3786, 25, 829, 12095, 11, 7042, 2163, 220, 17, 13, 16, 3526, 11, 3728, 304, 59108, 273, 6810, 7276, 34106, 11, 1045, 59924, 1075, 279, 468, 3092, 301, 21938, 323, 43464, 9420, 373, 11, 17121, 14336, 1075, 279, 9729, 48506, 11, 9977, 3546, 11, 27190, 1251, 1741, 438, 19685, 63750, 97889, 2863, 11, 323, 1045, 2464, 13064, 911, 17903, 323, 50577, 382, 40, 1744, 429, 14521, 432, 13, 4695, 11, 358, 3278, 3561, 432, 678, 1119, 264, 4718, 5944, 11, 3259, 2704, 1817, 1376, 374, 2797, 323, 279, 1995, 374, 13382, 13, 2938, 1616, 11, 279, 1196, 5221, 264, 1632, 64218, 323, 38219, 2033, 304, 279, 3561, 807, 11223, 624, 151649, 271, 39814, 0, 5692, 594, 279, 1995, 911, 279, 6722, 315, 9625, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 330, 17, 11, 16, 19, 22, 11, 18, 17, 20, 756, 220, 330, 4684, 788, 330, 59604, 374, 279, 6722, 3283, 323, 279, 1429, 94451, 3283, 304, 9625, 13, 1084, 374, 3881, 369, 1181, 9080, 3840, 11, 7674, 11, 323, 59924, 10346, 220, 330, 2527, 788, 330, 785, 6722, 374, 7407, 304, 59108, 273, 6810, 7276, 34106, 11, 892, 374, 458, 12922, 304, 279, 1345, 482, 10948, 10346, 220, 330, 13573, 10314, 788, 330, 59604, 374, 264, 3598, 17903, 18719, 448, 458, 16376, 584, 24065, 1849, 11, 2670, 279, 61258, 299, 323, 431, 640, 14155, 10346, 220, 330, 33110, 16960, 788, 341, 262, 330, 44441, 788, 330, 785, 7042, 315, 12095, 374, 13187, 220, 17, 13, 16, 3526, 320, 300, 315, 220, 17, 15, 17, 18, 568, 756, 262, 330, 1933, 15544, 788, 330, 785, 3283, 374, 11245, 369, 59924, 1741, 438, 279, 468, 3092, 301, 21938, 11, 43464, 9420, 373, 56729, 11, 323, 279, 9729, 48506, 16328, 10346, 262, 330, 42824, 788, 330, 785, 3946, 15459, 525, 8585, 323, 11, 311, 264, 31315, 12818, 11, 6364, 10346, 262, 330, 93774, 788, 330, 59604, 702, 264, 57536, 9977, 11, 448, 3040, 12460, 15584, 323, 9255, 84982, 448, 2805, 2849, 304, 12406, 323, 1293, 11, 39698, 2849, 304, 7324, 10346, 262, 330, 1440, 779, 788, 330, 59604, 374, 35948, 369, 1181, 35005, 11, 11153, 11, 323, 279, 18560, 11, 2670, 279, 4278, 315, 11245, 13511, 323, 16093, 10040, 220, 456, 532, 13874, 19324, 10061, 752, 1414, 421, 498, 4172, 1075, 803, 3565, 0, 151643], 'meta_info': {'id': '9ab67cafd32f4ad7958152cdde8762dc', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 477, 'completion_tokens': 761, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 6.991718551144004, 'response_sent_to_client_ts': 1779329223.674971}}</strong>



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

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.37s/it]


    2026-05-21 02:07:20,563 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-21 02:07:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:03,  5.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:03,  5.33s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:09,  2.32s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:09,  2.32s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:14,  1.35s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:14,  1.35s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:49,  1.09it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:49,  1.09it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:24,  2.09it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:24,  2.09it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.75it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.75it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.47it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.47it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.32it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.32it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.01it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.01it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.01it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.16it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.16it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:05,  7.47it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:05,  7.47it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:07<00:05,  7.47it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:04,  9.50it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:04,  9.50it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:07<00:04,  9.50it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:07<00:04,  9.50it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:07<00:03, 12.77it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:07<00:03, 12.77it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:07<00:03, 12.77it/s]

    Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:07<00:03, 12.77it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:07<00:03, 12.77it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:01, 17.85it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:01, 17.85it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:01, 17.85it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:07<00:01, 17.85it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:07<00:01, 17.85it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:07<00:01, 17.85it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:07<00:01, 24.83it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:07<00:01, 24.83it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:07<00:01, 24.83it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:07<00:01, 24.83it/s]

    Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:07<00:01, 24.83it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:07<00:01, 24.83it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:07<00:01, 24.83it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:07<00:00, 33.04it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:07<00:00, 33.04it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:07<00:00, 33.04it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:07<00:00, 33.04it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:07<00:00, 33.04it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:07<00:00, 33.04it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:07<00:00, 33.04it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:07<00:00, 33.04it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:07<00:00, 42.18it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:07<00:00, 42.18it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:07<00:00, 42.18it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:07<00:00, 42.18it/s]

    Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:07<00:00, 42.18it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:07<00:00, 42.18it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:07<00:00, 42.18it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:07<00:00, 42.18it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:07<00:00, 49.20it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:07<00:00, 49.20it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:07<00:00, 49.20it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:07<00:00, 49.20it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:08<00:00, 49.20it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:08<00:00, 49.20it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:08<00:00, 49.20it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:08<00:00, 49.20it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:08<00:00, 49.20it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:08<00:00, 49.20it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:08<00:00, 59.25it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:08<00:00, 59.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.96 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.96 GB):   2%|▏         | 1/58 [00:00<00:16,  3.39it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.92 GB):   2%|▏         | 1/58 [00:00<00:16,  3.39it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.92 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=24.92 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=24.92 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=24.92 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=24.92 GB):   7%|▋         | 4/58 [00:01<00:13,  3.96it/s]Capturing num tokens (num_tokens=6144 avail_mem=24.90 GB):   7%|▋         | 4/58 [00:01<00:13,  3.96it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=24.90 GB):   9%|▊         | 5/58 [00:01<00:13,  4.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=24.90 GB):   9%|▊         | 5/58 [00:01<00:13,  4.06it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=24.90 GB):  10%|█         | 6/58 [00:01<00:15,  3.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=24.88 GB):  10%|█         | 6/58 [00:01<00:15,  3.45it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=24.88 GB):  12%|█▏        | 7/58 [00:01<00:15,  3.35it/s]Capturing num tokens (num_tokens=4608 avail_mem=24.88 GB):  12%|█▏        | 7/58 [00:01<00:15,  3.35it/s]Capturing num tokens (num_tokens=4608 avail_mem=24.88 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=24.88 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.74it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=24.88 GB):  16%|█▌        | 9/58 [00:02<00:13,  3.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.99 GB):  16%|█▌        | 9/58 [00:02<00:13,  3.61it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.99 GB):  17%|█▋        | 10/58 [00:02<00:13,  3.56it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.99 GB):  17%|█▋        | 10/58 [00:02<00:13,  3.56it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.99 GB):  19%|█▉        | 11/58 [00:03<00:13,  3.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.99 GB):  19%|█▉        | 11/58 [00:03<00:13,  3.61it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=43.99 GB):  21%|██        | 12/58 [00:03<00:12,  3.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.98 GB):  21%|██        | 12/58 [00:03<00:12,  3.80it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=43.98 GB):  22%|██▏       | 13/58 [00:03<00:11,  4.02it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.98 GB):  22%|██▏       | 13/58 [00:03<00:11,  4.02it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.98 GB):  24%|██▍       | 14/58 [00:03<00:10,  4.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.98 GB):  24%|██▍       | 14/58 [00:03<00:10,  4.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.98 GB):  26%|██▌       | 15/58 [00:03<00:09,  4.47it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.98 GB):  26%|██▌       | 15/58 [00:03<00:09,  4.47it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=43.98 GB):  28%|██▊       | 16/58 [00:04<00:08,  4.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.97 GB):  28%|██▊       | 16/58 [00:04<00:08,  4.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.97 GB):  29%|██▉       | 17/58 [00:04<00:08,  5.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.97 GB):  29%|██▉       | 17/58 [00:04<00:08,  5.11it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=43.97 GB):  31%|███       | 18/58 [00:04<00:07,  5.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.97 GB):  31%|███       | 18/58 [00:04<00:07,  5.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.97 GB):  31%|███       | 18/58 [00:04<00:07,  5.49it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=43.97 GB):  34%|███▍      | 20/58 [00:04<00:05,  6.79it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.96 GB):  34%|███▍      | 20/58 [00:04<00:05,  6.79it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.96 GB):  36%|███▌      | 21/58 [00:04<00:05,  7.32it/s]Capturing num tokens (num_tokens=960 avail_mem=43.95 GB):  36%|███▌      | 21/58 [00:04<00:05,  7.32it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=43.95 GB):  36%|███▌      | 21/58 [00:04<00:05,  7.32it/s]Capturing num tokens (num_tokens=896 avail_mem=43.95 GB):  40%|███▉      | 23/58 [00:04<00:04,  8.24it/s]Capturing num tokens (num_tokens=832 avail_mem=43.95 GB):  40%|███▉      | 23/58 [00:04<00:04,  8.24it/s]Capturing num tokens (num_tokens=768 avail_mem=43.94 GB):  40%|███▉      | 23/58 [00:04<00:04,  8.24it/s]

    Capturing num tokens (num_tokens=768 avail_mem=43.94 GB):  43%|████▎     | 25/58 [00:05<00:03,  8.87it/s]Capturing num tokens (num_tokens=704 avail_mem=43.94 GB):  43%|████▎     | 25/58 [00:05<00:03,  8.87it/s]Capturing num tokens (num_tokens=640 avail_mem=43.93 GB):  43%|████▎     | 25/58 [00:05<00:03,  8.87it/s]Capturing num tokens (num_tokens=640 avail_mem=43.93 GB):  47%|████▋     | 27/58 [00:05<00:03, 10.17it/s]Capturing num tokens (num_tokens=576 avail_mem=43.93 GB):  47%|████▋     | 27/58 [00:05<00:03, 10.17it/s]Capturing num tokens (num_tokens=512 avail_mem=43.92 GB):  47%|████▋     | 27/58 [00:05<00:03, 10.17it/s]

    Capturing num tokens (num_tokens=480 avail_mem=43.92 GB):  47%|████▋     | 27/58 [00:05<00:03, 10.17it/s]Capturing num tokens (num_tokens=448 avail_mem=43.92 GB):  47%|████▋     | 27/58 [00:05<00:03, 10.17it/s]Capturing num tokens (num_tokens=448 avail_mem=43.92 GB):  53%|█████▎    | 31/58 [00:05<00:01, 15.18it/s]Capturing num tokens (num_tokens=416 avail_mem=43.92 GB):  53%|█████▎    | 31/58 [00:05<00:01, 15.18it/s]Capturing num tokens (num_tokens=384 avail_mem=43.91 GB):  53%|█████▎    | 31/58 [00:05<00:01, 15.18it/s]Capturing num tokens (num_tokens=352 avail_mem=43.91 GB):  53%|█████▎    | 31/58 [00:05<00:01, 15.18it/s]Capturing num tokens (num_tokens=320 avail_mem=43.90 GB):  53%|█████▎    | 31/58 [00:05<00:01, 15.18it/s]Capturing num tokens (num_tokens=320 avail_mem=43.90 GB):  60%|██████    | 35/58 [00:05<00:01, 19.99it/s]Capturing num tokens (num_tokens=288 avail_mem=43.91 GB):  60%|██████    | 35/58 [00:05<00:01, 19.99it/s]

    Capturing num tokens (num_tokens=256 avail_mem=43.90 GB):  60%|██████    | 35/58 [00:05<00:01, 19.99it/s]Capturing num tokens (num_tokens=240 avail_mem=43.90 GB):  60%|██████    | 35/58 [00:05<00:01, 19.99it/s]Capturing num tokens (num_tokens=224 avail_mem=43.90 GB):  60%|██████    | 35/58 [00:05<00:01, 19.99it/s]Capturing num tokens (num_tokens=224 avail_mem=43.90 GB):  67%|██████▋   | 39/58 [00:05<00:00, 24.10it/s]Capturing num tokens (num_tokens=208 avail_mem=43.89 GB):  67%|██████▋   | 39/58 [00:05<00:00, 24.10it/s]Capturing num tokens (num_tokens=192 avail_mem=43.89 GB):  67%|██████▋   | 39/58 [00:05<00:00, 24.10it/s]Capturing num tokens (num_tokens=176 avail_mem=43.89 GB):  67%|██████▋   | 39/58 [00:05<00:00, 24.10it/s]Capturing num tokens (num_tokens=160 avail_mem=43.88 GB):  67%|██████▋   | 39/58 [00:05<00:00, 24.10it/s]Capturing num tokens (num_tokens=160 avail_mem=43.88 GB):  74%|███████▍  | 43/58 [00:05<00:00, 27.80it/s]Capturing num tokens (num_tokens=144 avail_mem=43.88 GB):  74%|███████▍  | 43/58 [00:05<00:00, 27.80it/s]

    Capturing num tokens (num_tokens=128 avail_mem=43.88 GB):  74%|███████▍  | 43/58 [00:05<00:00, 27.80it/s]Capturing num tokens (num_tokens=112 avail_mem=43.88 GB):  74%|███████▍  | 43/58 [00:05<00:00, 27.80it/s]Capturing num tokens (num_tokens=96 avail_mem=43.87 GB):  74%|███████▍  | 43/58 [00:05<00:00, 27.80it/s] Capturing num tokens (num_tokens=96 avail_mem=43.87 GB):  81%|████████  | 47/58 [00:05<00:00, 30.40it/s]Capturing num tokens (num_tokens=80 avail_mem=43.87 GB):  81%|████████  | 47/58 [00:05<00:00, 30.40it/s]Capturing num tokens (num_tokens=64 avail_mem=43.86 GB):  81%|████████  | 47/58 [00:05<00:00, 30.40it/s]Capturing num tokens (num_tokens=48 avail_mem=43.86 GB):  81%|████████  | 47/58 [00:05<00:00, 30.40it/s]Capturing num tokens (num_tokens=32 avail_mem=43.86 GB):  81%|████████  | 47/58 [00:05<00:00, 30.40it/s]Capturing num tokens (num_tokens=32 avail_mem=43.86 GB):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Capturing num tokens (num_tokens=28 avail_mem=43.86 GB):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]

    Capturing num tokens (num_tokens=24 avail_mem=43.85 GB):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Capturing num tokens (num_tokens=20 avail_mem=43.85 GB):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Capturing num tokens (num_tokens=16 avail_mem=43.84 GB):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Capturing num tokens (num_tokens=16 avail_mem=43.84 GB):  95%|█████████▍| 55/58 [00:05<00:00, 34.15it/s]Capturing num tokens (num_tokens=12 avail_mem=43.84 GB):  95%|█████████▍| 55/58 [00:05<00:00, 34.15it/s]Capturing num tokens (num_tokens=8 avail_mem=43.84 GB):  95%|█████████▍| 55/58 [00:06<00:00, 34.15it/s] Capturing num tokens (num_tokens=4 avail_mem=43.83 GB):  95%|█████████▍| 55/58 [00:06<00:00, 34.15it/s]Capturing num tokens (num_tokens=4 avail_mem=43.83 GB): 100%|██████████| 58/58 [00:06<00:00,  9.52it/s]


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
    
    Generated text: Okay, so I'm trying to figure out how to respond to the user's request. They asked for the information and population of the capital of France in JSON format. Hmm, let me think about this step by step.
    
    First, I know that the capital of France is Paris. That's pretty straightforward. Now, the user wants the population. I'm not exactly sure about the current number, but I think it's around 2 million. I remember reading somewhere that Paris has a large population, but I'm not certain about the exact figure. Maybe it's best to check that.
    
    Wait, the user specified to provide the information and population in JSON format. So I need to structure it correctly. JSON typically uses key-value pairs. So the keys could be "capital" and "population". The value for "capital" is "Paris". For "population", I should put the number, but I'm not 100% sure about the exact number. I think it's approximately 2.1 million as of 2023, but I'm not entirely sure if that's the latest figure. Maybe it's better to confirm that.
    
    But since I can't access the internet right now, I'll go with my best estimate. So, I'll put "2100000" as the population. Now, putting that into JSON, it should look like this:
    
    {
      "capital": "Paris",
      "population": 2100000
    }
    
    I think that's correct. I should make sure the syntax is right, with proper commas and quotes. Wait, in JSON, the keys should be in double quotes, and the values can be strings or numbers. Since population is a number, it should be a number type, not a string. So the above structure is accurate.
    
    Is there anything else the user might need? They just asked for the information and population, so I think this should suffice. Maybe they want more details, but the user didn't specify that. So, I'll stick with the JSON I've created.
    </think>
    
    ```json
    {
      "capital": "Paris",
      "population": 2100000
    }
    ```



```python
llm.shutdown()
```
