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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.18s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.14s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.15s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:05,  2.24s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:05,  2.24s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:11,  1.30s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:11,  1.30s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.21it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.21it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.88it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.88it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.62it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.62it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.48it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.48it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.48it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.18it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.18it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.18it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.76it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.76it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.76it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.35it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.35it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.35it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.27it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.27it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.27it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.27it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.35it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.35it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.35it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.35it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.35it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.63it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.63it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.63it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.63it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.63it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.63it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.40it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.40it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.40it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.40it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.40it/s]

    Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.40it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.40it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 34.77it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 34.77it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 34.77it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 34.77it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 34.77it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 34.77it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 34.77it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 34.77it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:07<00:00, 34.77it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 45.08it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 45.08it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:07<00:00, 45.08it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:07<00:00, 45.08it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:07<00:00, 45.08it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:07<00:00, 45.08it/s]

    Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:07<00:00, 45.08it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:07<00:00, 45.08it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:07<00:00, 45.08it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:07<00:00, 53.24it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:07<00:00, 53.24it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:07<00:00, 53.24it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:07<00:00, 53.24it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:07<00:00, 53.24it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:07<00:00, 53.24it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:07<00:00, 53.24it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:07<00:00, 53.24it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=38.53 GB):   2%|▏         | 1/58 [00:00<00:17,  3.25it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.49 GB):   2%|▏         | 1/58 [00:00<00:17,  3.25it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=38.49 GB):   3%|▎         | 2/58 [00:00<00:18,  3.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.49 GB):   3%|▎         | 2/58 [00:00<00:18,  3.04it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=38.49 GB):   5%|▌         | 3/58 [00:00<00:15,  3.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.49 GB):   5%|▌         | 3/58 [00:00<00:15,  3.45it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=38.49 GB):   7%|▋         | 4/58 [00:01<00:14,  3.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.49 GB):   7%|▋         | 4/58 [00:01<00:14,  3.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=38.49 GB):   9%|▊         | 5/58 [00:01<00:12,  4.18it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.49 GB):   9%|▊         | 5/58 [00:01<00:12,  4.18it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.49 GB):  10%|█         | 6/58 [00:01<00:11,  4.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.49 GB):  10%|█         | 6/58 [00:01<00:11,  4.61it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=38.49 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.49 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.49 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.49 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.50it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=38.49 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.49 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.49 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.49 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.49it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=38.49 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.49 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.49 GB):  21%|██        | 12/58 [00:02<00:05,  7.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.48 GB):  21%|██        | 12/58 [00:02<00:05,  7.68it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=38.48 GB):  21%|██        | 12/58 [00:02<00:05,  7.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.48 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.48 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.48 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.84it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=38.48 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.48 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.48 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.48 GB):  31%|███       | 18/58 [00:02<00:03, 11.86it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.47 GB):  31%|███       | 18/58 [00:02<00:03, 11.86it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.47 GB):  31%|███       | 18/58 [00:02<00:03, 11.86it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=38.46 GB):  31%|███       | 18/58 [00:02<00:03, 11.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.46 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.72it/s]Capturing num tokens (num_tokens=960 avail_mem=38.45 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.72it/s] Capturing num tokens (num_tokens=896 avail_mem=38.45 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.72it/s]Capturing num tokens (num_tokens=832 avail_mem=38.45 GB):  36%|███▌      | 21/58 [00:02<00:02, 13.72it/s]Capturing num tokens (num_tokens=832 avail_mem=38.45 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.64it/s]Capturing num tokens (num_tokens=768 avail_mem=38.44 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.64it/s]

    Capturing num tokens (num_tokens=704 avail_mem=38.44 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.64it/s]Capturing num tokens (num_tokens=640 avail_mem=38.44 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.64it/s]Capturing num tokens (num_tokens=576 avail_mem=38.43 GB):  41%|████▏     | 24/58 [00:03<00:02, 16.64it/s]Capturing num tokens (num_tokens=576 avail_mem=38.43 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.53it/s]Capturing num tokens (num_tokens=512 avail_mem=38.43 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.53it/s]Capturing num tokens (num_tokens=480 avail_mem=38.42 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.53it/s]Capturing num tokens (num_tokens=448 avail_mem=38.42 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.53it/s]Capturing num tokens (num_tokens=416 avail_mem=38.42 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.53it/s]

    Capturing num tokens (num_tokens=416 avail_mem=38.42 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.00it/s]Capturing num tokens (num_tokens=384 avail_mem=38.41 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.00it/s]Capturing num tokens (num_tokens=352 avail_mem=38.41 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.00it/s]Capturing num tokens (num_tokens=320 avail_mem=38.41 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.00it/s]Capturing num tokens (num_tokens=288 avail_mem=38.41 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.00it/s]Capturing num tokens (num_tokens=288 avail_mem=38.41 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.07it/s]Capturing num tokens (num_tokens=256 avail_mem=38.41 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.07it/s]Capturing num tokens (num_tokens=240 avail_mem=38.40 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.07it/s]Capturing num tokens (num_tokens=224 avail_mem=38.40 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.07it/s]Capturing num tokens (num_tokens=208 avail_mem=38.39 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.07it/s]

    Capturing num tokens (num_tokens=208 avail_mem=38.39 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.23it/s]Capturing num tokens (num_tokens=192 avail_mem=38.39 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.23it/s]Capturing num tokens (num_tokens=176 avail_mem=38.39 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.23it/s]Capturing num tokens (num_tokens=160 avail_mem=38.39 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.23it/s]Capturing num tokens (num_tokens=144 avail_mem=38.38 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.23it/s]Capturing num tokens (num_tokens=144 avail_mem=38.38 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.73it/s]Capturing num tokens (num_tokens=128 avail_mem=38.38 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.73it/s]Capturing num tokens (num_tokens=112 avail_mem=38.38 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.73it/s]Capturing num tokens (num_tokens=96 avail_mem=38.37 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.73it/s] Capturing num tokens (num_tokens=80 avail_mem=38.37 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.73it/s]

    Capturing num tokens (num_tokens=64 avail_mem=38.37 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.73it/s]Capturing num tokens (num_tokens=64 avail_mem=38.37 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.35it/s]Capturing num tokens (num_tokens=48 avail_mem=38.36 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.35it/s]Capturing num tokens (num_tokens=32 avail_mem=38.36 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.35it/s]Capturing num tokens (num_tokens=28 avail_mem=38.36 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.35it/s]Capturing num tokens (num_tokens=24 avail_mem=38.35 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.35it/s]Capturing num tokens (num_tokens=20 avail_mem=38.35 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.35it/s]Capturing num tokens (num_tokens=20 avail_mem=38.35 GB):  93%|█████████▎| 54/58 [00:03<00:00, 36.31it/s]Capturing num tokens (num_tokens=16 avail_mem=38.35 GB):  93%|█████████▎| 54/58 [00:03<00:00, 36.31it/s]Capturing num tokens (num_tokens=12 avail_mem=38.34 GB):  93%|█████████▎| 54/58 [00:03<00:00, 36.31it/s]Capturing num tokens (num_tokens=8 avail_mem=38.34 GB):  93%|█████████▎| 54/58 [00:03<00:00, 36.31it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=38.34 GB):  93%|█████████▎| 54/58 [00:03<00:00, 36.31it/s]Capturing num tokens (num_tokens=4 avail_mem=38.34 GB): 100%|██████████| 58/58 [00:03<00:00, 37.05it/s]Capturing num tokens (num_tokens=4 avail_mem=38.34 GB): 100%|██████████| 58/58 [00:03<00:00, 14.67it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants to know the current date and time along with the weather. Let me figure out how to approach this.<br><br>First, I need to call the get_current_date function. The user mentioned the timezone is New York, so I should include that in the parameters. The function expects a 'timezone' key with the value 'America/New_York'. I'll structure the parameters accordingly.<br><br>Next, for the weather, I should use get_current_weather. The city is New York, the state is NY, and since the user didn't specify, I'll default to Celsius for the unit. So the parameters will have 'city' as 'New York', 'state' as 'NY', and 'unit' as 'celsius'.<br><br>I need to make sure each function call is separate and follows the required format. So I'll write two function calls: one for the date and time, and another for the weather. Each will have their parameters in JSON format within the function call syntax.<br><br>I should also remember to include the sources in the response, so I'll add the relevant function descriptions as comments. That way, the user knows where the information came from.<br><br>Putting it all together, I'll send the two function calls back to the user, each on their own line for clarity. That should give them the current date, time, and weather in New York.<br><br><br>content: <br><br><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "celsius"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '2bc66bc8e31441e58a22c5ab89ccdab2', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.180803809314966, 'response_sent_to_client_ts': 1780663340.6160681}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '3b3066610bf148d3823c752c71409ed8', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 16.435721118934453, 'response_sent_to_client_ts': 1780663357.0605643}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '06ec8b0463a142a5854eb0a0efa95070', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.17415558081120253, 'response_sent_to_client_ts': 1780663357.2614634}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'bb1338de747f414abe98c352f593f2be', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.17393885646015406, 'response_sent_to_client_ts': 1780663357.2614768}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'e3b0f54fd7e84089a6320d32464ced5a', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.17388905677944422, 'response_sent_to_client_ts': 1780663357.2614813}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '851a3b4bcea04c269d5a2a277d1bbb22', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 20.78065603878349, 'response_sent_to_client_ts': 1780663378.05053}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so I need to figure out how to provide information and the population of the capital of France in JSON format. Let me start by recalling what I know. The capital of France is Paris, right? That\'s pretty straightforward. So, the first part is definite. \n\nNow, for the population part. I remember that Paris is a major city, so it must have a large population. I think it\'s in the hundreds of millions, but I\'m not exactly sure of the exact number. I\'ve heard it mentioned sometimes as over 100 million people, but is it 105 or maybe 106? I\'m a bit fuzzy on the precise figure. I should try to get that right because providing the wrong number would be incorrect. Maybe I can double-check that later.\n\nWait, is Paris an agglomeration that includes other nearby cities like Île-de-France? I\'m not entirely certain. I know that北京市 in China includes both the city proper and its surrounding regions, like the urban agglomeration. So perhaps the population figure I find should consider that. That could affect the number, but I think the capital specifically refers just to the city of Paris, not the broader metropolitan area.\n\nNext, I need to structure this information into a JSON format. JSON stands for JavaScript Object Notation, and it\'s a lightweight data-interchange format. It\'s easy for humans to read and write, and easy for machines to parse and generate. So, I should create a JSON object with a key-value pair. The key should be something descriptive, maybe "capital" or "paris_info". The value would be another object containing more details, like "coordinates" and "population".\n\nFor the coordinates, I need to provide the latitude and longitude of Paris. I think the approximate coordinates are around 48.8566° N latitude and 2.3522° E longitude. I remember learning these in geography class, but I\'m not 100% sure. Maybe I should look them up again to confirm. \n\nThen, for the population, as I mentioned earlier, I\'m a bit uncertain about the exact number. I think it\'s somewhere around 105-106 million. Let me see if I can recall any recent data. I remember that in 2020, Paris had a population of about 105.7 million. Was it an increase or decrease from previous years? Did it grow or shrink? I think it\'s roughly stable, maybe a slight increase. So I\'ll go with 105.6 million to keep it precise.\n\nNow, putting it all together, the JSON structure should reflect this. I\'ll structure the main object to have a key, perhaps "paris", which holds another object with "coordinates" and "population". That way, it\'s organized and easy to parse.\n\nLet me make sure I\'m not missing anything. Does the JSON need to have more detailed structure, like including more fields? Maybe, but since the user only asked for population along with the information, I think two nested objects are sufficient. Including too many fields might complicate it without adding necessary information.\n\nI should also note that if I were to present this in a more formal setting, I might include standard information like the country it\'s part of, but since the question didn\'t specify that, I think it\'s okay to focus just on coordinates and population.\n\nWait, am I supposed to represent the population as an integer or a decimal? JSON doesn\'t enforce data types strictly, but it\'s better to be precise. 105.6 million is a decimal, so I should present it as such. Maybe rounding to one decimal place for clarity.\n\nAlso, the coordinates, should I represent them with decimal degrees or degrees and minutes? I think decimal degrees are more standard in JSON, so that\'s fine.\n\nAnother thing to consider is whether to include units for the population. Should I write the number followed by \'million\' or specify the unit as a string? I think it\'s better to just present the number as a number type in JSON since it\'s the standard approach.\n\nHmm, what if the exact population figure is more recent data? I might be out of date, but without current data, 105.6 million seems reasonable based on what I remember. I should keep in mind that numbers can change over time due to various factors like migration, birth rates, and death rates.\n\nOkay, so to summarize my steps:\n\n1. Identify the capital of France, which is Paris.\n2. Determine the correct coordinates of Paris, likely around 48.8566° N, 2.3522° E.\n3. Find the accurate population figure for Paris. I think 105.6 million is correct as of recent estimates.\n4. Structure this information into a JSON object with appropriate nested objects for coordinates and population.\n\nPutting it all together, the JSON should look something like this:\n\n{\n  "paris": {\n    "coordinates": {\n      "latitude": 48.8566,\n      "longitude": 2.3522\n    },\n    "population": 105600000\n  }\n}\n\nWait, 105.6 million is 105,600,000. So I can write it as 105600000 as an integer or keep it as 105600000. Alternatively, as a float, 105600000.0. But since the population count is usually an integer, perhaps it\'s better to leave it as 105600000.\n\nAlternatively, I can write it as a float with one decimal place: 105600000.0, but JSON can handle both. To be precise, maybe 105600000 is the right way.\n\nWait, another way: if I write 105.6 million as 105,600,000, then in JSON that\'s 105600000, without the comma, so I should write it as an integer.\n\nAnother thought: sometimes populations are given with an estimated range. Should I present it as a range, like between 105.1 million and 106.9 million? But since the user didn\'t specify, just a single value is acceptable.\n\nSo, final JSON:\n\n{\n  "paris": {\n    "coordinates": {\n      "latitude": 48.8566,\n      "longitude": 2.3522\n    },\n    "population": 105600000\n  }\n}\n\nI think this covers all the necessary information concisely. It\'s clear, well-structured, and provides both the location and population data for the capital of France.\n</think>\n\nHere is the JSON information about the capital of France, Paris, including its coordinates and population:\n\n```json\n{\n  "paris": {\n    "coordinates": {\n      "latitude": 48.8566,\n      "longitude": 2.3522\n    },\n    "population": 105600000\n  }\n}\n```\n\nThis JSON object provides the latitude and longitude of Paris and its population figure as of recent estimates.', 'output_ids': [32313, 11, 773, 358, 1184, 311, 7071, 700, 1246, 311, 3410, 1995, 323, 279, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1191, 553, 88646, 1128, 358, 1414, 13, 576, 6722, 315, 9625, 374, 12095, 11, 1290, 30, 2938, 594, 5020, 30339, 13, 2055, 11, 279, 1156, 949, 374, 43770, 13, 4710, 7039, 11, 369, 279, 7042, 949, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 11, 773, 432, 1969, 614, 264, 3460, 7042, 13, 358, 1744, 432, 594, 304, 279, 11499, 315, 11728, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 358, 3003, 6617, 432, 9733, 7025, 438, 916, 220, 16, 15, 15, 3526, 1251, 11, 714, 374, 432, 220, 16, 15, 20, 476, 7196, 220, 16, 15, 21, 30, 358, 2776, 264, 2699, 52733, 389, 279, 23560, 7071, 13, 358, 1265, 1430, 311, 633, 429, 1290, 1576, 8241, 279, 4969, 1372, 1035, 387, 15114, 13, 10696, 358, 646, 1990, 15934, 429, 2937, 382, 14190, 11, 374, 12095, 458, 933, 6072, 316, 20927, 429, 5646, 1008, 14046, 9720, 1075, 59108, 273, 6810, 7276, 34106, 30, 358, 2776, 537, 11368, 3654, 13, 358, 1414, 429, 104554, 304, 5616, 5646, 2176, 279, 3283, 6169, 323, 1181, 14590, 13604, 11, 1075, 279, 15662, 933, 6072, 316, 20927, 13, 2055, 8365, 279, 7042, 7071, 358, 1477, 1265, 2908, 429, 13, 2938, 1410, 7802, 279, 1372, 11, 714, 358, 1744, 279, 6722, 11689, 19257, 1101, 311, 279, 3283, 315, 12095, 11, 537, 279, 26829, 57406, 3082, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 323, 432, 594, 264, 29144, 821, 44894, 3373, 3561, 13, 1084, 594, 4135, 369, 12677, 311, 1349, 323, 3270, 11, 323, 4135, 369, 12645, 311, 4715, 323, 6923, 13, 2055, 11, 358, 1265, 1855, 264, 4718, 1633, 448, 264, 1376, 19083, 6716, 13, 576, 1376, 1265, 387, 2494, 52844, 11, 7196, 330, 65063, 1, 476, 330, 1732, 285, 3109, 3263, 576, 897, 1035, 387, 2441, 1633, 8482, 803, 3565, 11, 1075, 330, 34739, 1, 323, 330, 44441, 11436, 2461, 279, 13934, 11, 358, 1184, 311, 3410, 279, 20849, 323, 20515, 315, 12095, 13, 358, 1744, 279, 44868, 13934, 525, 2163, 220, 19, 23, 13, 23, 20, 21, 21, 11616, 451, 20849, 323, 220, 17, 13, 18, 20, 17, 17, 11616, 468, 20515, 13, 358, 6099, 6832, 1493, 304, 53142, 536, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 2704, 13, 10696, 358, 1265, 1401, 1105, 705, 1549, 311, 7683, 13, 4710, 12209, 11, 369, 279, 7042, 11, 438, 358, 9733, 6788, 11, 358, 2776, 264, 2699, 35118, 911, 279, 4734, 1372, 13, 358, 1744, 432, 594, 14696, 2163, 220, 16, 15, 20, 12, 16, 15, 21, 3526, 13, 6771, 752, 1490, 421, 358, 646, 19091, 894, 3213, 821, 13, 358, 6099, 429, 304, 220, 17, 15, 17, 15, 11, 12095, 1030, 264, 7042, 315, 911, 220, 16, 15, 20, 13, 22, 3526, 13, 14804, 432, 458, 5263, 476, 18472, 504, 3681, 1635, 30, 14568, 432, 3063, 476, 28900, 30, 358, 1744, 432, 594, 17267, 15175, 11, 7196, 264, 8112, 5263, 13, 2055, 358, 3278, 728, 448, 220, 16, 15, 20, 13, 21, 3526, 311, 2506, 432, 23560, 382, 7039, 11, 10687, 432, 678, 3786, 11, 279, 4718, 5944, 1265, 8708, 419, 13, 358, 3278, 5944, 279, 1887, 1633, 311, 614, 264, 1376, 11, 8365, 330, 1732, 285, 497, 892, 9982, 2441, 1633, 448, 330, 34739, 1, 323, 330, 44441, 3263, 2938, 1616, 11, 432, 594, 16645, 323, 4135, 311, 4715, 382, 10061, 752, 1281, 2704, 358, 2776, 537, 7402, 4113, 13, 12553, 279, 4718, 1184, 311, 614, 803, 11682, 5944, 11, 1075, 2670, 803, 5043, 30, 10696, 11, 714, 2474, 279, 1196, 1172, 4588, 369, 7042, 3156, 448, 279, 1995, 11, 358, 1744, 1378, 24034, 6171, 525, 14016, 13, 55121, 2238, 1657, 5043, 2578, 1367, 48795, 432, 2041, 7842, 5871, 1995, 382, 40, 1265, 1083, 5185, 429, 421, 358, 1033, 311, 3042, 419, 304, 264, 803, 15908, 6243, 11, 358, 2578, 2924, 5297, 1995, 1075, 279, 3146, 432, 594, 949, 315, 11, 714, 2474, 279, 3405, 3207, 944, 13837, 429, 11, 358, 1744, 432, 594, 16910, 311, 5244, 1101, 389, 13934, 323, 7042, 382, 14190, 11, 1079, 358, 9966, 311, 4009, 279, 7042, 438, 458, 7546, 476, 264, 12122, 30, 4718, 3171, 944, 28162, 821, 4494, 25470, 11, 714, 432, 594, 2664, 311, 387, 23560, 13, 220, 16, 15, 20, 13, 21, 3526, 374, 264, 12122, 11, 773, 358, 1265, 3042, 432, 438, 1741, 13, 10696, 51562, 311, 825, 12122, 1992, 369, 31273, 382, 13394, 11, 279, 13934, 11, 1265, 358, 4009, 1105, 448, 12122, 12348, 476, 12348, 323, 4420, 30, 358, 1744, 12122, 12348, 525, 803, 5297, 304, 4718, 11, 773, 429, 594, 6915, 382, 14037, 3166, 311, 2908, 374, 3425, 311, 2924, 8153, 369, 279, 7042, 13, 12260, 358, 3270, 279, 1372, 8110, 553, 364, 58313, 6, 476, 13837, 279, 4982, 438, 264, 914, 30, 358, 1744, 432, 594, 2664, 311, 1101, 3042, 279, 1372, 438, 264, 1372, 943, 304, 4718, 2474, 432, 594, 279, 5297, 5486, 382, 80022, 11, 1128, 421, 279, 4734, 7042, 7071, 374, 803, 3213, 821, 30, 358, 2578, 387, 700, 315, 2400, 11, 714, 2041, 1482, 821, 11, 220, 16, 15, 20, 13, 21, 3526, 4977, 13276, 3118, 389, 1128, 358, 6099, 13, 358, 1265, 2506, 304, 3971, 429, 5109, 646, 2297, 916, 882, 4152, 311, 5257, 9363, 1075, 11906, 11, 7194, 7813, 11, 323, 4545, 7813, 382, 32313, 11, 773, 311, 62079, 847, 7354, 1447, 16, 13, 64547, 279, 6722, 315, 9625, 11, 892, 374, 12095, 624, 17, 13, 29901, 279, 4396, 13934, 315, 12095, 11, 4363, 2163, 220, 19, 23, 13, 23, 20, 21, 21, 11616, 451, 11, 220, 17, 13, 18, 20, 17, 17, 11616, 468, 624, 18, 13, 7379, 279, 13382, 7042, 7071, 369, 12095, 13, 358, 1744, 220, 16, 15, 20, 13, 21, 3526, 374, 4396, 438, 315, 3213, 17530, 624, 19, 13, 28596, 419, 1995, 1119, 264, 4718, 1633, 448, 8311, 24034, 6171, 369, 13934, 323, 7042, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 1447, 515, 220, 330, 1732, 285, 788, 341, 262, 330, 34739, 788, 341, 414, 330, 23718, 788, 220, 19, 23, 13, 23, 20, 21, 21, 345, 414, 330, 25446, 788, 220, 17, 13, 18, 20, 17, 17, 198, 262, 1153, 262, 330, 44441, 788, 220, 16, 15, 20, 21, 15, 15, 15, 15, 15, 198, 220, 456, 630, 14190, 11, 220, 16, 15, 20, 13, 21, 3526, 374, 220, 16, 15, 20, 11, 21, 15, 15, 11, 15, 15, 15, 13, 2055, 358, 646, 3270, 432, 438, 220, 16, 15, 20, 21, 15, 15, 15, 15, 15, 438, 458, 7546, 476, 2506, 432, 438, 220, 16, 15, 20, 21, 15, 15, 15, 15, 15, 13, 38478, 11, 438, 264, 2224, 11, 220, 16, 15, 20, 21, 15, 15, 15, 15, 15, 13, 15, 13, 1988, 2474, 279, 7042, 1760, 374, 5990, 458, 7546, 11, 8365, 432, 594, 2664, 311, 5274, 432, 438, 220, 16, 15, 20, 21, 15, 15, 15, 15, 15, 382, 92014, 11, 358, 646, 3270, 432, 438, 264, 2224, 448, 825, 12122, 1992, 25, 220, 16, 15, 20, 21, 15, 15, 15, 15, 15, 13, 15, 11, 714, 4718, 646, 3705, 2176, 13, 2014, 387, 23560, 11, 7196, 220, 16, 15, 20, 21, 15, 15, 15, 15, 15, 374, 279, 1290, 1616, 382, 14190, 11, 2441, 1616, 25, 421, 358, 3270, 220, 16, 15, 20, 13, 21, 3526, 438, 220, 16, 15, 20, 11, 21, 15, 15, 11, 15, 15, 15, 11, 1221, 304, 4718, 429, 594, 220, 16, 15, 20, 21, 15, 15, 15, 15, 15, 11, 2041, 279, 31683, 11, 773, 358, 1265, 3270, 432, 438, 458, 7546, 382, 14037, 3381, 25, 7025, 21910, 525, 2661, 448, 458, 12943, 2088, 13, 12260, 358, 3042, 432, 438, 264, 2088, 11, 1075, 1948, 220, 16, 15, 20, 13, 16, 3526, 323, 220, 16, 15, 21, 13, 24, 3526, 30, 1988, 2474, 279, 1196, 3207, 944, 13837, 11, 1101, 264, 3175, 897, 374, 21555, 382, 4416, 11, 1590, 4718, 1447, 515, 220, 330, 1732, 285, 788, 341, 262, 330, 34739, 788, 341, 414, 330, 23718, 788, 220, 19, 23, 13, 23, 20, 21, 21, 345, 414, 330, 25446, 788, 220, 17, 13, 18, 20, 17, 17, 198, 262, 1153, 262, 330, 44441, 788, 220, 16, 15, 20, 21, 15, 15, 15, 15, 15, 198, 220, 456, 630, 40, 1744, 419, 14521, 678, 279, 5871, 1995, 3529, 285, 974, 13, 1084, 594, 2797, 11, 1632, 12, 51143, 11, 323, 5707, 2176, 279, 3728, 323, 7042, 821, 369, 279, 6722, 315, 9625, 624, 151649, 271, 8420, 374, 279, 4718, 1995, 911, 279, 6722, 315, 9625, 11, 12095, 11, 2670, 1181, 13934, 323, 7042, 1447, 73594, 2236, 198, 515, 220, 330, 1732, 285, 788, 341, 262, 330, 34739, 788, 341, 414, 330, 23718, 788, 220, 19, 23, 13, 23, 20, 21, 21, 345, 414, 330, 25446, 788, 220, 17, 13, 18, 20, 17, 17, 198, 262, 1153, 262, 330, 44441, 788, 220, 16, 15, 20, 21, 15, 15, 15, 15, 15, 198, 220, 456, 532, 13874, 19324, 1986, 4718, 1633, 5707, 279, 20849, 323, 20515, 315, 12095, 323, 1181, 7042, 7071, 438, 315, 3213, 17530, 13, 151643], 'meta_info': {'id': '83f4fdf84a4c4598ac6b0c78ffdbf034', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 1452, 'completion_tokens': 1554, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 17.425106250680983, 'response_sent_to_client_ts': 1780663395.4846678}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.17s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.10s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.11s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:50,  5.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:50,  5.10s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:04,  2.23s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:04,  2.23s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:11,  1.30s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:11,  1.30s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.19it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.19it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.85it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.85it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.59it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.59it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.12it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.12it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.12it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.66it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.66it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.66it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.26it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.26it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.26it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.21it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.21it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.21it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.21it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.50it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 19.95it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.84it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.84it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.84it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.84it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.84it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.84it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.84it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.09it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.09it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.09it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.09it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.09it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.09it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.09it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.09it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 43.02it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 43.02it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:07<00:00, 43.02it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:07<00:00, 43.02it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:07<00:00, 43.02it/s] 

    Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:07<00:00, 43.02it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:07<00:00, 43.02it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:07<00:00, 43.02it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:07<00:00, 43.02it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:07<00:00, 51.66it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:07<00:00, 51.66it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:07<00:00, 51.66it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:07<00:00, 51.66it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:07<00:00, 51.66it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:07<00:00, 51.66it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:07<00:00, 51.66it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:07<00:00, 51.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.51it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.19 GB):   2%|▏         | 1/58 [00:00<00:17,  3.29it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.44 GB):   2%|▏         | 1/58 [00:00<00:17,  3.29it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.44 GB):   3%|▎         | 2/58 [00:00<00:25,  2.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.22 GB):   3%|▎         | 2/58 [00:00<00:25,  2.22it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.22 GB):   5%|▌         | 3/58 [00:01<00:19,  2.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.22 GB):   5%|▌         | 3/58 [00:01<00:19,  2.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.22 GB):   7%|▋         | 4/58 [00:01<00:16,  3.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.22 GB):   7%|▋         | 4/58 [00:01<00:16,  3.31it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.22 GB):   9%|▊         | 5/58 [00:01<00:14,  3.73it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.22 GB):   9%|▊         | 5/58 [00:01<00:14,  3.73it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.22 GB):  10%|█         | 6/58 [00:01<00:12,  4.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.22 GB):  10%|█         | 6/58 [00:01<00:12,  4.22it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.22 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.22 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.22 GB):  14%|█▍        | 8/58 [00:02<00:09,  5.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.22 GB):  14%|█▍        | 8/58 [00:02<00:09,  5.24it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.22 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.22 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.22 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.22 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.30it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.22 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.22 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.22 GB):  21%|██        | 12/58 [00:02<00:06,  7.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.21 GB):  21%|██        | 12/58 [00:02<00:06,  7.52it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=42.21 GB):  21%|██        | 12/58 [00:02<00:06,  7.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.21 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.21 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.21 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.69it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=42.21 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.21 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.21 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.21 GB):  31%|███       | 18/58 [00:02<00:03, 11.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.20 GB):  31%|███       | 18/58 [00:02<00:03, 11.59it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.20 GB):  31%|███       | 18/58 [00:03<00:03, 11.59it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=42.20 GB):  34%|███▍      | 20/58 [00:03<00:02, 13.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.19 GB):  34%|███▍      | 20/58 [00:03<00:02, 13.19it/s]Capturing num tokens (num_tokens=960 avail_mem=42.18 GB):  34%|███▍      | 20/58 [00:03<00:02, 13.19it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=42.18 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.40it/s]Capturing num tokens (num_tokens=896 avail_mem=42.18 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.40it/s]Capturing num tokens (num_tokens=832 avail_mem=42.18 GB):  38%|███▊      | 22/58 [00:03<00:03, 10.40it/s]

    Capturing num tokens (num_tokens=832 avail_mem=42.18 GB):  41%|████▏     | 24/58 [00:03<00:03,  9.38it/s]Capturing num tokens (num_tokens=768 avail_mem=42.17 GB):  41%|████▏     | 24/58 [00:03<00:03,  9.38it/s]Capturing num tokens (num_tokens=704 avail_mem=42.17 GB):  41%|████▏     | 24/58 [00:03<00:03,  9.38it/s]

    Capturing num tokens (num_tokens=704 avail_mem=42.17 GB):  45%|████▍     | 26/58 [00:03<00:03,  9.11it/s]Capturing num tokens (num_tokens=640 avail_mem=42.17 GB):  45%|████▍     | 26/58 [00:03<00:03,  9.11it/s]Capturing num tokens (num_tokens=640 avail_mem=42.17 GB):  47%|████▋     | 27/58 [00:03<00:03,  8.82it/s]Capturing num tokens (num_tokens=576 avail_mem=42.16 GB):  47%|████▋     | 27/58 [00:03<00:03,  8.82it/s]

    Capturing num tokens (num_tokens=576 avail_mem=42.16 GB):  48%|████▊     | 28/58 [00:04<00:03,  8.82it/s]Capturing num tokens (num_tokens=512 avail_mem=42.16 GB):  48%|████▊     | 28/58 [00:04<00:03,  8.82it/s]Capturing num tokens (num_tokens=512 avail_mem=42.16 GB):  50%|█████     | 29/58 [00:04<00:03,  8.86it/s]Capturing num tokens (num_tokens=480 avail_mem=42.16 GB):  50%|█████     | 29/58 [00:04<00:03,  8.86it/s]

    Capturing num tokens (num_tokens=480 avail_mem=42.16 GB):  52%|█████▏    | 30/58 [00:04<00:03,  9.00it/s]Capturing num tokens (num_tokens=448 avail_mem=42.15 GB):  52%|█████▏    | 30/58 [00:04<00:03,  9.00it/s]Capturing num tokens (num_tokens=448 avail_mem=42.15 GB):  53%|█████▎    | 31/58 [00:04<00:03,  8.71it/s]Capturing num tokens (num_tokens=416 avail_mem=42.15 GB):  53%|█████▎    | 31/58 [00:04<00:03,  8.71it/s]

    Capturing num tokens (num_tokens=416 avail_mem=42.15 GB):  55%|█████▌    | 32/58 [00:04<00:02,  8.98it/s]Capturing num tokens (num_tokens=384 avail_mem=42.15 GB):  55%|█████▌    | 32/58 [00:04<00:02,  8.98it/s]Capturing num tokens (num_tokens=384 avail_mem=42.15 GB):  57%|█████▋    | 33/58 [00:04<00:02,  8.99it/s]Capturing num tokens (num_tokens=352 avail_mem=42.14 GB):  57%|█████▋    | 33/58 [00:04<00:02,  8.99it/s]

    Capturing num tokens (num_tokens=320 avail_mem=42.14 GB):  57%|█████▋    | 33/58 [00:04<00:02,  8.99it/s]Capturing num tokens (num_tokens=320 avail_mem=42.14 GB):  60%|██████    | 35/58 [00:04<00:02,  9.42it/s]Capturing num tokens (num_tokens=288 avail_mem=42.14 GB):  60%|██████    | 35/58 [00:04<00:02,  9.42it/s]Capturing num tokens (num_tokens=256 avail_mem=42.14 GB):  60%|██████    | 35/58 [00:04<00:02,  9.42it/s]

    Capturing num tokens (num_tokens=256 avail_mem=42.14 GB):  64%|██████▍   | 37/58 [00:05<00:02,  9.82it/s]Capturing num tokens (num_tokens=240 avail_mem=42.13 GB):  64%|██████▍   | 37/58 [00:05<00:02,  9.82it/s]Capturing num tokens (num_tokens=240 avail_mem=42.13 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.83it/s]Capturing num tokens (num_tokens=224 avail_mem=42.13 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.83it/s]Capturing num tokens (num_tokens=208 avail_mem=42.12 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.83it/s]

    Capturing num tokens (num_tokens=208 avail_mem=42.12 GB):  69%|██████▉   | 40/58 [00:05<00:01, 10.12it/s]Capturing num tokens (num_tokens=192 avail_mem=42.12 GB):  69%|██████▉   | 40/58 [00:05<00:01, 10.12it/s]Capturing num tokens (num_tokens=176 avail_mem=42.12 GB):  69%|██████▉   | 40/58 [00:05<00:01, 10.12it/s]Capturing num tokens (num_tokens=176 avail_mem=42.12 GB):  72%|███████▏  | 42/58 [00:05<00:01, 10.31it/s]Capturing num tokens (num_tokens=160 avail_mem=42.12 GB):  72%|███████▏  | 42/58 [00:05<00:01, 10.31it/s]

    Capturing num tokens (num_tokens=144 avail_mem=42.11 GB):  72%|███████▏  | 42/58 [00:05<00:01, 10.31it/s]Capturing num tokens (num_tokens=144 avail_mem=42.11 GB):  76%|███████▌  | 44/58 [00:05<00:01, 10.56it/s]Capturing num tokens (num_tokens=128 avail_mem=42.11 GB):  76%|███████▌  | 44/58 [00:05<00:01, 10.56it/s]Capturing num tokens (num_tokens=112 avail_mem=42.11 GB):  76%|███████▌  | 44/58 [00:05<00:01, 10.56it/s]

    Capturing num tokens (num_tokens=112 avail_mem=42.11 GB):  79%|███████▉  | 46/58 [00:05<00:01, 10.75it/s]Capturing num tokens (num_tokens=96 avail_mem=42.10 GB):  79%|███████▉  | 46/58 [00:05<00:01, 10.75it/s] Capturing num tokens (num_tokens=80 avail_mem=42.10 GB):  79%|███████▉  | 46/58 [00:05<00:01, 10.75it/s]Capturing num tokens (num_tokens=80 avail_mem=42.10 GB):  83%|████████▎ | 48/58 [00:06<00:00, 11.38it/s]Capturing num tokens (num_tokens=64 avail_mem=42.10 GB):  83%|████████▎ | 48/58 [00:06<00:00, 11.38it/s]

    Capturing num tokens (num_tokens=48 avail_mem=42.09 GB):  83%|████████▎ | 48/58 [00:06<00:00, 11.38it/s]Capturing num tokens (num_tokens=48 avail_mem=42.09 GB):  86%|████████▌ | 50/58 [00:06<00:00, 11.83it/s]Capturing num tokens (num_tokens=32 avail_mem=42.09 GB):  86%|████████▌ | 50/58 [00:06<00:00, 11.83it/s]Capturing num tokens (num_tokens=28 avail_mem=42.09 GB):  86%|████████▌ | 50/58 [00:06<00:00, 11.83it/s]Capturing num tokens (num_tokens=28 avail_mem=42.09 GB):  90%|████████▉ | 52/58 [00:06<00:00, 12.88it/s]Capturing num tokens (num_tokens=24 avail_mem=42.08 GB):  90%|████████▉ | 52/58 [00:06<00:00, 12.88it/s]

    Capturing num tokens (num_tokens=20 avail_mem=42.08 GB):  90%|████████▉ | 52/58 [00:06<00:00, 12.88it/s]Capturing num tokens (num_tokens=20 avail_mem=42.08 GB):  93%|█████████▎| 54/58 [00:06<00:00, 14.05it/s]Capturing num tokens (num_tokens=16 avail_mem=42.08 GB):  93%|█████████▎| 54/58 [00:06<00:00, 14.05it/s]Capturing num tokens (num_tokens=12 avail_mem=42.07 GB):  93%|█████████▎| 54/58 [00:06<00:00, 14.05it/s]Capturing num tokens (num_tokens=8 avail_mem=42.07 GB):  93%|█████████▎| 54/58 [00:06<00:00, 14.05it/s] Capturing num tokens (num_tokens=8 avail_mem=42.07 GB):  98%|█████████▊| 57/58 [00:06<00:00, 16.51it/s]Capturing num tokens (num_tokens=4 avail_mem=42.07 GB):  98%|█████████▊| 57/58 [00:06<00:00, 16.51it/s]

    Capturing num tokens (num_tokens=4 avail_mem=42.07 GB): 100%|██████████| 58/58 [00:06<00:00,  8.81it/s]


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
    
    Generated text: Alright, so the user is asking for the information and population of the capital of France in JSON format. Hmm, okay, first I need to figure out what exactly they need. The capital of France is definitely Paris, right? So, I should start by confirming that.
    
    Now, they want the population, so I need to find the most recent data. I remember that population numbers can change every year, so I should look up the latest estimates. I think as of 2023, Paris has a population around 2.2 million. But I should double-check that to make sure it's accurate. Maybe there are some sources like the official census or recent reports from the city government that I can refer to.
    
    Next, the user specified JSON format. JSON is a data interchange format, so I need to structure the information neatly in that format. I should probably include the country name, the city, its population, and maybe some additional info like the area in square kilometers. That way, the data is comprehensive and easy to read.
    
    I should also consider why the user is asking this. Maybe they're a student working on a project, or perhaps they're a researcher compiling data. Either way, providing accurate and up-to-date information is crucial. I don't want to give them incorrect numbers, so I'll verify the population figure again.
    
    Another thought: should I include the date of the population estimate? That might be useful depending on when they're using this data. I think including the year, maybe 2022 or 2023, would make it clear. Also, if the population is projected or estimated, I should note that it's an approximate figure.
    
    I should structure the JSON with key-value pairs. The main keys would be "country," "city," "population," and "additional_info." Under each key, I'll place the relevant data. That way, it's organized and easy to parse.
    
    Let me think if there's anything else the user might need. They didn't ask for a chart or graph, just the information in JSON. So, I'll stick to that format without adding unnecessary details.
    
    Finally, I'll make sure the JSON syntax is correct, with proper commas and brackets. No trailing commas to avoid errors. Once everything is structured correctly, I can present it to the user. That should fulfill their request effectively.
    </think>
    
    Here is the information about the capital of France in JSON format:
    
    ```json
    {
      "country": "France",
      "city": "Paris",
      "population": 2215000,
      "additional_info": "Paris is the capital city of France and has a population of approximately 2.215 million as of the latest estimates."
    }
    ```



```python
llm.shutdown()
```
