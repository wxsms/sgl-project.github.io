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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.43s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.33s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:44,  5.00s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:44,  5.00s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:02,  2.18s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:02,  2.18s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:10,  1.27s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:10,  1.27s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:31,  1.67it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:23,  2.24it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:23,  2.24it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.91it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.91it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.65it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.65it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.20it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.20it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.20it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.70it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.24it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.24it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.24it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.18it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.18it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.18it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:06<00:03, 11.18it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:02, 14.46it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:02, 14.46it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.46it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.46it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.46it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.82it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.82it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.82it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.82it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.82it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.82it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]

    Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 35.14it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 35.14it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 35.14it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 35.14it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 35.14it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 35.14it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 35.14it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 35.14it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:07<00:00, 35.14it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 45.49it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 45.49it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:07<00:00, 45.49it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:07<00:00, 45.49it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:07<00:00, 45.49it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:07<00:00, 45.49it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:07<00:00, 45.49it/s]

    Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:07<00:00, 45.49it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:07<00:00, 51.46it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:07<00:00, 51.46it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 51.46it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 51.46it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:07<00:00, 51.46it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:07<00:00, 51.46it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:07<00:00, 51.46it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:07<00:00, 53.48it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:07<00:00, 53.48it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:07<00:00, 53.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=22.83 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=22.83 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=22.80 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=22.80 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=22.80 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=22.80 GB):   5%|▌         | 3/58 [00:00<00:14,  3.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=22.80 GB):   5%|▌         | 3/58 [00:00<00:14,  3.82it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=22.80 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=22.80 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=22.80 GB):   9%|▊         | 5/58 [00:01<00:12,  4.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=22.73 GB):   9%|▊         | 5/58 [00:01<00:12,  4.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=22.73 GB):  10%|█         | 6/58 [00:01<00:11,  4.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=20.01 GB):  10%|█         | 6/58 [00:01<00:11,  4.70it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=20.01 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=20.01 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=20.01 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=20.01 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.56it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=20.01 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.05it/s]Capturing num tokens (num_tokens=3840 avail_mem=20.01 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.05it/s]Capturing num tokens (num_tokens=3840 avail_mem=20.01 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=20.01 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.50it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=20.01 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=20.00 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=20.00 GB):  21%|██        | 12/58 [00:02<00:06,  7.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=20.00 GB):  21%|██        | 12/58 [00:02<00:06,  7.66it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=20.00 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.05it/s]Capturing num tokens (num_tokens=2816 avail_mem=20.00 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.05it/s]Capturing num tokens (num_tokens=2560 avail_mem=20.00 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.05it/s]Capturing num tokens (num_tokens=2560 avail_mem=20.00 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.46it/s]Capturing num tokens (num_tokens=2304 avail_mem=20.00 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.46it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=19.99 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=19.99 GB):  29%|██▉       | 17/58 [00:02<00:03, 11.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=19.99 GB):  29%|██▉       | 17/58 [00:02<00:03, 11.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=19.99 GB):  29%|██▉       | 17/58 [00:02<00:03, 11.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=19.99 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=19.99 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.88it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=19.98 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.88it/s]Capturing num tokens (num_tokens=960 avail_mem=19.97 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.88it/s] Capturing num tokens (num_tokens=960 avail_mem=19.97 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.21it/s]Capturing num tokens (num_tokens=896 avail_mem=19.97 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.21it/s]Capturing num tokens (num_tokens=832 avail_mem=19.97 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.21it/s]Capturing num tokens (num_tokens=768 avail_mem=19.96 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.21it/s]Capturing num tokens (num_tokens=768 avail_mem=19.96 GB):  43%|████▎     | 25/58 [00:02<00:01, 19.32it/s]Capturing num tokens (num_tokens=704 avail_mem=19.96 GB):  43%|████▎     | 25/58 [00:02<00:01, 19.32it/s]

    Capturing num tokens (num_tokens=640 avail_mem=19.95 GB):  43%|████▎     | 25/58 [00:02<00:01, 19.32it/s]Capturing num tokens (num_tokens=576 avail_mem=19.95 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.32it/s]Capturing num tokens (num_tokens=512 avail_mem=19.94 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.32it/s]Capturing num tokens (num_tokens=512 avail_mem=19.94 GB):  50%|█████     | 29/58 [00:03<00:01, 23.06it/s]Capturing num tokens (num_tokens=480 avail_mem=19.94 GB):  50%|█████     | 29/58 [00:03<00:01, 23.06it/s]Capturing num tokens (num_tokens=448 avail_mem=19.94 GB):  50%|█████     | 29/58 [00:03<00:01, 23.06it/s]Capturing num tokens (num_tokens=416 avail_mem=19.94 GB):  50%|█████     | 29/58 [00:03<00:01, 23.06it/s]Capturing num tokens (num_tokens=384 avail_mem=19.93 GB):  50%|█████     | 29/58 [00:03<00:01, 23.06it/s]

    Capturing num tokens (num_tokens=384 avail_mem=19.93 GB):  57%|█████▋    | 33/58 [00:03<00:00, 26.15it/s]Capturing num tokens (num_tokens=352 avail_mem=19.93 GB):  57%|█████▋    | 33/58 [00:03<00:00, 26.15it/s]Capturing num tokens (num_tokens=320 avail_mem=19.92 GB):  57%|█████▋    | 33/58 [00:03<00:00, 26.15it/s]Capturing num tokens (num_tokens=288 avail_mem=19.93 GB):  57%|█████▋    | 33/58 [00:03<00:00, 26.15it/s]Capturing num tokens (num_tokens=256 avail_mem=19.92 GB):  57%|█████▋    | 33/58 [00:03<00:00, 26.15it/s]Capturing num tokens (num_tokens=256 avail_mem=19.92 GB):  64%|██████▍   | 37/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=240 avail_mem=19.92 GB):  64%|██████▍   | 37/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=224 avail_mem=19.92 GB):  64%|██████▍   | 37/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=208 avail_mem=19.91 GB):  64%|██████▍   | 37/58 [00:03<00:00, 29.25it/s]Capturing num tokens (num_tokens=192 avail_mem=19.89 GB):  64%|██████▍   | 37/58 [00:03<00:00, 29.25it/s]

    Capturing num tokens (num_tokens=192 avail_mem=19.89 GB):  71%|███████   | 41/58 [00:03<00:00, 31.55it/s]Capturing num tokens (num_tokens=176 avail_mem=19.89 GB):  71%|███████   | 41/58 [00:03<00:00, 31.55it/s]Capturing num tokens (num_tokens=160 avail_mem=19.88 GB):  71%|███████   | 41/58 [00:03<00:00, 31.55it/s]Capturing num tokens (num_tokens=144 avail_mem=19.88 GB):  71%|███████   | 41/58 [00:03<00:00, 31.55it/s]Capturing num tokens (num_tokens=128 avail_mem=19.88 GB):  71%|███████   | 41/58 [00:03<00:00, 31.55it/s]Capturing num tokens (num_tokens=128 avail_mem=19.88 GB):  78%|███████▊  | 45/58 [00:03<00:00, 32.72it/s]Capturing num tokens (num_tokens=112 avail_mem=19.85 GB):  78%|███████▊  | 45/58 [00:03<00:00, 32.72it/s]Capturing num tokens (num_tokens=96 avail_mem=19.84 GB):  78%|███████▊  | 45/58 [00:03<00:00, 32.72it/s] Capturing num tokens (num_tokens=80 avail_mem=19.84 GB):  78%|███████▊  | 45/58 [00:03<00:00, 32.72it/s]Capturing num tokens (num_tokens=64 avail_mem=19.84 GB):  78%|███████▊  | 45/58 [00:03<00:00, 32.72it/s]

    Capturing num tokens (num_tokens=64 avail_mem=19.84 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.47it/s]Capturing num tokens (num_tokens=48 avail_mem=19.83 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.47it/s]Capturing num tokens (num_tokens=32 avail_mem=19.83 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.47it/s]Capturing num tokens (num_tokens=28 avail_mem=19.83 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.47it/s]Capturing num tokens (num_tokens=24 avail_mem=19.82 GB):  84%|████████▍ | 49/58 [00:03<00:00, 34.47it/s]Capturing num tokens (num_tokens=24 avail_mem=19.82 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.93it/s]Capturing num tokens (num_tokens=20 avail_mem=19.82 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.93it/s]Capturing num tokens (num_tokens=16 avail_mem=19.82 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.93it/s]Capturing num tokens (num_tokens=12 avail_mem=19.81 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.93it/s]Capturing num tokens (num_tokens=8 avail_mem=19.81 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.93it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=19.81 GB):  98%|█████████▊| 57/58 [00:03<00:00, 36.91it/s]Capturing num tokens (num_tokens=4 avail_mem=19.81 GB):  98%|█████████▊| 57/58 [00:03<00:00, 36.91it/s]Capturing num tokens (num_tokens=4 avail_mem=19.81 GB): 100%|██████████| 58/58 [00:03<00:00, 15.05it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out how to get the current date and time in New York, along with the weather. The user mentioned they're in New York, so I should probably use the functions provided: get_current_weather and get_current_date.<br><br>First, for the date and time, I'll use get_current_date. The function requires a timezone parameter. New York is in the Eastern Time Zone, so the abbreviation is 'EST'. I'll format the function call as <function=date>{{"timezone": "EST"}}</function>.<br><br>Next, for the weather, I'll use get_current_weather. This function needs the city, state, and unit. The city is New York, the state is NY, and since the user didn't specify, I'll default to Fahrenheit. So the parameters will be city: 'New York', state: 'NY', unit: 'fahrenheit'. The function call will be <function=weather>{{"city": "New York", "state": "NY", "unit": "fahrenheit"}}</function>.<br><br>I should make sure each function call is on its own line and includes the parameters correctly. Also, I need to remember to add the sources when I use the functions. So, I'll mention that I used the get_current_weather and get_current_date functions with the appropriate parameters.<br><br><br>content: <br><br><function=get_current_date>{  <br>  "timezone": "EST"  <br>}</function>  <br><function=get_current_weather>{  <br>  "city": "New York",  <br>  "state": "NY",  <br>  "unit": "fahrenheit"  <br>}</function>  <br><br>I used the get_current_date function to retrieve the current date and time for the Eastern Time Zone (EST) and the get_current_weather function to obtain the current weather in New York, NY, in Fahrenheit.</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '8dd993c1890b4bdd90fb7ecada8263e2', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.7891763648949564, 'response_sent_to_client_ts': 1779416224.5206993}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '88ca3e92e0bc4f2697c6c58fe5ac7877', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 18.793740246910602, 'response_sent_to_client_ts': 1779416243.3226275}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'a533a598e5014b0c8bb4be86adb858e0', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1408476186916232, 'response_sent_to_client_ts': 1779416243.5091617}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'fe7624ef5f0e43e291cb97355b2b978d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14078412391245365, 'response_sent_to_client_ts': 1779416243.509171}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'c1f6c710fb194e45aa4d05fc6747a541', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14074602397158742, 'response_sent_to_client_ts': 1779416243.5091743}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'bee8d6d4a3f94a50937ef97f934adc24', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 26.81364162499085, 'response_sent_to_client_ts': 1779416270.3298428}}


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


<strong style='color: #00008B;'>{'text': 'Alright, so the user is asking for the information and population of the capital of France in JSON format. Hmm, I need to make sure I understand exactly what they\'re looking for. Let me think about the key points they might be interested in.\n\nFirst off, the capital of France is definitely Paris. That part is straightforward. Now, they want population data. I should check the latest available stats. I remember that populations can change every five years based on censuses, so I need to find the most recent one, probably from 2020 or 2021.\n\nWait, Paris is known for being a vibrant city with a largeальное population, but it\'s a metro area. I think the metropolitan area of Paris has a bigger population than just the city limits. I should clarify if they want just Paris proper or the entire metro area.\n\nLooking it up, as of the 2021 census, the population of Paris is around 2,200,000 within the city limits. But the metro area has over 13 million people. So, it\'s good to note both figures just in case they specify which one they need.\n\nAlso, besides the population, they asked for information. Maybe they want more details about Paris, like its location, history, attractions, or something like that. But the user specifically mentioned population along with other info in the JSON, so perhaps they just need the population number. Still, including location might be helpful.\n\nWait, the user said "information and population." So maybe they want a fields of location and population. That makes sense. So the JSON should have a "name" field which is "Paris," a "location" that could be "The Île-de-France region in northern France," and a "population" field with the number.\n\nI should present this in a JSON structure that\'s clear and easy to understand. Let me double-check the population numbers to make sure I have the most accurate and recent data. Yep, 2021 estimates say Paris proper is about 2.2 million, metro area around 13 million.\n\nPutting it all together, the JSON object should have a structure like this: a main key, probably "ParisInfo" which contains an object with "name," "location," and "population." That way, the user can easily see the information and the population separately.\n\nI think that covers everything the user is asking for. I just need to present it neatly without any extra fluff. Keeping it concise but informative should be the way to go.\n</think>\n\nHere is the information and population of the capital of France (Paris) in JSON format:\n\n```json\n{\n  "ParisInfo": {\n    "name": "Paris",\n    "location": "The Île-de-France region in northern France",\n    "population": 2215235  // As of the 2021 census estimate\n  }\n}\n```', 'output_ids': [71486, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 88190, 11, 358, 1184, 311, 1281, 2704, 358, 3535, 6896, 1128, 807, 2299, 3330, 369, 13, 6771, 752, 1744, 911, 279, 1376, 3501, 807, 2578, 387, 8014, 304, 382, 5338, 1007, 11, 279, 6722, 315, 9625, 374, 8491, 12095, 13, 2938, 949, 374, 30339, 13, 4695, 11, 807, 1366, 7042, 821, 13, 358, 1265, 1779, 279, 5535, 2500, 10472, 13, 358, 6099, 429, 21910, 646, 2297, 1449, 4236, 1635, 3118, 389, 272, 724, 4776, 11, 773, 358, 1184, 311, 1477, 279, 1429, 3213, 825, 11, 4658, 504, 220, 17, 15, 17, 15, 476, 220, 17, 15, 17, 16, 382, 14190, 11, 12095, 374, 3881, 369, 1660, 264, 32976, 3283, 448, 264, 3460, 137618, 7042, 11, 714, 432, 594, 264, 33482, 3082, 13, 358, 1744, 279, 57406, 3082, 315, 12095, 702, 264, 11243, 7042, 1091, 1101, 279, 3283, 13388, 13, 358, 1265, 37163, 421, 807, 1366, 1101, 12095, 6169, 476, 279, 4453, 33482, 3082, 382, 22464, 432, 705, 11, 438, 315, 279, 220, 17, 15, 17, 16, 43602, 11, 279, 7042, 315, 12095, 374, 2163, 220, 17, 11, 17, 15, 15, 11, 15, 15, 15, 2878, 279, 3283, 13388, 13, 1988, 279, 33482, 3082, 702, 916, 220, 16, 18, 3526, 1251, 13, 2055, 11, 432, 594, 1661, 311, 5185, 2176, 12396, 1101, 304, 1142, 807, 13837, 892, 825, 807, 1184, 382, 13394, 11, 27758, 279, 7042, 11, 807, 4588, 369, 1995, 13, 10696, 807, 1366, 803, 3565, 911, 12095, 11, 1075, 1181, 3728, 11, 3840, 11, 38491, 11, 476, 2494, 1075, 429, 13, 1988, 279, 1196, 11689, 9733, 7042, 3156, 448, 1008, 3546, 304, 279, 4718, 11, 773, 8365, 807, 1101, 1184, 279, 7042, 1372, 13, 16377, 11, 2670, 3728, 2578, 387, 10950, 382, 14190, 11, 279, 1196, 1053, 330, 25069, 323, 7042, 1189, 2055, 7196, 807, 1366, 264, 5043, 315, 3728, 323, 7042, 13, 2938, 3643, 5530, 13, 2055, 279, 4718, 1265, 614, 264, 330, 606, 1, 2070, 892, 374, 330, 59604, 1335, 264, 330, 2527, 1, 429, 1410, 387, 330, 785, 59108, 273, 6810, 7276, 34106, 5537, 304, 18172, 9625, 1335, 323, 264, 330, 44441, 1, 2070, 448, 279, 1372, 382, 40, 1265, 3042, 419, 304, 264, 4718, 5944, 429, 594, 2797, 323, 4135, 311, 3535, 13, 6771, 752, 1990, 15934, 279, 7042, 5109, 311, 1281, 2704, 358, 614, 279, 1429, 13382, 323, 3213, 821, 13, 84194, 11, 220, 17, 15, 17, 16, 17530, 1977, 12095, 6169, 374, 911, 220, 17, 13, 17, 3526, 11, 33482, 3082, 2163, 220, 16, 18, 3526, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 614, 264, 5944, 1075, 419, 25, 264, 1887, 1376, 11, 4658, 330, 59604, 1731, 1, 892, 5610, 458, 1633, 448, 330, 606, 1335, 330, 2527, 1335, 323, 330, 44441, 1189, 2938, 1616, 11, 279, 1196, 646, 6707, 1490, 279, 1995, 323, 279, 7042, 25156, 382, 40, 1744, 429, 14521, 4297, 279, 1196, 374, 10161, 369, 13, 358, 1101, 1184, 311, 3042, 432, 62166, 2041, 894, 4960, 1320, 1362, 13, 55378, 432, 63594, 714, 38219, 1265, 387, 279, 1616, 311, 728, 624, 151649, 271, 8420, 374, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 320, 59604, 8, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 59604, 1731, 788, 341, 262, 330, 606, 788, 330, 59604, 756, 262, 330, 2527, 788, 330, 785, 59108, 273, 6810, 7276, 34106, 5537, 304, 18172, 9625, 756, 262, 330, 44441, 788, 220, 17, 17, 16, 20, 17, 18, 20, 220, 442, 1634, 315, 279, 220, 17, 15, 17, 16, 43602, 16045, 198, 220, 456, 532, 73594, 151643], 'meta_info': {'id': '94624990942446e88bab406973277ce2', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 526, 'completion_tokens': 608, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 6.51228205114603, 'response_sent_to_client_ts': 1779416276.8622472}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.42s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.29s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.31s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:06,  5.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:06,  5.37s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:10,  2.34s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:10,  2.34s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:14,  1.36s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:14,  1.36s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:48,  1.12it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:48,  1.12it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:33,  1.60it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:33,  1.60it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:24,  2.15it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:24,  2.15it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.81it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.81it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.56it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.56it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:12,  3.87it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:12,  3.87it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:11,  4.07it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:11,  4.07it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:10,  4.35it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:10,  4.35it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:10,  4.59it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:10,  4.59it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:09,  4.96it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:09,  4.96it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:08,  5.25it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:08,  5.25it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:07,  5.70it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:07,  5.70it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:06,  6.17it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:06,  6.17it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:08<00:06,  6.71it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:08<00:06,  6.71it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:08<00:06,  6.71it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:08<00:04,  7.98it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:08<00:04,  7.98it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:08<00:04,  7.98it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:08<00:04,  9.20it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:08<00:04,  9.20it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:08<00:04,  9.20it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:08<00:03, 10.94it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:08<00:03, 10.94it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:08<00:03, 10.94it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:08<00:02, 12.48it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:08<00:02, 12.48it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:08<00:02, 12.48it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:08<00:02, 14.19it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:08<00:02, 14.19it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:08<00:02, 14.19it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:08<00:02, 14.19it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:08<00:01, 17.17it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:08<00:01, 17.17it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:08<00:01, 17.17it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:08<00:01, 17.17it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:09<00:01, 18.32it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:09<00:01, 18.32it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:09<00:01, 18.32it/s]

    Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:09<00:01, 18.32it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:09<00:01, 20.56it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:09<00:01, 20.56it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:09<00:01, 20.56it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:09<00:01, 20.56it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:09<00:01, 20.56it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:09<00:00, 23.85it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:09<00:00, 23.85it/s]

    Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:09<00:00, 23.85it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:09<00:00, 23.85it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:09<00:00, 24.91it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:09<00:00, 24.91it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:09<00:00, 24.91it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:09<00:00, 24.91it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:09<00:00, 24.91it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:09<00:00, 26.70it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:09<00:00, 26.70it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:09<00:00, 26.70it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:09<00:00, 26.70it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:09<00:00, 27.05it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:09<00:00, 27.05it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:09<00:00, 27.05it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:09<00:00, 27.05it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:09<00:00, 27.05it/s]

    Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:09<00:00, 30.01it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:09<00:00, 30.01it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:09<00:00, 30.01it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:09<00:00, 30.01it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:09<00:00, 30.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  5.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=44.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=44.19 GB):   2%|▏         | 1/58 [00:00<00:28,  2.03it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.15 GB):   2%|▏         | 1/58 [00:00<00:28,  2.03it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.15 GB):   3%|▎         | 2/58 [00:00<00:24,  2.33it/s]Capturing num tokens (num_tokens=7168 avail_mem=44.15 GB):   3%|▎         | 2/58 [00:00<00:24,  2.33it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=44.15 GB):   5%|▌         | 3/58 [00:01<00:20,  2.70it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.15 GB):   5%|▌         | 3/58 [00:01<00:20,  2.70it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=44.15 GB):   7%|▋         | 4/58 [00:01<00:16,  3.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.15 GB):   7%|▋         | 4/58 [00:01<00:16,  3.25it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.15 GB):   9%|▊         | 5/58 [00:01<00:17,  2.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.01 GB):   9%|▊         | 5/58 [00:01<00:17,  2.98it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.01 GB):  10%|█         | 6/58 [00:02<00:17,  2.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.13 GB):  10%|█         | 6/58 [00:02<00:17,  2.99it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.13 GB):  12%|█▏        | 7/58 [00:02<00:17,  2.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.11 GB):  12%|█▏        | 7/58 [00:02<00:17,  2.87it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=44.11 GB):  14%|█▍        | 8/58 [00:02<00:16,  2.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.19 GB):  14%|█▍        | 8/58 [00:02<00:16,  2.97it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.19 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.26 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.10it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.26 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.21it/s]Capturing num tokens (num_tokens=3584 avail_mem=44.11 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.21it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=44.11 GB):  19%|█▉        | 11/58 [00:03<00:13,  3.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.32 GB):  19%|█▉        | 11/58 [00:03<00:13,  3.40it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=43.32 GB):  21%|██        | 12/58 [00:03<00:12,  3.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.38 GB):  21%|██        | 12/58 [00:03<00:12,  3.67it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=43.38 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=44.10 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.87it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=44.10 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.44 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.44 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.52it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.51 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.52it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=43.51 GB):  28%|██▊       | 16/58 [00:04<00:08,  4.91it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.10 GB):  28%|██▊       | 16/58 [00:04<00:08,  4.91it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.10 GB):  29%|██▉       | 17/58 [00:04<00:08,  5.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.10 GB):  29%|██▉       | 17/58 [00:04<00:08,  5.12it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=44.10 GB):  31%|███       | 18/58 [00:04<00:07,  5.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.59 GB):  31%|███       | 18/58 [00:04<00:07,  5.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.59 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.09 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.75it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=44.09 GB):  34%|███▍      | 20/58 [00:05<00:06,  6.33it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.61 GB):  34%|███▍      | 20/58 [00:05<00:06,  6.33it/s]Capturing num tokens (num_tokens=960 avail_mem=43.76 GB):  34%|███▍      | 20/58 [00:05<00:06,  6.33it/s] Capturing num tokens (num_tokens=960 avail_mem=43.76 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.62it/s]Capturing num tokens (num_tokens=896 avail_mem=44.07 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.62it/s]

    Capturing num tokens (num_tokens=832 avail_mem=43.66 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.62it/s]Capturing num tokens (num_tokens=832 avail_mem=43.66 GB):  41%|████▏     | 24/58 [00:05<00:03,  8.69it/s]Capturing num tokens (num_tokens=768 avail_mem=44.04 GB):  41%|████▏     | 24/58 [00:05<00:03,  8.69it/s]Capturing num tokens (num_tokens=768 avail_mem=44.04 GB):  43%|████▎     | 25/58 [00:05<00:03,  8.81it/s]Capturing num tokens (num_tokens=704 avail_mem=44.05 GB):  43%|████▎     | 25/58 [00:05<00:03,  8.81it/s]

    Capturing num tokens (num_tokens=640 avail_mem=43.72 GB):  43%|████▎     | 25/58 [00:05<00:03,  8.81it/s]Capturing num tokens (num_tokens=640 avail_mem=43.72 GB):  47%|████▋     | 27/58 [00:05<00:03,  9.90it/s]Capturing num tokens (num_tokens=576 avail_mem=43.86 GB):  47%|████▋     | 27/58 [00:05<00:03,  9.90it/s]Capturing num tokens (num_tokens=512 avail_mem=43.74 GB):  47%|████▋     | 27/58 [00:05<00:03,  9.90it/s]

    Capturing num tokens (num_tokens=512 avail_mem=43.74 GB):  50%|█████     | 29/58 [00:06<00:02, 11.17it/s]Capturing num tokens (num_tokens=480 avail_mem=43.75 GB):  50%|█████     | 29/58 [00:06<00:02, 11.17it/s]Capturing num tokens (num_tokens=448 avail_mem=43.77 GB):  50%|█████     | 29/58 [00:06<00:02, 11.17it/s]Capturing num tokens (num_tokens=448 avail_mem=43.77 GB):  53%|█████▎    | 31/58 [00:06<00:02, 11.51it/s]Capturing num tokens (num_tokens=416 avail_mem=44.02 GB):  53%|█████▎    | 31/58 [00:06<00:02, 11.51it/s]

    Capturing num tokens (num_tokens=384 avail_mem=44.02 GB):  53%|█████▎    | 31/58 [00:06<00:02, 11.51it/s]Capturing num tokens (num_tokens=384 avail_mem=44.02 GB):  57%|█████▋    | 33/58 [00:06<00:02, 12.34it/s]Capturing num tokens (num_tokens=352 avail_mem=43.81 GB):  57%|█████▋    | 33/58 [00:06<00:02, 12.34it/s]Capturing num tokens (num_tokens=320 avail_mem=43.83 GB):  57%|█████▋    | 33/58 [00:06<00:02, 12.34it/s]Capturing num tokens (num_tokens=320 avail_mem=43.83 GB):  60%|██████    | 35/58 [00:06<00:01, 13.78it/s]Capturing num tokens (num_tokens=288 avail_mem=44.00 GB):  60%|██████    | 35/58 [00:06<00:01, 13.78it/s]

    Capturing num tokens (num_tokens=256 avail_mem=43.92 GB):  60%|██████    | 35/58 [00:06<00:01, 13.78it/s]Capturing num tokens (num_tokens=256 avail_mem=43.92 GB):  64%|██████▍   | 37/58 [00:06<00:01, 14.59it/s]Capturing num tokens (num_tokens=240 avail_mem=43.86 GB):  64%|██████▍   | 37/58 [00:06<00:01, 14.59it/s]Capturing num tokens (num_tokens=224 avail_mem=43.86 GB):  64%|██████▍   | 37/58 [00:06<00:01, 14.59it/s]Capturing num tokens (num_tokens=224 avail_mem=43.86 GB):  67%|██████▋   | 39/58 [00:06<00:01, 15.45it/s]Capturing num tokens (num_tokens=208 avail_mem=43.89 GB):  67%|██████▋   | 39/58 [00:06<00:01, 15.45it/s]

    Capturing num tokens (num_tokens=192 avail_mem=43.96 GB):  67%|██████▋   | 39/58 [00:06<00:01, 15.45it/s]Capturing num tokens (num_tokens=192 avail_mem=43.96 GB):  71%|███████   | 41/58 [00:06<00:01, 16.21it/s]Capturing num tokens (num_tokens=176 avail_mem=43.96 GB):  71%|███████   | 41/58 [00:06<00:01, 16.21it/s]Capturing num tokens (num_tokens=160 avail_mem=43.95 GB):  71%|███████   | 41/58 [00:06<00:01, 16.21it/s]Capturing num tokens (num_tokens=144 avail_mem=43.94 GB):  71%|███████   | 41/58 [00:06<00:01, 16.21it/s]Capturing num tokens (num_tokens=144 avail_mem=43.94 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.17it/s]Capturing num tokens (num_tokens=128 avail_mem=43.94 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.17it/s]

    Capturing num tokens (num_tokens=112 avail_mem=43.94 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.17it/s]Capturing num tokens (num_tokens=96 avail_mem=43.93 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.17it/s] Capturing num tokens (num_tokens=96 avail_mem=43.93 GB):  81%|████████  | 47/58 [00:07<00:00, 19.92it/s]Capturing num tokens (num_tokens=80 avail_mem=43.92 GB):  81%|████████  | 47/58 [00:07<00:00, 19.92it/s]Capturing num tokens (num_tokens=64 avail_mem=43.91 GB):  81%|████████  | 47/58 [00:07<00:00, 19.92it/s]Capturing num tokens (num_tokens=48 avail_mem=43.91 GB):  81%|████████  | 47/58 [00:07<00:00, 19.92it/s]Capturing num tokens (num_tokens=48 avail_mem=43.91 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.82it/s]Capturing num tokens (num_tokens=32 avail_mem=43.86 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.82it/s]

    Capturing num tokens (num_tokens=28 avail_mem=43.86 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.82it/s]Capturing num tokens (num_tokens=24 avail_mem=43.86 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.82it/s]Capturing num tokens (num_tokens=24 avail_mem=43.86 GB):  91%|█████████▏| 53/58 [00:07<00:00, 23.94it/s]Capturing num tokens (num_tokens=20 avail_mem=43.85 GB):  91%|█████████▏| 53/58 [00:07<00:00, 23.94it/s]Capturing num tokens (num_tokens=16 avail_mem=43.88 GB):  91%|█████████▏| 53/58 [00:07<00:00, 23.94it/s]Capturing num tokens (num_tokens=12 avail_mem=43.87 GB):  91%|█████████▏| 53/58 [00:07<00:00, 23.94it/s]Capturing num tokens (num_tokens=12 avail_mem=43.87 GB):  97%|█████████▋| 56/58 [00:07<00:00, 23.87it/s]Capturing num tokens (num_tokens=8 avail_mem=43.86 GB):  97%|█████████▋| 56/58 [00:07<00:00, 23.87it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=43.86 GB):  97%|█████████▋| 56/58 [00:07<00:00, 23.87it/s]Capturing num tokens (num_tokens=4 avail_mem=43.86 GB): 100%|██████████| 58/58 [00:07<00:00,  7.80it/s]


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
    
    Generated text: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what exactly they're asking for. The capital of France is Paris, so I should focus on that. They want the information and population, so I should gather both the general information about Paris and its current population.
    
    I should consider the user's intent here. Are they looking for a quick fact check? Maybe they're doing a school project, or perhaps they're working on a data visualization. Either way, providing accurate and concise information is key. They specifically mentioned JSON format, so I need to make sure the response is structured correctly with proper keys and values.
    
    Next, I need to ensure the information is up-to-date. Population numbers can change, so I should look up the latest data. As of 2023, I believe Paris has a population around 2.2 million. But I should confirm that. Also, including geographical details like the area and the prefecture can add completeness to the JSON.
    
    I should think about the structure of the JSON. It should have an "info" key with "capital" and "general_info" fields. The "capital" field will have the city's name, and "general_info" can include population, area, and prefecture. This makes the data organized and easy to parse if needed.
    
    I also need to make sure the JSON syntax is correct to avoid any errors. Using proper braces, colons, and commas is essential. Maybe I should double-check the commas after each key-value pair to prevent any syntax issues.
    
    The user might not just want the population figure; they might be interested in where Paris is located. So including the prefecture name, which is Île-de-France, adds context. Also, mentioning that Paris is the largest city in the country and the second most populous in Europe gives a bit more depth to the information.
    
    I should present the information clearly, making sure each field is distinct and easy to read. Keeping the JSON concise but informative is the goal here. No need to add extra fluff; just the necessary data.
    
    Lastly, I'll review the JSON to ensure there are no typos or formatting errors. A correct JSON structure is crucial for any application that might parse this data. Once everything looks good, I can present it back to the user.
    </think>
    
    Here is the information and population of the capital of France (Paris) in JSON format:
    
    ```json
    {
      "info": {
        "capital": "Paris",
        "general_info": {
          "population": 2154000,
          "area": 105.55,
          "prefecture": "Île-de-France"
        },
        "additional": {
          "city特点": "The City of Light and Love, Paris is the capital and most populous city in France. It is the political, cultural, and economic center of the country. Paris is known for its landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral.",
          "statistics": "As of the latest estimates, Paris has a population of around 2.154 million people. The city covers an area of approximately 105.55 square kilometers."
        }
      }
    }
    ```
    
    Let me know if you need further details!



```python
llm.shutdown()
```
