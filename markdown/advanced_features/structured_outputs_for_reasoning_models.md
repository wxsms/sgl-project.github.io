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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.53s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.49s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.49s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:44,  4.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:44,  4.99s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:16,  2.43s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:16,  2.43s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:26,  1.58s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:26,  1.58s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<01:02,  1.15s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<01:02,  1.15s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:47,  1.12it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:47,  1.12it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:31,  1.64it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:31,  1.64it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:25,  1.94it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:25,  1.94it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:21,  2.28it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:21,  2.28it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:18,  2.65it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:18,  2.65it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:15,  3.01it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:15,  3.01it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:13,  3.39it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:13,  3.39it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:11,  3.79it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:11,  3.79it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:10,  4.25it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:10,  4.25it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:09,  4.68it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:09,  4.68it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:08,  5.14it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:08,  5.14it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:07,  5.83it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:07,  5.83it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:10<00:06,  6.38it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:10<00:06,  6.38it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:10<00:06,  6.38it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:10<00:05,  7.58it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:10<00:05,  7.58it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:10<00:05,  7.58it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:10<00:03,  9.13it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:10<00:03,  9.13it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:10<00:03,  9.13it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:10<00:03, 10.84it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:10<00:03, 10.84it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:10<00:03, 10.84it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:10<00:02, 12.28it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:10<00:02, 12.28it/s]

    Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:10<00:02, 12.28it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:10<00:02, 13.51it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:10<00:02, 13.51it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:10<00:02, 13.51it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:10<00:02, 13.51it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:10<00:01, 16.09it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:10<00:01, 16.09it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:11<00:01, 16.09it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:11<00:01, 16.94it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:11<00:01, 16.94it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:11<00:01, 16.94it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:11<00:01, 16.94it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:11<00:01, 18.82it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:11<00:01, 18.82it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:11<00:01, 18.82it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:11<00:01, 18.82it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:11<00:00, 21.60it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:11<00:00, 21.60it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:11<00:00, 21.60it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:11<00:00, 21.60it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:11<00:00, 23.09it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:11<00:00, 23.09it/s]

    Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:11<00:00, 23.09it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:11<00:00, 23.09it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:11<00:00, 24.32it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:11<00:00, 24.32it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:11<00:00, 24.32it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:11<00:00, 24.32it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:11<00:00, 25.53it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:11<00:00, 25.53it/s]

    Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:11<00:00, 25.53it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:11<00:00, 25.53it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:11<00:00, 26.56it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:11<00:00, 26.56it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:11<00:00, 26.56it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:11<00:00, 26.56it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:11<00:00, 26.56it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:11<00:00, 29.49it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:11<00:00, 29.49it/s]

    Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:11<00:00, 29.49it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:11<00:00, 29.49it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  4.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=20.93 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=20.93 GB):   2%|▏         | 1/58 [00:00<00:39,  1.45it/s]Capturing num tokens (num_tokens=7680 avail_mem=20.90 GB):   2%|▏         | 1/58 [00:00<00:39,  1.45it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=20.90 GB):   3%|▎         | 2/58 [00:01<00:35,  1.58it/s]Capturing num tokens (num_tokens=7168 avail_mem=20.89 GB):   3%|▎         | 2/58 [00:01<00:35,  1.58it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=20.89 GB):   5%|▌         | 3/58 [00:01<00:34,  1.58it/s]Capturing num tokens (num_tokens=6656 avail_mem=20.88 GB):   5%|▌         | 3/58 [00:01<00:34,  1.58it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=20.88 GB):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=20.88 GB):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=20.88 GB):   9%|▊         | 5/58 [00:02<00:27,  1.93it/s]Capturing num tokens (num_tokens=5632 avail_mem=20.88 GB):   9%|▊         | 5/58 [00:02<00:27,  1.93it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=20.88 GB):  10%|█         | 6/58 [00:03<00:24,  2.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=20.88 GB):  10%|█         | 6/58 [00:03<00:24,  2.12it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=20.88 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=20.88 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.30it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=20.88 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=20.88 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.54it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=20.88 GB):  16%|█▌        | 9/58 [00:04<00:17,  2.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=20.88 GB):  16%|█▌        | 9/58 [00:04<00:17,  2.78it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=20.88 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=20.88 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=20.88 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]Capturing num tokens (num_tokens=3328 avail_mem=20.87 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=20.87 GB):  21%|██        | 12/58 [00:04<00:09,  4.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=20.87 GB):  21%|██        | 12/58 [00:04<00:09,  4.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=20.87 GB):  22%|██▏       | 13/58 [00:04<00:08,  5.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=20.87 GB):  22%|██▏       | 13/58 [00:04<00:08,  5.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=20.87 GB):  22%|██▏       | 13/58 [00:04<00:08,  5.50it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=20.87 GB):  26%|██▌       | 15/58 [00:04<00:05,  7.33it/s]Capturing num tokens (num_tokens=2304 avail_mem=20.87 GB):  26%|██▌       | 15/58 [00:04<00:05,  7.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=20.87 GB):  26%|██▌       | 15/58 [00:04<00:05,  7.33it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=20.87 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.98 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.98 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.98 GB):  33%|███▎      | 19/58 [00:05<00:05,  7.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.98 GB):  33%|███▎      | 19/58 [00:05<00:05,  7.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.97 GB):  33%|███▎      | 19/58 [00:05<00:05,  7.64it/s]Capturing num tokens (num_tokens=960 avail_mem=41.96 GB):  33%|███▎      | 19/58 [00:05<00:05,  7.64it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=41.96 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.86it/s]Capturing num tokens (num_tokens=896 avail_mem=41.96 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.86it/s]Capturing num tokens (num_tokens=832 avail_mem=41.96 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.86it/s]Capturing num tokens (num_tokens=768 avail_mem=41.95 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.86it/s]Capturing num tokens (num_tokens=768 avail_mem=41.95 GB):  43%|████▎     | 25/58 [00:05<00:02, 14.06it/s]Capturing num tokens (num_tokens=704 avail_mem=41.95 GB):  43%|████▎     | 25/58 [00:05<00:02, 14.06it/s]Capturing num tokens (num_tokens=640 avail_mem=41.94 GB):  43%|████▎     | 25/58 [00:05<00:02, 14.06it/s]Capturing num tokens (num_tokens=576 avail_mem=41.94 GB):  43%|████▎     | 25/58 [00:05<00:02, 14.06it/s]

    Capturing num tokens (num_tokens=576 avail_mem=41.94 GB):  48%|████▊     | 28/58 [00:05<00:01, 17.22it/s]Capturing num tokens (num_tokens=512 avail_mem=41.93 GB):  48%|████▊     | 28/58 [00:05<00:01, 17.22it/s]Capturing num tokens (num_tokens=480 avail_mem=41.93 GB):  48%|████▊     | 28/58 [00:05<00:01, 17.22it/s]Capturing num tokens (num_tokens=448 avail_mem=41.93 GB):  48%|████▊     | 28/58 [00:05<00:01, 17.22it/s]Capturing num tokens (num_tokens=416 avail_mem=41.93 GB):  48%|████▊     | 28/58 [00:05<00:01, 17.22it/s]Capturing num tokens (num_tokens=416 avail_mem=41.93 GB):  55%|█████▌    | 32/58 [00:05<00:01, 21.18it/s]Capturing num tokens (num_tokens=384 avail_mem=41.92 GB):  55%|█████▌    | 32/58 [00:05<00:01, 21.18it/s]Capturing num tokens (num_tokens=352 avail_mem=41.92 GB):  55%|█████▌    | 32/58 [00:05<00:01, 21.18it/s]Capturing num tokens (num_tokens=320 avail_mem=41.91 GB):  55%|█████▌    | 32/58 [00:05<00:01, 21.18it/s]

    Capturing num tokens (num_tokens=288 avail_mem=41.92 GB):  55%|█████▌    | 32/58 [00:06<00:01, 21.18it/s]Capturing num tokens (num_tokens=288 avail_mem=41.92 GB):  62%|██████▏   | 36/58 [00:06<00:00, 24.68it/s]Capturing num tokens (num_tokens=256 avail_mem=41.91 GB):  62%|██████▏   | 36/58 [00:06<00:00, 24.68it/s]Capturing num tokens (num_tokens=240 avail_mem=41.91 GB):  62%|██████▏   | 36/58 [00:06<00:00, 24.68it/s]Capturing num tokens (num_tokens=224 avail_mem=41.91 GB):  62%|██████▏   | 36/58 [00:06<00:00, 24.68it/s]Capturing num tokens (num_tokens=208 avail_mem=41.90 GB):  62%|██████▏   | 36/58 [00:06<00:00, 24.68it/s]Capturing num tokens (num_tokens=208 avail_mem=41.90 GB):  69%|██████▉   | 40/58 [00:06<00:00, 27.84it/s]Capturing num tokens (num_tokens=192 avail_mem=41.90 GB):  69%|██████▉   | 40/58 [00:06<00:00, 27.84it/s]Capturing num tokens (num_tokens=176 avail_mem=41.90 GB):  69%|██████▉   | 40/58 [00:06<00:00, 27.84it/s]Capturing num tokens (num_tokens=160 avail_mem=41.89 GB):  69%|██████▉   | 40/58 [00:06<00:00, 27.84it/s]

    Capturing num tokens (num_tokens=144 avail_mem=41.89 GB):  69%|██████▉   | 40/58 [00:06<00:00, 27.84it/s]Capturing num tokens (num_tokens=144 avail_mem=41.89 GB):  76%|███████▌  | 44/58 [00:06<00:00, 30.23it/s]Capturing num tokens (num_tokens=128 avail_mem=41.89 GB):  76%|███████▌  | 44/58 [00:06<00:00, 30.23it/s]Capturing num tokens (num_tokens=112 avail_mem=41.89 GB):  76%|███████▌  | 44/58 [00:06<00:00, 30.23it/s]Capturing num tokens (num_tokens=96 avail_mem=41.88 GB):  76%|███████▌  | 44/58 [00:06<00:00, 30.23it/s] Capturing num tokens (num_tokens=80 avail_mem=41.88 GB):  76%|███████▌  | 44/58 [00:06<00:00, 30.23it/s]Capturing num tokens (num_tokens=80 avail_mem=41.88 GB):  83%|████████▎ | 48/58 [00:06<00:00, 32.38it/s]Capturing num tokens (num_tokens=64 avail_mem=41.87 GB):  83%|████████▎ | 48/58 [00:06<00:00, 32.38it/s]Capturing num tokens (num_tokens=48 avail_mem=41.87 GB):  83%|████████▎ | 48/58 [00:06<00:00, 32.38it/s]Capturing num tokens (num_tokens=32 avail_mem=41.87 GB):  83%|████████▎ | 48/58 [00:06<00:00, 32.38it/s]

    Capturing num tokens (num_tokens=28 avail_mem=41.87 GB):  83%|████████▎ | 48/58 [00:06<00:00, 32.38it/s]Capturing num tokens (num_tokens=28 avail_mem=41.87 GB):  90%|████████▉ | 52/58 [00:06<00:00, 34.14it/s]Capturing num tokens (num_tokens=24 avail_mem=41.86 GB):  90%|████████▉ | 52/58 [00:06<00:00, 34.14it/s]Capturing num tokens (num_tokens=20 avail_mem=41.86 GB):  90%|████████▉ | 52/58 [00:06<00:00, 34.14it/s]Capturing num tokens (num_tokens=16 avail_mem=41.85 GB):  90%|████████▉ | 52/58 [00:06<00:00, 34.14it/s]Capturing num tokens (num_tokens=12 avail_mem=41.85 GB):  90%|████████▉ | 52/58 [00:06<00:00, 34.14it/s]Capturing num tokens (num_tokens=12 avail_mem=41.85 GB):  97%|█████████▋| 56/58 [00:06<00:00, 35.27it/s]Capturing num tokens (num_tokens=8 avail_mem=41.85 GB):  97%|█████████▋| 56/58 [00:06<00:00, 35.27it/s] Capturing num tokens (num_tokens=4 avail_mem=41.84 GB):  97%|█████████▋| 56/58 [00:06<00:00, 35.27it/s]Capturing num tokens (num_tokens=4 avail_mem=41.84 GB): 100%|██████████| 58/58 [00:06<00:00,  8.75it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants the current date and time along with the weather. Let me break this down step by step.<br><br>First, I need to figure out which functions to use. The user mentioned two specific functions: 'get_current_weather' and 'get_current_date'. Both are available, so I can use them.<br><br>Starting with the date and time. The function 'get_current_date' requires a timezone parameter. Since the user is in New York, I should provide the timezone as 'America/New_York'. That should give me the correct current date and time when the function is called.<br><br>Next, for the weather. The function 'get_current_weather' needs a city, state, and unit. The city is New York, but I should also include the state, which is 'NY'. The unit should be Fahrenheit since the user didn't specify, but it's good to default to a common one. So, I'll structure the parameters accordingly.<br><br>I need to make sure each function call is properly formatted. For the date, it's straightforward: just the timezone. For the weather, I have to include all three parameters in the JSON object. <br><br>Putting it all together, I'll first call 'get_current_date' with the timezone parameter. Then, I'll call 'get_current_weather' with city, state, and unit. I'll make sure each function call is on its own line and follows the specified format with the correct start and end tags.<br><br>I should also remember to add the sources when using any data from these functions, but since the example didn't specify, I might just mention the functions as sources in the response.<br><br>Double-checking everything: functions are called correctly, parameters are included, and the format matches the user's instructions. Looks good!<br><br><br>content: <br><br><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '388025eff3d248a3adadd28d3fc6fb04', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.353237546980381, 'response_sent_to_client_ts': 1780888348.7128487}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': 'f8a9d22a38114595ae755c21bc1e4ae6', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 22.569203708320856, 'response_sent_to_client_ts': 1780888371.2904565}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '017764494d0e449f8e88baa363bcd50d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.19235185906291008, 'response_sent_to_client_ts': 1780888371.5094123}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'b5ccaeac29d04a2ba27a6cbf17e6c345', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.19229348748922348, 'response_sent_to_client_ts': 1780888371.5094244}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '0ce52fd2e44546949d97487c2a0a6a08', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.19225294142961502, 'response_sent_to_client_ts': 1780888371.509428}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '1f60721bda764481b70e8b85ff6426fd', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 17.60744265653193, 'response_sent_to_client_ts': 1780888389.1231065}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the population of the capital of France. I need to make sure I understand exactly what they\'re looking for. They want the information in JSON format, so my response should be a JSON object containing the necessary details.\n\nFirst, who is the capital of France? That\'s straightforward; it\'s Paris. I should include its location coordinates. Paris is right in the center of France, around 48.8566°N and 2.3522°E. That\'s the latitude and longitude.\n\nNext, the population. I need an approximate figure. I recall from previous knowledge that as of 2022, Paris had a population of about 2.2 million people. I should double-check that to make sure it\'s accurate. Maybe I can confirm online or think about recent demographic trends. I\'m pretty confident it\'s around that number, but it\'s important to present it as an approximation unless I can access the exact data right now.\n\nFormatting this information in JSON, I\'ll structure it with the key "Name" as "Paris" and "Population" as the number. Including the location as a separate key seems helpful for quick reference. I should also add a top-level key like "capital_information" to make it clear what the data represents.\n\nThinking about the user\'s possible needs, they might be doing research or a project that requires quick data retrieval. Providing structured data like JSON would be useful for them, as it can be easily parsed by other programs or used in further data processing.\n\nI should also consider the user\'s level of detail. They might not need the full city information, but it\'s helpful to have the coordinates for mapping or geographical studies. Including both the general population and the urban population could add more value, depending on their requirements.\n\nLastly, I should make the response clear and concise, avoiding any unnecessary jargon. Since the instruction was in English, it\'s important that the JSON is correctly formatted and easy to understand.\n</think>\n\n```json\n{\n  "capital_information": {\n    "Name": "Paris",\n    "Population": 2215000,\n    "Location": {\n      "Latitude": 48.8566,\n      "Longitude": 2.3522\n    },\n    "UrbanPopulation": 2156000\n  }\n}\n```', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 7042, 315, 279, 6722, 315, 9625, 13, 358, 1184, 311, 1281, 2704, 358, 3535, 6896, 1128, 807, 2299, 3330, 369, 13, 2379, 1366, 279, 1995, 304, 4718, 3561, 11, 773, 847, 2033, 1265, 387, 264, 4718, 1633, 8482, 279, 5871, 3565, 382, 5338, 11, 879, 374, 279, 6722, 315, 9625, 30, 2938, 594, 30339, 26, 432, 594, 12095, 13, 358, 1265, 2924, 1181, 3728, 13934, 13, 12095, 374, 1290, 304, 279, 4126, 315, 9625, 11, 2163, 220, 19, 23, 13, 23, 20, 21, 21, 11616, 45, 323, 220, 17, 13, 18, 20, 17, 17, 11616, 36, 13, 2938, 594, 279, 20849, 323, 20515, 382, 5847, 11, 279, 7042, 13, 358, 1184, 458, 44868, 7071, 13, 358, 19091, 504, 3681, 6540, 429, 438, 315, 220, 17, 15, 17, 17, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 13, 17, 3526, 1251, 13, 358, 1265, 1990, 15934, 429, 311, 1281, 2704, 432, 594, 13382, 13, 10696, 358, 646, 7683, 2860, 476, 1744, 911, 3213, 37362, 18339, 13, 358, 2776, 5020, 16506, 432, 594, 2163, 429, 1372, 11, 714, 432, 594, 2989, 311, 3042, 432, 438, 458, 56204, 7241, 358, 646, 2615, 279, 4734, 821, 1290, 1431, 382, 82135, 419, 1995, 304, 4718, 11, 358, 3278, 5944, 432, 448, 279, 1376, 330, 675, 1, 438, 330, 59604, 1, 323, 330, 53371, 1, 438, 279, 1372, 13, 55121, 279, 3728, 438, 264, 8651, 1376, 4977, 10950, 369, 3974, 5785, 13, 358, 1265, 1083, 912, 264, 1909, 11591, 1376, 1075, 330, 65063, 35212, 1, 311, 1281, 432, 2797, 1128, 279, 821, 10868, 382, 93945, 911, 279, 1196, 594, 3204, 3880, 11, 807, 2578, 387, 3730, 3412, 476, 264, 2390, 429, 7460, 3974, 821, 56370, 13, 80100, 32930, 821, 1075, 4718, 1035, 387, 5390, 369, 1105, 11, 438, 432, 646, 387, 6707, 15676, 553, 1008, 7468, 476, 1483, 304, 4623, 821, 8692, 382, 40, 1265, 1083, 2908, 279, 1196, 594, 2188, 315, 7716, 13, 2379, 2578, 537, 1184, 279, 2480, 3283, 1995, 11, 714, 432, 594, 10950, 311, 614, 279, 13934, 369, 12731, 476, 52901, 7822, 13, 55121, 2176, 279, 4586, 7042, 323, 279, 15662, 7042, 1410, 912, 803, 897, 11, 11649, 389, 862, 8502, 382, 80486, 11, 358, 1265, 1281, 279, 2033, 2797, 323, 63594, 11, 30426, 894, 25165, 502, 70821, 13, 8704, 279, 7600, 572, 304, 6364, 11, 432, 594, 2989, 429, 279, 4718, 374, 12440, 23126, 323, 4135, 311, 3535, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 65063, 35212, 788, 341, 262, 330, 675, 788, 330, 59604, 756, 262, 330, 53371, 788, 220, 17, 17, 16, 20, 15, 15, 15, 345, 262, 330, 4707, 788, 341, 414, 330, 38011, 788, 220, 19, 23, 13, 23, 20, 21, 21, 345, 414, 330, 39065, 788, 220, 17, 13, 18, 20, 17, 17, 198, 262, 1153, 262, 330, 93268, 53371, 788, 220, 17, 16, 20, 21, 15, 15, 15, 198, 220, 456, 532, 73594, 151643], 'meta_info': {'id': '8eaa65ab1c174a9b89014ab33ce1e434', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 409, 'completion_tokens': 491, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.38084520958364, 'response_sent_to_client_ts': 1780888393.511664}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.29s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.30s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.29s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:42,  4.95s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:42,  4.95s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:01,  2.16s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:01,  2.16s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:09,  1.27s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:09,  1.27s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:45,  1.20it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:45,  1.20it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:31,  1.69it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:31,  1.69it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:22,  2.26it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:22,  2.26it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:17,  2.93it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:17,  2.93it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.69it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.69it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.56it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.56it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.56it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.27it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.27it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.27it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.84it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.84it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:05,  7.95it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:05,  7.95it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:05,  8.09it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:05,  8.09it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:04,  8.45it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:04,  8.45it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:06<00:04,  8.45it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:07<00:04,  9.68it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:07<00:04,  9.68it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:07<00:04,  9.68it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:03, 11.44it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:03, 11.44it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:03, 11.44it/s] 

    Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:03, 11.44it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:02, 15.20it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:02, 15.20it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:02, 15.20it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:07<00:02, 15.20it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:07<00:02, 15.20it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:07<00:01, 20.45it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:07<00:01, 20.45it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:07<00:01, 20.45it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:07<00:01, 20.45it/s]

    Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:07<00:01, 20.45it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:07<00:01, 20.45it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:07<00:00, 26.76it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:07<00:00, 26.76it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:07<00:00, 26.76it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:07<00:00, 26.76it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:07<00:00, 26.76it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:07<00:00, 26.76it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:07<00:00, 32.72it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:07<00:00, 32.72it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:07<00:00, 32.72it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:07<00:00, 32.72it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:07<00:00, 32.72it/s]

    Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:07<00:00, 32.72it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:07<00:00, 32.72it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:07<00:00, 32.72it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 40.73it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 40.73it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 40.73it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 40.73it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 40.73it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 40.73it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 40.73it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 40.73it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:07<00:00, 46.47it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:07<00:00, 46.47it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:07<00:00, 46.47it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:07<00:00, 46.47it/s]

    Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:07<00:00, 46.47it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:07<00:00, 46.47it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:07<00:00, 46.47it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:07<00:00, 46.47it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=56.71 GB):   2%|▏         | 1/58 [00:00<00:26,  2.16it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.42 GB):   2%|▏         | 1/58 [00:00<00:26,  2.16it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=56.42 GB):   3%|▎         | 2/58 [00:00<00:24,  2.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.62 GB):   3%|▎         | 2/58 [00:00<00:24,  2.26it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=56.62 GB):   5%|▌         | 3/58 [00:01<00:22,  2.44it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.49 GB):   5%|▌         | 3/58 [00:01<00:22,  2.44it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.49 GB):   7%|▋         | 4/58 [00:01<00:20,  2.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.60 GB):   7%|▋         | 4/58 [00:01<00:20,  2.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.60 GB):   9%|▊         | 5/58 [00:01<00:18,  2.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.59 GB):   9%|▊         | 5/58 [00:01<00:18,  2.82it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=56.59 GB):  10%|█         | 6/58 [00:02<00:16,  3.08it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.56 GB):  10%|█         | 6/58 [00:02<00:16,  3.08it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=56.56 GB):  12%|█▏        | 7/58 [00:02<00:14,  3.46it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.57 GB):  12%|█▏        | 7/58 [00:02<00:14,  3.46it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=56.57 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.17 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.17 GB):  16%|█▌        | 9/58 [00:02<00:11,  4.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.17 GB):  16%|█▌        | 9/58 [00:02<00:11,  4.30it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=59.17 GB):  17%|█▋        | 10/58 [00:02<00:10,  4.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.16 GB):  17%|█▋        | 10/58 [00:02<00:10,  4.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.16 GB):  19%|█▉        | 11/58 [00:03<00:09,  5.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.16 GB):  19%|█▉        | 11/58 [00:03<00:09,  5.12it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=59.16 GB):  21%|██        | 12/58 [00:03<00:07,  5.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.15 GB):  21%|██        | 12/58 [00:03<00:07,  5.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.15 GB):  22%|██▏       | 13/58 [00:03<00:07,  6.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.14 GB):  22%|██▏       | 13/58 [00:03<00:07,  6.39it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=59.14 GB):  24%|██▍       | 14/58 [00:03<00:06,  7.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.13 GB):  24%|██▍       | 14/58 [00:03<00:06,  7.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.13 GB):  26%|██▌       | 15/58 [00:03<00:05,  7.44it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.12 GB):  26%|██▌       | 15/58 [00:03<00:05,  7.44it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=59.12 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.63it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.09 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.63it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.11 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.63it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.11 GB):  31%|███       | 18/58 [00:03<00:04,  9.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.08 GB):  31%|███       | 18/58 [00:03<00:04,  9.49it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=59.10 GB):  31%|███       | 18/58 [00:03<00:04,  9.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.10 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.08 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.51it/s]Capturing num tokens (num_tokens=960 avail_mem=59.07 GB):  34%|███▍      | 20/58 [00:04<00:03, 11.51it/s] Capturing num tokens (num_tokens=896 avail_mem=59.06 GB):  34%|███▍      | 20/58 [00:04<00:03, 11.51it/s]Capturing num tokens (num_tokens=896 avail_mem=59.06 GB):  40%|███▉      | 23/58 [00:04<00:02, 14.56it/s]Capturing num tokens (num_tokens=832 avail_mem=59.06 GB):  40%|███▉      | 23/58 [00:04<00:02, 14.56it/s]

    Capturing num tokens (num_tokens=768 avail_mem=59.05 GB):  40%|███▉      | 23/58 [00:04<00:02, 14.56it/s]Capturing num tokens (num_tokens=704 avail_mem=59.04 GB):  40%|███▉      | 23/58 [00:04<00:02, 14.56it/s]Capturing num tokens (num_tokens=704 avail_mem=59.04 GB):  45%|████▍     | 26/58 [00:04<00:01, 17.07it/s]Capturing num tokens (num_tokens=640 avail_mem=59.04 GB):  45%|████▍     | 26/58 [00:04<00:01, 17.07it/s]Capturing num tokens (num_tokens=576 avail_mem=59.03 GB):  45%|████▍     | 26/58 [00:04<00:01, 17.07it/s]Capturing num tokens (num_tokens=512 avail_mem=59.02 GB):  45%|████▍     | 26/58 [00:04<00:01, 17.07it/s]

    Capturing num tokens (num_tokens=512 avail_mem=59.02 GB):  50%|█████     | 29/58 [00:04<00:01, 19.48it/s]Capturing num tokens (num_tokens=480 avail_mem=59.01 GB):  50%|█████     | 29/58 [00:04<00:01, 19.48it/s]Capturing num tokens (num_tokens=448 avail_mem=59.01 GB):  50%|█████     | 29/58 [00:04<00:01, 19.48it/s]Capturing num tokens (num_tokens=416 avail_mem=59.00 GB):  50%|█████     | 29/58 [00:04<00:01, 19.48it/s]Capturing num tokens (num_tokens=384 avail_mem=59.00 GB):  50%|█████     | 29/58 [00:04<00:01, 19.48it/s]Capturing num tokens (num_tokens=384 avail_mem=59.00 GB):  57%|█████▋    | 33/58 [00:04<00:01, 23.30it/s]Capturing num tokens (num_tokens=352 avail_mem=58.99 GB):  57%|█████▋    | 33/58 [00:04<00:01, 23.30it/s]Capturing num tokens (num_tokens=320 avail_mem=58.99 GB):  57%|█████▋    | 33/58 [00:04<00:01, 23.30it/s]Capturing num tokens (num_tokens=288 avail_mem=59.00 GB):  57%|█████▋    | 33/58 [00:04<00:01, 23.30it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.99 GB):  57%|█████▋    | 33/58 [00:04<00:01, 23.30it/s]Capturing num tokens (num_tokens=256 avail_mem=58.99 GB):  64%|██████▍   | 37/58 [00:04<00:00, 27.11it/s]Capturing num tokens (num_tokens=240 avail_mem=58.99 GB):  64%|██████▍   | 37/58 [00:04<00:00, 27.11it/s]Capturing num tokens (num_tokens=224 avail_mem=58.98 GB):  64%|██████▍   | 37/58 [00:04<00:00, 27.11it/s]Capturing num tokens (num_tokens=208 avail_mem=58.98 GB):  64%|██████▍   | 37/58 [00:04<00:00, 27.11it/s]Capturing num tokens (num_tokens=192 avail_mem=58.98 GB):  64%|██████▍   | 37/58 [00:04<00:00, 27.11it/s]Capturing num tokens (num_tokens=192 avail_mem=58.98 GB):  71%|███████   | 41/58 [00:04<00:00, 30.21it/s]Capturing num tokens (num_tokens=176 avail_mem=58.97 GB):  71%|███████   | 41/58 [00:04<00:00, 30.21it/s]Capturing num tokens (num_tokens=160 avail_mem=58.97 GB):  71%|███████   | 41/58 [00:04<00:00, 30.21it/s]Capturing num tokens (num_tokens=144 avail_mem=58.96 GB):  71%|███████   | 41/58 [00:04<00:00, 30.21it/s]

    Capturing num tokens (num_tokens=128 avail_mem=58.97 GB):  71%|███████   | 41/58 [00:04<00:00, 30.21it/s]Capturing num tokens (num_tokens=128 avail_mem=58.97 GB):  78%|███████▊  | 45/58 [00:04<00:00, 32.74it/s]Capturing num tokens (num_tokens=112 avail_mem=58.96 GB):  78%|███████▊  | 45/58 [00:04<00:00, 32.74it/s]Capturing num tokens (num_tokens=96 avail_mem=58.96 GB):  78%|███████▊  | 45/58 [00:04<00:00, 32.74it/s] Capturing num tokens (num_tokens=80 avail_mem=58.95 GB):  78%|███████▊  | 45/58 [00:04<00:00, 32.74it/s]Capturing num tokens (num_tokens=64 avail_mem=58.95 GB):  78%|███████▊  | 45/58 [00:04<00:00, 32.74it/s]Capturing num tokens (num_tokens=64 avail_mem=58.95 GB):  84%|████████▍ | 49/58 [00:04<00:00, 34.44it/s]Capturing num tokens (num_tokens=48 avail_mem=58.95 GB):  84%|████████▍ | 49/58 [00:04<00:00, 34.44it/s]Capturing num tokens (num_tokens=32 avail_mem=58.94 GB):  84%|████████▍ | 49/58 [00:04<00:00, 34.44it/s]Capturing num tokens (num_tokens=28 avail_mem=58.94 GB):  84%|████████▍ | 49/58 [00:04<00:00, 34.44it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.94 GB):  84%|████████▍ | 49/58 [00:04<00:00, 34.44it/s]Capturing num tokens (num_tokens=20 avail_mem=58.93 GB):  84%|████████▍ | 49/58 [00:04<00:00, 34.44it/s]Capturing num tokens (num_tokens=20 avail_mem=58.93 GB):  93%|█████████▎| 54/58 [00:05<00:00, 36.55it/s]Capturing num tokens (num_tokens=16 avail_mem=58.93 GB):  93%|█████████▎| 54/58 [00:05<00:00, 36.55it/s]Capturing num tokens (num_tokens=12 avail_mem=58.93 GB):  93%|█████████▎| 54/58 [00:05<00:00, 36.55it/s]Capturing num tokens (num_tokens=8 avail_mem=58.92 GB):  93%|█████████▎| 54/58 [00:05<00:00, 36.55it/s] Capturing num tokens (num_tokens=4 avail_mem=58.92 GB):  93%|█████████▎| 54/58 [00:05<00:00, 36.55it/s]Capturing num tokens (num_tokens=4 avail_mem=58.92 GB): 100%|██████████| 58/58 [00:05<00:00, 37.35it/s]Capturing num tokens (num_tokens=4 avail_mem=58.92 GB): 100%|██████████| 58/58 [00:05<00:00, 11.37it/s]


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
    
    Generated text: Alright, so the user asked for the information and population of the capital of France in JSON format. Let me break this down.
    
    First, I need to identify what the capital of France is. That's straightforward—Paris is the capital. So, I'll set up the JSON object with a "capital" key and assign it the value "Paris".
    
    Next, the user also asked for the population. I should provide a recent estimate, preferably from 2023. Paris is a major city, so its population is significant. I recall that the population is around 2.1 million, but I should verify that. Checking some reliable sources, I confirm that the population is approximately 2,155,000 as of 2023.
    
    Now, I need to structure this information into a JSON format. JSON typically uses key-value pairs, so I'll create an object with "capital" and "population" as keys. The population should be a number, so I'll write it without quotes. The capital is a string, so it'll be in quotes.
    
    Putting it all together, the JSON object will look like this:
    {
      "capital": "Paris",
      "population": 2155000
    }
    
    I should also make sure that the response is clear and concise, without any unnecessary details. The user might be using this information for a report or a presentation, so accuracy is crucial. Double-checking the population number and the current status of Paris as the capital ensures the information is up-to-date.
    
    Additionally, since the user requested JSON format, I'll present it without any markdown, as per their instruction. Keeping it simple and straightforward will meet their needs effectively.
    </think>
    
    ```json
    {
      "capital": "Paris",
      "population": 2155000
    }
    ```



```python
llm.shutdown()
```
