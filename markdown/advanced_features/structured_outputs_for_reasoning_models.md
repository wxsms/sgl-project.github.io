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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.98s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.78s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.81s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:17,  2.45s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:17,  2.45s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:29,  1.63s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:29,  1.63s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<01:05,  1.22s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<01:05,  1.22s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:51,  1.03it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:51,  1.03it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:42,  1.23it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:42,  1.23it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:34,  1.46it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:34,  1.46it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:29,  1.68it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:29,  1.68it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:25,  1.93it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:25,  1.93it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:21,  2.20it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:21,  2.20it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:19,  2.45it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:19,  2.45it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:16,  2.73it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:16,  2.73it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:10<00:14,  3.03it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:10<00:14,  3.03it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:10<00:13,  3.34it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:10<00:13,  3.34it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:10<00:11,  3.73it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:10<00:11,  3.73it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:10<00:10,  4.17it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:10<00:10,  4.17it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:10<00:08,  4.63it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:10<00:08,  4.63it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:11<00:07,  5.18it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:11<00:07,  5.18it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:11<00:06,  5.83it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:11<00:06,  5.83it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:11<00:05,  6.65it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:11<00:05,  6.65it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:11<00:05,  6.65it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:11<00:04,  8.41it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:11<00:04,  8.41it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:11<00:04,  8.41it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:11<00:03, 10.08it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:11<00:03, 10.08it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:11<00:03, 10.08it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:11<00:02, 11.95it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:11<00:02, 11.95it/s]

    Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:11<00:02, 11.95it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:11<00:02, 13.56it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:11<00:02, 13.56it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:11<00:02, 13.56it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:11<00:02, 13.56it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:11<00:01, 16.38it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:11<00:01, 16.38it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:12<00:01, 16.38it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:12<00:01, 16.38it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:12<00:01, 18.77it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:12<00:01, 18.77it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:12<00:01, 18.77it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:12<00:01, 18.77it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:12<00:00, 21.37it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:12<00:00, 21.37it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:12<00:00, 21.37it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:12<00:00, 21.37it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:12<00:00, 21.37it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:12<00:00, 24.88it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:12<00:00, 24.88it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:12<00:00, 24.88it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:12<00:00, 24.88it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:12<00:00, 24.88it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:12<00:00, 27.43it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:12<00:00, 27.43it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:12<00:00, 27.43it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:12<00:00, 27.43it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:12<00:00, 27.43it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:12<00:00, 29.23it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:12<00:00, 29.23it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:12<00:00, 29.23it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:12<00:00, 29.23it/s]

    Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:12<00:00, 29.23it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:12<00:00, 30.87it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:12<00:00, 30.87it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:12<00:00, 30.87it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:12<00:00, 30.87it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:12<00:00, 30.87it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:12<00:00, 30.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:12<00:00,  4.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.70 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.70 GB):   3%|▎         | 2/58 [00:01<00:48,  1.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   3%|▎         | 2/58 [00:01<00:48,  1.14it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   5%|▌         | 3/58 [00:02<00:44,  1.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.83 GB):   5%|▌         | 3/58 [00:02<00:44,  1.22it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.83 GB):   7%|▋         | 4/58 [00:03<00:41,  1.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.90 GB):   7%|▋         | 4/58 [00:03<00:41,  1.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.90 GB):   9%|▊         | 5/58 [00:03<00:38,  1.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.96 GB):   9%|▊         | 5/58 [00:03<00:38,  1.39it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.96 GB):  10%|█         | 6/58 [00:04<00:34,  1.52it/s]Capturing num tokens (num_tokens=5120 avail_mem=26.02 GB):  10%|█         | 6/58 [00:04<00:34,  1.52it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=26.02 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.43 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.65it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=25.43 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=25.56 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.81it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=25.56 GB):  16%|█▌        | 9/58 [00:05<00:24,  1.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.66 GB):  16%|█▌        | 9/58 [00:05<00:24,  1.99it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=25.66 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=25.72 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.15it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=25.72 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=25.78 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.34it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=25.78 GB):  21%|██        | 12/58 [00:06<00:18,  2.54it/s]Capturing num tokens (num_tokens=3072 avail_mem=26.26 GB):  21%|██        | 12/58 [00:06<00:18,  2.54it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=26.26 GB):  22%|██▏       | 13/58 [00:07<00:16,  2.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=25.90 GB):  22%|██▏       | 13/58 [00:07<00:16,  2.67it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=25.90 GB):  24%|██▍       | 14/58 [00:07<00:14,  2.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.29 GB):  24%|██▍       | 14/58 [00:07<00:14,  2.97it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.29 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.32 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.32 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.34 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.70it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=26.34 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.37 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.37 GB):  31%|███       | 18/58 [00:08<00:08,  4.63it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.39 GB):  31%|███       | 18/58 [00:08<00:08,  4.63it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=26.39 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.40 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.40 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.94it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.41 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.94it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=26.41 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.75it/s]Capturing num tokens (num_tokens=960 avail_mem=26.44 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.75it/s] Capturing num tokens (num_tokens=896 avail_mem=26.44 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.75it/s]Capturing num tokens (num_tokens=896 avail_mem=26.44 GB):  40%|███▉      | 23/58 [00:08<00:04,  8.21it/s]Capturing num tokens (num_tokens=832 avail_mem=26.43 GB):  40%|███▉      | 23/58 [00:08<00:04,  8.21it/s]

    Capturing num tokens (num_tokens=768 avail_mem=26.43 GB):  40%|███▉      | 23/58 [00:08<00:04,  8.21it/s]Capturing num tokens (num_tokens=768 avail_mem=26.43 GB):  43%|████▎     | 25/58 [00:08<00:03,  9.24it/s]Capturing num tokens (num_tokens=704 avail_mem=26.42 GB):  43%|████▎     | 25/58 [00:08<00:03,  9.24it/s]Capturing num tokens (num_tokens=640 avail_mem=26.42 GB):  43%|████▎     | 25/58 [00:08<00:03,  9.24it/s]

    Capturing num tokens (num_tokens=640 avail_mem=26.42 GB):  47%|████▋     | 27/58 [00:08<00:03, 10.08it/s]Capturing num tokens (num_tokens=576 avail_mem=26.51 GB):  47%|████▋     | 27/58 [00:08<00:03, 10.08it/s]Capturing num tokens (num_tokens=512 avail_mem=26.49 GB):  47%|████▋     | 27/58 [00:09<00:03, 10.08it/s]Capturing num tokens (num_tokens=512 avail_mem=26.49 GB):  50%|█████     | 29/58 [00:09<00:02, 11.39it/s]Capturing num tokens (num_tokens=480 avail_mem=26.48 GB):  50%|█████     | 29/58 [00:09<00:02, 11.39it/s]Capturing num tokens (num_tokens=448 avail_mem=26.48 GB):  50%|█████     | 29/58 [00:09<00:02, 11.39it/s]

    Capturing num tokens (num_tokens=448 avail_mem=26.48 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.26it/s]Capturing num tokens (num_tokens=416 avail_mem=26.46 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.26it/s]Capturing num tokens (num_tokens=384 avail_mem=26.40 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.26it/s]Capturing num tokens (num_tokens=384 avail_mem=26.40 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.54it/s]Capturing num tokens (num_tokens=352 avail_mem=26.40 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.54it/s]Capturing num tokens (num_tokens=320 avail_mem=26.40 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.54it/s]

    Capturing num tokens (num_tokens=288 avail_mem=26.41 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.54it/s]Capturing num tokens (num_tokens=288 avail_mem=26.41 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.33it/s]Capturing num tokens (num_tokens=256 avail_mem=26.40 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.33it/s]Capturing num tokens (num_tokens=240 avail_mem=26.38 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.33it/s]Capturing num tokens (num_tokens=240 avail_mem=26.38 GB):  66%|██████▌   | 38/58 [00:09<00:01, 16.35it/s]Capturing num tokens (num_tokens=224 avail_mem=26.37 GB):  66%|██████▌   | 38/58 [00:09<00:01, 16.35it/s]

    Capturing num tokens (num_tokens=208 avail_mem=26.36 GB):  66%|██████▌   | 38/58 [00:09<00:01, 16.35it/s]Capturing num tokens (num_tokens=208 avail_mem=26.36 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.11it/s]Capturing num tokens (num_tokens=192 avail_mem=26.35 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.11it/s]Capturing num tokens (num_tokens=176 avail_mem=26.34 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.11it/s]Capturing num tokens (num_tokens=160 avail_mem=26.35 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.11it/s]Capturing num tokens (num_tokens=160 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.12it/s]Capturing num tokens (num_tokens=144 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.12it/s]

    Capturing num tokens (num_tokens=128 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.12it/s]Capturing num tokens (num_tokens=112 avail_mem=26.34 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.12it/s]Capturing num tokens (num_tokens=112 avail_mem=26.34 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.17it/s]Capturing num tokens (num_tokens=96 avail_mem=26.33 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.17it/s] Capturing num tokens (num_tokens=80 avail_mem=26.32 GB):  79%|███████▉  | 46/58 [00:10<00:00, 19.17it/s]Capturing num tokens (num_tokens=80 avail_mem=26.32 GB):  83%|████████▎ | 48/58 [00:10<00:00, 19.35it/s]Capturing num tokens (num_tokens=64 avail_mem=26.31 GB):  83%|████████▎ | 48/58 [00:10<00:00, 19.35it/s]

    Capturing num tokens (num_tokens=48 avail_mem=26.30 GB):  83%|████████▎ | 48/58 [00:10<00:00, 19.35it/s]Capturing num tokens (num_tokens=32 avail_mem=26.29 GB):  83%|████████▎ | 48/58 [00:10<00:00, 19.35it/s]Capturing num tokens (num_tokens=32 avail_mem=26.29 GB):  88%|████████▊ | 51/58 [00:10<00:00, 19.92it/s]Capturing num tokens (num_tokens=28 avail_mem=26.28 GB):  88%|████████▊ | 51/58 [00:10<00:00, 19.92it/s]Capturing num tokens (num_tokens=24 avail_mem=26.27 GB):  88%|████████▊ | 51/58 [00:10<00:00, 19.92it/s]Capturing num tokens (num_tokens=20 avail_mem=26.26 GB):  88%|████████▊ | 51/58 [00:10<00:00, 19.92it/s]

    Capturing num tokens (num_tokens=20 avail_mem=26.26 GB):  93%|█████████▎| 54/58 [00:10<00:00, 20.66it/s]Capturing num tokens (num_tokens=16 avail_mem=26.24 GB):  93%|█████████▎| 54/58 [00:10<00:00, 20.66it/s]Capturing num tokens (num_tokens=12 avail_mem=26.23 GB):  93%|█████████▎| 54/58 [00:10<00:00, 20.66it/s]Capturing num tokens (num_tokens=8 avail_mem=26.23 GB):  93%|█████████▎| 54/58 [00:10<00:00, 20.66it/s] Capturing num tokens (num_tokens=8 avail_mem=26.23 GB):  98%|█████████▊| 57/58 [00:10<00:00, 21.16it/s]Capturing num tokens (num_tokens=4 avail_mem=26.21 GB):  98%|█████████▊| 57/58 [00:10<00:00, 21.16it/s]Capturing num tokens (num_tokens=4 avail_mem=26.21 GB): 100%|██████████| 58/58 [00:10<00:00,  5.51it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should check if that's correct.<br><br>Wait, I think the population might have changed a bit over the years. I recall reading somewhere that Paris has grown a lot, especially with the influx of people moving there for work. So maybe it's more than 20 million now. I'm trying to remember if it's closer to 21 or 22 million. I think it's around 21.5 million, but I'm not 100% sure. I should probably look up the latest data to confirm.<br><br>Looking it up, I see that as of 2023, the population of Paris is approximately 21,609,357. That's about 21.6 million. So I was close with my initial thought of around 21.5 million. I guess the number has increased a bit since I last heard about it. It makes sense because Paris is a major economic hub and attracts a lot of people for jobs and studies.<br><br>I should also consider the source of this information. Population numbers can vary depending on when they're measured and how they're reported. It's important to use reliable sources like official statistics from the French National Institute of Statistics and Registration (INSEE) or other reputable organizations that track population data.<br><br>So, putting it all together, the capital of France is Paris, and its population is approximately 21.6 million as of the latest data. I should present this information in a clear and concise JSON format, making sure to include both the city name and the population number accurately.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21609357<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think the population has been growing over the years. I recall that in recent years, Paris has been increasing its population. Maybe it's over 21 million now? I'm not sure if it's 2021 or 2022 data. Also, I should consider whether the population figure includes just the city proper or the entire metropolitan area. Sometimes, people talk about the metro area, which might be larger.<br><br>I think the metropolitan area of Paris is bigger, maybe around 12 million? But the city itself is smaller. I'm a bit confused about the exact numbers. I should look up the latest data to be accurate. But since I can't access the internet right now, I'll have to go with my best guess based on what I remember.<br><br>So, putting it all together, the capital is Paris, and the population is approximately 21 million. I'll present this information in JSON format as requested.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21000000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light." I've heard people talk about the Eiffel Tower being in Paris, which is a famous landmark. But is Paris the capital? I think it is, but I'm not entirely certain. Maybe I should consider other major cities in France. There's Lyon, which I think is the second largest city, and then there's Marseille. But I don't recall hearing them referred to as capitals. Then there's also the capital region, which includes Paris, but I believe the actual capital is just Paris itself. I don't think it's a region or a larger city. So, putting it all together, I'm pretty confident that Paris is the capital of France.<br><br><br>content: Paris</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants the current date and time, along with the weather. Let me figure out how to approach this.<br><br>First, I need to determine which functions to use. The user mentioned two specific functions: 'get_current_date' and 'get_current_weather'. Both are available, so I should use them.<br><br>Starting with the date and time. The function requires a timezone parameter. New York is in the 'America/New_York' timezone, so that's straightforward. I'll structure the function call with timezone set to 'America/New_York'.<br><br>Next, for the weather. The function needs a city and state, along with the unit. The user is in New York, so the city is 'New York' and the state is 'NY'. The unit isn't specified, so I'll default to Fahrenheit since that's a common preference, but I should note that if Celsius is needed, the user can specify.<br><br>Putting it all together, I'll make two separate function calls. The first will fetch the current date and time in New York's timezone, and the second will get the weather in Fahrenheit.<br><br>I should also include the sources for each function call to provide transparency and help the user if they need more information. So, I'll add a note about the functions used and their respective parameters.<br><br>Finally, I'll present the answers clearly, making sure each function call is on a new line and properly formatted with the start and end tags as specified.<br><br><br>content: <br><br><function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>  <br><br>Sources:  <br>- get_current_date: Fetches current date and time for a given timezone  <br>- get_current_weather: Fetches current weather in a specified location, including temperature unit  <br><br>Let me know if you need further details!</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '5eeabd9e1a214de4a6c8ef0bb56988ba', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.121878102421761, 'response_sent_to_client_ts': 1779706496.0267196}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '960785c9197b4913bdda04a1380f4a03', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 26.57575847953558, 'response_sent_to_client_ts': 1779706522.6120863}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'a00d0783f909487d9792ef7179802d05', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.10760119184851646, 'response_sent_to_client_ts': 1779706522.76176}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '06af90e29c494b09af12af6ff84398b7', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.10751363635063171, 'response_sent_to_client_ts': 1779706522.7617753}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '9d8e26e4fc444eabaf68c2c66fded03c', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.10747489891946316, 'response_sent_to_client_ts': 1779706522.7617795}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'a9204f625255478aa428001ff67e2733', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.58288929797709, 'response_sent_to_client_ts': 1779706542.3524983}}


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


<strong style='color: #00008B;'>{'text': 'Okay, the user is asking for the information and population of the capital of France. Since France\'s capital is Paris, I need to find the latest population data. I should check a reliable source, maybe the World Bank or a recent census. I remember that the population has been growing steadily.\n\nI think as of 2023, Paris has a population around 2 million, but I\'m not 100% sure. I should double-check the numbers to make sure they\'re accurate. Also, including some key facts about Paris would be helpful, like its geographical location, languages spoken, and cultural significance.\n\nThe user didn\'t specify the format beyond JSON, but providing a structured response would be best. I\'ll create a JSON object with population and other relevant information. I\'ll also include an example of how to use the data, showing that it\'s easy to access the information from the JSON.\n\nI should ensure that the information is clear and concise, avoiding any unnecessary jargon. This way, the user can quickly understand the essential details about Paris\'s population and its importance. I need to present the information accurately to meet the user\'s needs effectively.\n</think>\n\nCertainly! The capital of France is Paris. Here is the information in JSON format:\n\n```json\n{\n  "capital": "Paris",\n  "population": "Approximately 2,200,000 as of 2023 estimates",\n  "factoid": "Paris is the largest city in France and one of the most populous cities in the world, located in northern France near the Seine River. The city has a rich history, cultural significance, and a diverse population with major languages including French, English, Spanish, and Italian."\n}\n```\n\nThis JSON object contains the population of Paris and some additional contextual information. Let me know if you need further details!', 'output_ids': [32313, 11, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 13, 8704, 9625, 594, 6722, 374, 12095, 11, 358, 1184, 311, 1477, 279, 5535, 7042, 821, 13, 358, 1265, 1779, 264, 14720, 2530, 11, 7196, 279, 4337, 8547, 476, 264, 3213, 43602, 13, 358, 6099, 429, 279, 7042, 702, 1012, 7826, 41971, 382, 40, 1744, 438, 315, 220, 17, 15, 17, 18, 11, 12095, 702, 264, 7042, 2163, 220, 17, 3526, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 2704, 13, 358, 1265, 1990, 15934, 279, 5109, 311, 1281, 2704, 807, 2299, 13382, 13, 7281, 11, 2670, 1045, 1376, 13064, 911, 12095, 1035, 387, 10950, 11, 1075, 1181, 52901, 3728, 11, 15459, 21355, 11, 323, 12752, 25361, 382, 785, 1196, 3207, 944, 13837, 279, 3561, 7797, 4718, 11, 714, 8241, 264, 32930, 2033, 1035, 387, 1850, 13, 358, 3278, 1855, 264, 4718, 1633, 448, 7042, 323, 1008, 9760, 1995, 13, 358, 3278, 1083, 2924, 458, 3110, 315, 1246, 311, 990, 279, 821, 11, 9027, 429, 432, 594, 4135, 311, 2615, 279, 1995, 504, 279, 4718, 382, 40, 1265, 5978, 429, 279, 1995, 374, 2797, 323, 63594, 11, 30426, 894, 25165, 502, 70821, 13, 1096, 1616, 11, 279, 1196, 646, 6157, 3535, 279, 7565, 3565, 911, 12095, 594, 7042, 323, 1181, 12650, 13, 358, 1184, 311, 3042, 279, 1995, 29257, 311, 3367, 279, 1196, 594, 3880, 13444, 624, 151649, 271, 95456, 0, 576, 6722, 315, 9625, 374, 12095, 13, 5692, 374, 279, 1995, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 330, 69520, 7108, 220, 17, 11, 17, 15, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 17530, 756, 220, 330, 33110, 588, 788, 330, 59604, 374, 279, 7772, 3283, 304, 9625, 323, 825, 315, 279, 1429, 94451, 9720, 304, 279, 1879, 11, 7407, 304, 18172, 9625, 3143, 279, 1345, 482, 10948, 13, 576, 3283, 702, 264, 9080, 3840, 11, 12752, 25361, 11, 323, 264, 16807, 7042, 448, 3598, 15459, 2670, 8585, 11, 6364, 11, 15154, 11, 323, 14811, 10040, 532, 13874, 19324, 1986, 4718, 1633, 5610, 279, 7042, 315, 12095, 323, 1045, 5107, 65151, 1995, 13, 6771, 752, 1414, 421, 498, 1184, 4623, 3565, 0, 151643], 'meta_info': {'id': 'ddacb73d8155459ab3f67b044ba52d2d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 236, 'completion_tokens': 379, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.559210380539298, 'response_sent_to_client_ts': 1779706545.919937}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.53s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.51s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.52s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:29,  5.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:29,  5.78s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:21,  2.53s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:21,  2.53s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:20,  1.47s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:20,  1.47s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:52,  1.04it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:52,  1.04it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:25,  2.01it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:25,  2.01it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:19,  2.65it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:19,  2.65it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.37it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.37it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:11,  4.21it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:11,  4.21it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:07<00:11,  4.21it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:08,  5.85it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:08,  5.85it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:08,  5.85it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.03it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.03it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.03it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]

    Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.13it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.13it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.13it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.13it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.13it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.37it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.37it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.37it/s]

    Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:08<00:01, 19.37it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:08<00:01, 19.37it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:08<00:01, 19.37it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:08<00:01, 25.79it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:08<00:01, 25.79it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:08<00:01, 25.79it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:08<00:01, 25.79it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:08<00:01, 25.79it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:08<00:01, 25.79it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:08<00:01, 25.79it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:08<00:00, 33.60it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:08<00:00, 33.60it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:08<00:00, 33.60it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:08<00:00, 33.60it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:08<00:00, 33.60it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:08<00:00, 33.60it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:08<00:00, 33.60it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:08<00:00, 40.37it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:08<00:00, 40.37it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:08<00:00, 40.37it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:08<00:00, 40.37it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:08<00:00, 40.37it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:08<00:00, 40.37it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:08<00:00, 40.37it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:08<00:00, 44.30it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:08<00:00, 44.30it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:08<00:00, 44.30it/s]

    Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:08<00:00, 44.30it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:08<00:00, 44.30it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:08<00:00, 44.30it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:08<00:00, 44.30it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:08<00:00, 48.26it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:08<00:00, 48.26it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:08<00:00, 48.26it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:08<00:00, 48.26it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:08<00:00, 48.26it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:08<00:00, 48.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=33.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=33.75 GB):   2%|▏         | 1/58 [00:00<00:21,  2.71it/s]Capturing num tokens (num_tokens=7680 avail_mem=33.72 GB):   2%|▏         | 1/58 [00:00<00:21,  2.71it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=33.72 GB):   3%|▎         | 2/58 [00:00<00:18,  3.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=51.98 GB):   3%|▎         | 2/58 [00:00<00:18,  3.10it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=51.98 GB):   5%|▌         | 3/58 [00:00<00:15,  3.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=51.98 GB):   5%|▌         | 3/58 [00:00<00:15,  3.45it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=51.98 GB):   7%|▋         | 4/58 [00:01<00:14,  3.78it/s]Capturing num tokens (num_tokens=6144 avail_mem=51.98 GB):   7%|▋         | 4/58 [00:01<00:14,  3.78it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=51.98 GB):   9%|▊         | 5/58 [00:01<00:13,  4.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=51.98 GB):   9%|▊         | 5/58 [00:01<00:13,  4.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=51.98 GB):  10%|█         | 6/58 [00:01<00:11,  4.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=51.97 GB):  10%|█         | 6/58 [00:01<00:11,  4.49it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=51.97 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=51.98 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=51.98 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=51.98 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.38it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=51.98 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=51.98 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=51.98 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=51.97 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.33it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=51.97 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=51.97 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=51.97 GB):  21%|██        | 12/58 [00:02<00:06,  7.37it/s]Capturing num tokens (num_tokens=3072 avail_mem=51.97 GB):  21%|██        | 12/58 [00:02<00:06,  7.37it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=51.97 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=51.97 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=51.97 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=51.97 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.22it/s]Capturing num tokens (num_tokens=2304 avail_mem=51.96 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.22it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=51.96 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=51.96 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.61it/s]Capturing num tokens (num_tokens=1792 avail_mem=51.96 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=51.95 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=51.95 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=51.95 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.01it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=51.94 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.01it/s]Capturing num tokens (num_tokens=960 avail_mem=51.94 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.01it/s] Capturing num tokens (num_tokens=960 avail_mem=51.94 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.05it/s]Capturing num tokens (num_tokens=896 avail_mem=51.93 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.05it/s]Capturing num tokens (num_tokens=832 avail_mem=51.93 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.05it/s]Capturing num tokens (num_tokens=768 avail_mem=51.93 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.05it/s]

    Capturing num tokens (num_tokens=768 avail_mem=51.93 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.80it/s]Capturing num tokens (num_tokens=704 avail_mem=51.92 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.80it/s]Capturing num tokens (num_tokens=640 avail_mem=51.92 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.80it/s]Capturing num tokens (num_tokens=576 avail_mem=51.92 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.80it/s]Capturing num tokens (num_tokens=576 avail_mem=51.92 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.34it/s]Capturing num tokens (num_tokens=512 avail_mem=51.91 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.34it/s]Capturing num tokens (num_tokens=480 avail_mem=51.91 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.34it/s]Capturing num tokens (num_tokens=448 avail_mem=51.91 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.34it/s]

    Capturing num tokens (num_tokens=448 avail_mem=51.91 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.56it/s]Capturing num tokens (num_tokens=416 avail_mem=51.90 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.56it/s]Capturing num tokens (num_tokens=384 avail_mem=51.90 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.56it/s]Capturing num tokens (num_tokens=352 avail_mem=51.89 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.56it/s]Capturing num tokens (num_tokens=352 avail_mem=51.89 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.07it/s]Capturing num tokens (num_tokens=320 avail_mem=51.89 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.07it/s]Capturing num tokens (num_tokens=288 avail_mem=51.90 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.07it/s]Capturing num tokens (num_tokens=256 avail_mem=51.89 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.07it/s]

    Capturing num tokens (num_tokens=240 avail_mem=51.89 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.07it/s]Capturing num tokens (num_tokens=240 avail_mem=51.89 GB):  66%|██████▌   | 38/58 [00:03<00:00, 23.64it/s]Capturing num tokens (num_tokens=224 avail_mem=51.88 GB):  66%|██████▌   | 38/58 [00:03<00:00, 23.64it/s]Capturing num tokens (num_tokens=208 avail_mem=51.88 GB):  66%|██████▌   | 38/58 [00:03<00:00, 23.64it/s]Capturing num tokens (num_tokens=192 avail_mem=51.88 GB):  66%|██████▌   | 38/58 [00:03<00:00, 23.64it/s]Capturing num tokens (num_tokens=192 avail_mem=51.88 GB):  71%|███████   | 41/58 [00:03<00:00, 23.83it/s]Capturing num tokens (num_tokens=176 avail_mem=51.87 GB):  71%|███████   | 41/58 [00:03<00:00, 23.83it/s]

    Capturing num tokens (num_tokens=160 avail_mem=51.87 GB):  71%|███████   | 41/58 [00:03<00:00, 23.83it/s]Capturing num tokens (num_tokens=144 avail_mem=51.86 GB):  71%|███████   | 41/58 [00:03<00:00, 23.83it/s]Capturing num tokens (num_tokens=144 avail_mem=51.86 GB):  76%|███████▌  | 44/58 [00:03<00:00, 22.98it/s]Capturing num tokens (num_tokens=128 avail_mem=51.87 GB):  76%|███████▌  | 44/58 [00:03<00:00, 22.98it/s]Capturing num tokens (num_tokens=112 avail_mem=51.86 GB):  76%|███████▌  | 44/58 [00:03<00:00, 22.98it/s]

    Capturing num tokens (num_tokens=96 avail_mem=51.86 GB):  76%|███████▌  | 44/58 [00:03<00:00, 22.98it/s] Capturing num tokens (num_tokens=96 avail_mem=51.86 GB):  81%|████████  | 47/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=80 avail_mem=51.85 GB):  81%|████████  | 47/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=64 avail_mem=51.85 GB):  81%|████████  | 47/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=48 avail_mem=51.85 GB):  81%|████████  | 47/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=48 avail_mem=51.85 GB):  86%|████████▌ | 50/58 [00:04<00:00, 22.19it/s]Capturing num tokens (num_tokens=32 avail_mem=51.84 GB):  86%|████████▌ | 50/58 [00:04<00:00, 22.19it/s]

    Capturing num tokens (num_tokens=28 avail_mem=51.84 GB):  86%|████████▌ | 50/58 [00:04<00:00, 22.19it/s]Capturing num tokens (num_tokens=24 avail_mem=51.84 GB):  86%|████████▌ | 50/58 [00:04<00:00, 22.19it/s]Capturing num tokens (num_tokens=24 avail_mem=51.84 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.98it/s]Capturing num tokens (num_tokens=20 avail_mem=51.84 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.98it/s]Capturing num tokens (num_tokens=16 avail_mem=51.83 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.98it/s]Capturing num tokens (num_tokens=12 avail_mem=51.83 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.98it/s]

    Capturing num tokens (num_tokens=12 avail_mem=51.83 GB):  97%|█████████▋| 56/58 [00:04<00:00, 22.86it/s]Capturing num tokens (num_tokens=8 avail_mem=51.82 GB):  97%|█████████▋| 56/58 [00:04<00:00, 22.86it/s] Capturing num tokens (num_tokens=4 avail_mem=51.82 GB):  97%|█████████▋| 56/58 [00:04<00:00, 22.86it/s]Capturing num tokens (num_tokens=4 avail_mem=51.82 GB): 100%|██████████| 58/58 [00:04<00:00, 12.79it/s]


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
    
    Generated text: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, the capital of France is Paris. That's straightforward. Now, the user wants this information in JSON. JSON typically uses key-value pairs, so I'll need to structure it accordingly.
    
    I should figure out what specific information they're looking for. They mentioned population, so I'll definitely include that. But maybe they need more details about Paris. Let me think about what else is important. Population is a number, so I'll need to check the current estimate. I recall that Paris has a large population, but it's growing. I'll have to make sure the number is up to date.
    
    Wait, do I know the exact population? I think it's around 2.2 million, but I'm not 100% sure. Maybe I should double-check that. Also, besides the population, perhaps they want some key facts about the city. Like, what's the significance of the Eiffel Tower? Or maybe some landmarks. Including famous attractions could be useful.
    
    I should organize the JSON with clear keys. Population is a number, so that's straightforward. Then, maybe a "city_info" key that includes the population, the Eiffel Tower info, and some other notable points. That way, the JSON is comprehensive yet organized.
    
    Wait, should I include the exact population as of a specific year? I'm not sure, but perhaps just stating it as an estimate is fine. Also, I need to make sure the JSON syntax is correct—commas in the right places, proper braces. I'll format it neatly so it's readable.
    
    Putting it all together, the JSON will have a "capital" key with the name "Paris," and a "city_info" key that contains a "population" number and a "famous Landmarks" object with Eiffel Tower, the Arc de Triomphe, and the Louvre Museum. That should cover the essentials.
    
    I should also consider if the user might need more details. Maybe they're doing a project or a presentation, so providing clear and concise information is key. I think this structure should meet their needs effectively.
    </think>
    
    Here is the information and population of the capital of France (Paris) in JSON format:
    
    ```json
    {
      "capital": "Paris",
      "city_info": {
        "population": 2181000,
        "famous_landmarks": {
          "eiffel_tower": "The Eiffel Tower is a symbol of Paris and France, named after the French military engineer Gustave Eiffel.",
          "arc_de_triomphe": "The Arc de Triomphe is a French historical monument dedicated to the French Revolution.",
          "louvre_museum": "The Louvre is one of the largest and most famous museums in the world."
        }
      }
    }
    ```



```python
llm.shutdown()
```
