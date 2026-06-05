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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.52s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.48s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.49s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:20,  5.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:20,  5.62s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:25,  2.59s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:25,  2.59s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:27,  1.59s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:27,  1.59s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<01:00,  1.11s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<01:00,  1.11s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:44,  1.20it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:44,  1.20it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:34,  1.51it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:34,  1.51it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:26,  1.90it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:26,  1.90it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:21,  2.29it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:21,  2.29it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:18,  2.71it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:18,  2.71it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:14,  3.29it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:14,  3.29it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:12,  3.63it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:12,  3.63it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:10,  4.35it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:10,  4.35it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:09,  4.68it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:09,  4.68it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:07,  5.56it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:07,  5.56it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:09<00:07,  5.56it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:06,  6.68it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:06,  6.68it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:09<00:06,  6.68it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:04,  8.35it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:04,  8.35it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:09<00:04,  8.66it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:09<00:04,  8.66it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:09<00:04,  8.66it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:09<00:03, 10.68it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:09<00:03, 10.68it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:09<00:03, 10.68it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:09<00:02, 12.40it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:09<00:02, 12.40it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:09<00:02, 12.40it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:09<00:02, 12.40it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:09<00:02, 16.00it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:09<00:02, 16.00it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:09<00:02, 16.00it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:09<00:02, 16.00it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:09<00:01, 18.84it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:09<00:01, 18.84it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:09<00:01, 18.84it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:10<00:01, 18.84it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:10<00:01, 20.95it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:10<00:01, 20.95it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:10<00:01, 20.95it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:10<00:01, 20.95it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:10<00:01, 20.95it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:10<00:00, 25.17it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:10<00:00, 25.17it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:10<00:00, 25.17it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:10<00:00, 25.17it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:10<00:00, 25.17it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:10<00:00, 25.17it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:10<00:00, 30.92it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:10<00:00, 30.92it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:10<00:00, 30.92it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:10<00:00, 30.92it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:10<00:00, 30.92it/s]

    Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:10<00:00, 30.92it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:10<00:00, 34.69it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:10<00:00, 34.69it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:10<00:00, 34.69it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:10<00:00, 34.69it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:10<00:00, 34.69it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:10<00:00, 34.69it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:10<00:00, 37.54it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:10<00:00, 37.54it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:10<00:00, 37.54it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:10<00:00, 37.54it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:10<00:00, 37.54it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:10<00:00, 37.54it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:10<00:00, 40.35it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:10<00:00, 40.35it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:10<00:00, 40.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=42.13 GB):   2%|▏         | 1/58 [00:00<00:22,  2.57it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.08 GB):   2%|▏         | 1/58 [00:00<00:22,  2.57it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.08 GB):   3%|▎         | 2/58 [00:00<00:20,  2.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.06 GB):   3%|▎         | 2/58 [00:00<00:20,  2.75it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.06 GB):   5%|▌         | 3/58 [00:01<00:18,  2.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.05 GB):   5%|▌         | 3/58 [00:01<00:18,  2.95it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.05 GB):   7%|▋         | 4/58 [00:01<00:16,  3.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.02 GB):   7%|▋         | 4/58 [00:01<00:16,  3.21it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.02 GB):   9%|▊         | 5/58 [00:01<00:15,  3.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.03 GB):   9%|▊         | 5/58 [00:01<00:15,  3.41it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=42.03 GB):  10%|█         | 6/58 [00:01<00:14,  3.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.02 GB):  10%|█         | 6/58 [00:01<00:14,  3.69it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.02 GB):  12%|█▏        | 7/58 [00:02<00:12,  3.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.01 GB):  12%|█▏        | 7/58 [00:02<00:12,  3.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.01 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.00 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.44it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.00 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.00 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.00 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.00 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.00 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.00 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.00 GB):  21%|██        | 12/58 [00:02<00:06,  7.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.99 GB):  21%|██        | 12/58 [00:02<00:06,  7.04it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=41.99 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.99 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.99 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.99 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.99 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.03it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=41.99 GB):  26%|██▌       | 15/58 [00:03<00:04,  9.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.99 GB):  29%|██▉       | 17/58 [00:03<00:03, 10.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.99 GB):  29%|██▉       | 17/58 [00:03<00:03, 10.47it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.98 GB):  29%|██▉       | 17/58 [00:03<00:03, 10.47it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.98 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.98 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.06it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=41.97 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.06it/s]Capturing num tokens (num_tokens=960 avail_mem=41.97 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.06it/s] Capturing num tokens (num_tokens=960 avail_mem=41.97 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.97it/s]Capturing num tokens (num_tokens=896 avail_mem=41.96 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.97it/s]Capturing num tokens (num_tokens=832 avail_mem=41.96 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.97it/s]Capturing num tokens (num_tokens=768 avail_mem=41.95 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.97it/s]

    Capturing num tokens (num_tokens=768 avail_mem=41.95 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.52it/s]Capturing num tokens (num_tokens=704 avail_mem=41.95 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.52it/s]Capturing num tokens (num_tokens=640 avail_mem=41.95 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.52it/s]Capturing num tokens (num_tokens=576 avail_mem=41.94 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.52it/s]Capturing num tokens (num_tokens=576 avail_mem=41.94 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.60it/s]Capturing num tokens (num_tokens=512 avail_mem=41.94 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.60it/s]Capturing num tokens (num_tokens=480 avail_mem=41.94 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.60it/s]Capturing num tokens (num_tokens=448 avail_mem=41.93 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.60it/s]

    Capturing num tokens (num_tokens=448 avail_mem=41.93 GB):  53%|█████▎    | 31/58 [00:03<00:01, 21.87it/s]Capturing num tokens (num_tokens=416 avail_mem=41.93 GB):  53%|█████▎    | 31/58 [00:03<00:01, 21.87it/s]Capturing num tokens (num_tokens=384 avail_mem=41.93 GB):  53%|█████▎    | 31/58 [00:03<00:01, 21.87it/s]Capturing num tokens (num_tokens=352 avail_mem=41.92 GB):  53%|█████▎    | 31/58 [00:03<00:01, 21.87it/s]Capturing num tokens (num_tokens=320 avail_mem=41.92 GB):  53%|█████▎    | 31/58 [00:03<00:01, 21.87it/s]Capturing num tokens (num_tokens=320 avail_mem=41.92 GB):  60%|██████    | 35/58 [00:03<00:00, 25.45it/s]Capturing num tokens (num_tokens=288 avail_mem=41.92 GB):  60%|██████    | 35/58 [00:03<00:00, 25.45it/s]Capturing num tokens (num_tokens=256 avail_mem=41.92 GB):  60%|██████    | 35/58 [00:03<00:00, 25.45it/s]Capturing num tokens (num_tokens=240 avail_mem=41.91 GB):  60%|██████    | 35/58 [00:03<00:00, 25.45it/s]Capturing num tokens (num_tokens=224 avail_mem=41.91 GB):  60%|██████    | 35/58 [00:03<00:00, 25.45it/s]

    Capturing num tokens (num_tokens=224 avail_mem=41.91 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.55it/s]Capturing num tokens (num_tokens=208 avail_mem=41.91 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.55it/s]Capturing num tokens (num_tokens=192 avail_mem=41.90 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.55it/s]Capturing num tokens (num_tokens=176 avail_mem=41.90 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.55it/s]Capturing num tokens (num_tokens=160 avail_mem=41.90 GB):  67%|██████▋   | 39/58 [00:04<00:00, 28.55it/s]Capturing num tokens (num_tokens=160 avail_mem=41.90 GB):  74%|███████▍  | 43/58 [00:04<00:00, 30.03it/s]Capturing num tokens (num_tokens=144 avail_mem=41.89 GB):  74%|███████▍  | 43/58 [00:04<00:00, 30.03it/s]Capturing num tokens (num_tokens=128 avail_mem=41.89 GB):  74%|███████▍  | 43/58 [00:04<00:00, 30.03it/s]Capturing num tokens (num_tokens=112 avail_mem=41.89 GB):  74%|███████▍  | 43/58 [00:04<00:00, 30.03it/s]

    Capturing num tokens (num_tokens=96 avail_mem=41.89 GB):  74%|███████▍  | 43/58 [00:04<00:00, 30.03it/s] Capturing num tokens (num_tokens=96 avail_mem=41.89 GB):  81%|████████  | 47/58 [00:04<00:00, 30.56it/s]Capturing num tokens (num_tokens=80 avail_mem=41.88 GB):  81%|████████  | 47/58 [00:04<00:00, 30.56it/s]Capturing num tokens (num_tokens=64 avail_mem=41.88 GB):  81%|████████  | 47/58 [00:04<00:00, 30.56it/s]Capturing num tokens (num_tokens=48 avail_mem=41.87 GB):  81%|████████  | 47/58 [00:04<00:00, 30.56it/s]Capturing num tokens (num_tokens=32 avail_mem=41.87 GB):  81%|████████  | 47/58 [00:04<00:00, 30.56it/s]Capturing num tokens (num_tokens=32 avail_mem=41.87 GB):  88%|████████▊ | 51/58 [00:04<00:00, 32.40it/s]Capturing num tokens (num_tokens=28 avail_mem=41.87 GB):  88%|████████▊ | 51/58 [00:04<00:00, 32.40it/s]Capturing num tokens (num_tokens=24 avail_mem=41.87 GB):  88%|████████▊ | 51/58 [00:04<00:00, 32.40it/s]Capturing num tokens (num_tokens=20 avail_mem=41.86 GB):  88%|████████▊ | 51/58 [00:04<00:00, 32.40it/s]

    Capturing num tokens (num_tokens=16 avail_mem=41.86 GB):  88%|████████▊ | 51/58 [00:04<00:00, 32.40it/s]Capturing num tokens (num_tokens=16 avail_mem=41.86 GB):  95%|█████████▍| 55/58 [00:04<00:00, 33.81it/s]Capturing num tokens (num_tokens=12 avail_mem=41.85 GB):  95%|█████████▍| 55/58 [00:04<00:00, 33.81it/s]Capturing num tokens (num_tokens=8 avail_mem=41.85 GB):  95%|█████████▍| 55/58 [00:04<00:00, 33.81it/s] Capturing num tokens (num_tokens=4 avail_mem=41.85 GB):  95%|█████████▍| 55/58 [00:04<00:00, 33.81it/s]Capturing num tokens (num_tokens=4 avail_mem=41.85 GB): 100%|██████████| 58/58 [00:04<00:00, 12.97it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, the user is asking for the current date and time in New York and the weather there. I need to figure out how to get this information using the functions provided. <br><br>First, I should use the get_current_date function. The function requires a timezone parameter. Since the user is in New York, I can use 'America/New_York' as the timezone. I'll structure this as a JSON object with timezone as the key and the value as 'America/New_York'.<br><br>Next, I need to get the current weather. For that, I'll use the get_current_weather function. The city is New York, the state is NY, and I should get the temperature in Fahrenheit since that's a common unit people use. So the parameters will include city: 'New York', state: 'NY', unit: 'fahrenheit'.<br><br>I should make sure to call each function separately, as per the instructions, and present the responses clearly. I'll format each function call correctly, using the specified tags and including all required parameters. Also, I'll remember to add the sources where I got the function information from.<br><br><br>content: <br><br><function=get_current_date>{  <br>  "timezone": "America/New_York"  <br>}</function>  <br><function=get_current_weather>{  <br>  "city": "New York",  <br>  "state": "NY",  <br>  "unit": "fahrenheit"  <br>}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '6529c10c53b4458f99d0c52fe99ef07d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.256855854764581, 'response_sent_to_client_ts': 1780622820.8571544}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': 'cc3a294be1564de89b3ccd16ecd7aceb', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 21.2171015040949, 'response_sent_to_client_ts': 1780622842.0856395}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '1224fb019a28482e9303bc39063895c7', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.23762556351721287, 'response_sent_to_client_ts': 1780622842.3670845}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'a5e08aa0d1d14e6f8ab0c2e29aba3dca', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.23726645484566689, 'response_sent_to_client_ts': 1780622842.3670998}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '0e37eef076f94b9188552cff2c792fad', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.23720300290733576, 'response_sent_to_client_ts': 1780622842.3671052}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'b0f40addaeb24f2dacbb225ada4c882d', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 22.366031911224127, 'response_sent_to_client_ts': 1780622864.7465777}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the population and the information about the capital of France in JSON format. Hmm, I need to figure out exactly what they need. Maybe they\'re doing a project or research on France, specifically the capital city, Paris. They might be creating a data structure for an app or a website. \n\nFirst, I should confirm the basic facts. The capital of France is definitely Paris. Its population has been changing over the years, so I should look up the most recent data. I remember that in 2020, Paris had around 3.5 million people. That\'s a city that often grows and shrinks, so I\'ll need to make sure the figure is up to date, maybe from 2023 just in case.\n\nNext, they asked for information about the capital. So, I should include more than just the population. I can think of cultural highlights, major landmarks like the Eiffel Tower, the Île de la Cité with兵马俑, and some significant museums. Also, important institutions such as the Louvre and CNRS, along with some famous people born there, like.Titley-Ajello. \n\nPublic transportation and infrastructure are important too. The地铁 network in Paris is one of the most extensive in the world, so that\'s worth mentioning. The Île St-Lazare gardens are a well-knownette, so I\'ll add that. Then, other important areas like Latin Quarter and Yaounde could be places of interest. Maybe include some notable landmarks like the Notre-Dame and Sacré-Cœur basilica.\n\nI should make sure all this information is accurate and concise. Since the user wants it in JSON format, I need to structure it properly with appropriate keys. Let me list out all the points I\'ve thought of and assign them to the correct keys. \n\nWait, should I include the population growth trend? Maybe not, unless the user specifically asks. They just asked for the population, so I\'ll stick to providing the current figure. \n\nI should double-check the population numbers to ensure they\'re accurate for 2023. If I\'m unsure, perhaps I should provide a note saying the figure is approximate in 2023. That way, I\'m being cautious and not giving incorrect information. \n\nPutting it all together, the JSON should have the "capital" key with Paris, and an "information" array containing various points about Paris, including population, landmarks, institutions, etc. I need to make sure each information item is a string and that the JSON syntax is correct, with commas and brackets properly placed.\n\nSo, the final JSON will have two keys: "capital" which is a string, and "information" which is an array of strings with all the relevant details. I\'ll structure it carefully to avoid any syntax errors. \n\nI think that covers everything the user asked for. They might be looking for clear, structured data, so providing it in JSON format will likely be very helpful for whatever project they\'re working on.\n</think>\n\n```json\n{\n  "capital": "Paris",\n  "information": [\n    {\n      "key": "population",\n      "value": "As of 2023, the population of Paris is approximately 3,600,000 people."\n    },\n    {\n      "key": "cultural_highlights",\n      "value": "Paris is the capital city of France and one of the most important cultural and economic centers in the world."\n    },\n    {\n      "key": "landmarks",\n      "value": "Some of the most famous landmarks in Paris include the Eiffel Tower, the Île de la Cité (with its兵马俑 replicas), and the Louvre Museum."\n    },\n    {\n      "key": "institutions",\n      "value": "Paris is home to several world-class institutions, including the French National Center for Scientific Research (CNRS), the Louvre, and the grandes écoles (e.g., L\'École normale supérieure)."\n    },\n    {\n      "key": "famous_people",\n      "value": "Paris is the birthplace of many famous people, including-Michel Titley-Ajello, Jean Clairambault, and Claireтеlor."\n    },\n    {\n      "key": "public_transportation",\n      "value": "The Paris地铁 network is one of the most extensive in the world, with an extensive trunk network covering much of the city."\n    },\n    {\n      "key": "gardens",\n      "value": "The Île St-Lazare gardens, located in Paris, are one of the most famous gardens in France."\n    },\n    {\n      "key": "notable_area",\n      "value": "The Latin Quarter is one of Paris\'s most famous and historic student neighborhoods, located in the Île Saint-Lazare."\n    },\n    {\n      "key": "landmark",\n      "value": "The Notre-Dame Cathedral and the Sacré-Cœur Basilica are iconic landmarks in Paris."\n    }\n  ]\n}\n```', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 7042, 323, 279, 1995, 911, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 88190, 11, 358, 1184, 311, 7071, 700, 6896, 1128, 807, 1184, 13, 10696, 807, 2299, 3730, 264, 2390, 476, 3412, 389, 9625, 11, 11689, 279, 6722, 3283, 11, 12095, 13, 2379, 2578, 387, 6825, 264, 821, 5944, 369, 458, 906, 476, 264, 3910, 13, 4710, 5338, 11, 358, 1265, 7683, 279, 6770, 13064, 13, 576, 6722, 315, 9625, 374, 8491, 12095, 13, 11445, 7042, 702, 1012, 10018, 916, 279, 1635, 11, 773, 358, 1265, 1401, 705, 279, 1429, 3213, 821, 13, 358, 6099, 429, 304, 220, 17, 15, 17, 15, 11, 12095, 1030, 2163, 220, 18, 13, 20, 3526, 1251, 13, 2938, 594, 264, 3283, 429, 3545, 27715, 323, 14035, 15504, 11, 773, 358, 3278, 1184, 311, 1281, 2704, 279, 7071, 374, 705, 311, 2400, 11, 7196, 504, 220, 17, 15, 17, 18, 1101, 304, 1142, 382, 5847, 11, 807, 4588, 369, 1995, 911, 279, 6722, 13, 2055, 11, 358, 1265, 2924, 803, 1091, 1101, 279, 7042, 13, 358, 646, 1744, 315, 12752, 21314, 11, 3598, 59924, 1075, 279, 468, 3092, 301, 21938, 11, 279, 59108, 273, 409, 1187, 356, 12815, 448, 116396, 119146, 11, 323, 1045, 5089, 50577, 13, 7281, 11, 2989, 14336, 1741, 438, 279, 9729, 48506, 323, 24872, 11451, 11, 3156, 448, 1045, 11245, 1251, 9223, 1052, 11, 1075, 22967, 88, 6691, 73, 4791, 13, 4710, 12676, 17903, 323, 13737, 525, 2989, 2238, 13, 576, 104851, 3922, 304, 12095, 374, 825, 315, 279, 1429, 16376, 304, 279, 1879, 11, 773, 429, 594, 5802, 44291, 13, 576, 59108, 273, 794, 8125, 1370, 546, 35436, 525, 264, 1632, 21309, 6535, 11, 773, 358, 3278, 912, 429, 13, 5005, 11, 1008, 2989, 5671, 1075, 19458, 34394, 323, 39384, 795, 68, 1410, 387, 7482, 315, 2734, 13, 10696, 2924, 1045, 27190, 59924, 1075, 279, 43464, 9420, 373, 323, 23047, 42611, 7658, 88312, 67086, 3001, 382, 40, 1265, 1281, 2704, 678, 419, 1995, 374, 13382, 323, 63594, 13, 8704, 279, 1196, 6801, 432, 304, 4718, 3561, 11, 358, 1184, 311, 5944, 432, 10277, 448, 8311, 6894, 13, 6771, 752, 1140, 700, 678, 279, 3501, 358, 3003, 3381, 315, 323, 9793, 1105, 311, 279, 4396, 6894, 13, 4710, 14190, 11, 1265, 358, 2924, 279, 7042, 6513, 9149, 30, 10696, 537, 11, 7241, 279, 1196, 11689, 17064, 13, 2379, 1101, 4588, 369, 279, 7042, 11, 773, 358, 3278, 9214, 311, 8241, 279, 1482, 7071, 13, 4710, 40, 1265, 1990, 15934, 279, 7042, 5109, 311, 5978, 807, 2299, 13382, 369, 220, 17, 15, 17, 18, 13, 1416, 358, 2776, 42903, 11, 8365, 358, 1265, 3410, 264, 5185, 5488, 279, 7071, 374, 44868, 304, 220, 17, 15, 17, 18, 13, 2938, 1616, 11, 358, 2776, 1660, 45778, 323, 537, 7086, 15114, 1995, 13, 4710, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 614, 279, 330, 65063, 1, 1376, 448, 12095, 11, 323, 458, 330, 25069, 1, 1334, 8482, 5257, 3501, 911, 12095, 11, 2670, 7042, 11, 59924, 11, 14336, 11, 4992, 13, 358, 1184, 311, 1281, 2704, 1817, 1995, 1509, 374, 264, 914, 323, 429, 279, 4718, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 10277, 9099, 382, 4416, 11, 279, 1590, 4718, 686, 614, 1378, 6894, 25, 330, 65063, 1, 892, 374, 264, 914, 11, 323, 330, 25069, 1, 892, 374, 458, 1334, 315, 9069, 448, 678, 279, 9760, 3565, 13, 358, 3278, 5944, 432, 15516, 311, 5648, 894, 19482, 5975, 13, 4710, 40, 1744, 429, 14521, 4297, 279, 1196, 4588, 369, 13, 2379, 2578, 387, 3330, 369, 2797, 11, 32930, 821, 11, 773, 8241, 432, 304, 4718, 3561, 686, 4363, 387, 1602, 10950, 369, 8820, 2390, 807, 2299, 3238, 389, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 25069, 788, 2278, 262, 341, 414, 330, 792, 788, 330, 44441, 756, 414, 330, 957, 788, 330, 2121, 315, 220, 17, 15, 17, 18, 11, 279, 7042, 315, 12095, 374, 13187, 220, 18, 11, 21, 15, 15, 11, 15, 15, 15, 1251, 10040, 262, 1153, 262, 341, 414, 330, 792, 788, 330, 66, 43447, 22680, 13826, 756, 414, 330, 957, 788, 330, 59604, 374, 279, 6722, 3283, 315, 9625, 323, 825, 315, 279, 1429, 2989, 12752, 323, 6955, 18652, 304, 279, 1879, 10040, 262, 1153, 262, 341, 414, 330, 792, 788, 330, 1933, 15544, 756, 414, 330, 957, 788, 330, 8373, 315, 279, 1429, 11245, 59924, 304, 12095, 2924, 279, 468, 3092, 301, 21938, 11, 279, 59108, 273, 409, 1187, 356, 12815, 320, 4197, 1181, 116396, 119146, 80801, 701, 323, 279, 9729, 48506, 16328, 10040, 262, 1153, 262, 341, 414, 330, 792, 788, 330, 258, 92678, 756, 414, 330, 957, 788, 330, 59604, 374, 2114, 311, 3807, 1879, 14800, 14336, 11, 2670, 279, 8585, 5055, 5832, 369, 37030, 8319, 320, 28668, 11451, 701, 279, 9729, 48506, 11, 323, 279, 37446, 3958, 84838, 320, 68, 1302, 2572, 444, 6, 26789, 55645, 7465, 1574, 1026, 34185, 552, 568, 698, 262, 1153, 262, 341, 414, 330, 792, 788, 330, 69, 22517, 62717, 756, 414, 330, 957, 788, 330, 59604, 374, 279, 7194, 2007, 315, 1657, 11245, 1251, 11, 2670, 5251, 90240, 10869, 88, 6691, 73, 4791, 11, 19685, 92510, 2969, 945, 11, 323, 42575, 50527, 9566, 10040, 262, 1153, 262, 341, 414, 330, 792, 788, 330, 888, 46398, 367, 756, 414, 330, 957, 788, 330, 785, 12095, 104851, 3922, 374, 825, 315, 279, 1429, 16376, 304, 279, 1879, 11, 448, 458, 16376, 37311, 3922, 18202, 1753, 315, 279, 3283, 10040, 262, 1153, 262, 341, 414, 330, 792, 788, 330, 70, 567, 724, 756, 414, 330, 957, 788, 330, 785, 59108, 273, 794, 8125, 1370, 546, 35436, 11, 7407, 304, 12095, 11, 525, 825, 315, 279, 1429, 11245, 35436, 304, 9625, 10040, 262, 1153, 262, 341, 414, 330, 792, 788, 330, 1921, 480, 15030, 756, 414, 330, 957, 788, 330, 785, 19458, 34394, 374, 825, 315, 12095, 594, 1429, 11245, 323, 18035, 5458, 32000, 11, 7407, 304, 279, 59108, 273, 14205, 8125, 1370, 546, 10040, 262, 1153, 262, 341, 414, 330, 792, 788, 330, 1933, 3987, 756, 414, 330, 957, 788, 330, 785, 2806, 265, 9420, 373, 56729, 323, 279, 23047, 42611, 7658, 88312, 65970, 3001, 525, 26277, 59924, 304, 12095, 10040, 262, 456, 220, 5133, 532, 73594, 151643], 'meta_info': {'id': 'fd0a83ff325b46d7a4af67b285da0bd5', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 620, 'completion_tokens': 1049, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 9.888941031880677, 'response_sent_to_client_ts': 1780622874.6439474}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.62s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.53s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.54s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:02,  5.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:02,  5.31s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:09,  2.32s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:09,  2.32s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:14,  1.35s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:14,  1.35s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:48,  1.12it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:48,  1.12it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:33,  1.59it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:24,  2.13it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:24,  2.13it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.78it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.78it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.49it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.49it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.32it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.32it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:09,  5.22it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:09,  5.22it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:06<00:09,  5.22it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:06,  6.95it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:06,  6.95it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:06<00:06,  6.95it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:05,  8.54it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:05,  8.54it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:07<00:05,  8.54it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:04, 10.31it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:04, 10.31it/s]

    Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:07<00:04, 10.31it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:07<00:03, 12.33it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:07<00:03, 12.33it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:07<00:03, 12.33it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:07<00:03, 12.33it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:07<00:02, 15.80it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:07<00:02, 15.80it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:07<00:02, 15.80it/s]

    Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:07<00:02, 15.80it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:07<00:02, 15.80it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:07<00:01, 21.34it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:07<00:01, 21.34it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:07<00:01, 21.34it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:07<00:01, 21.34it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:07<00:01, 21.34it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:07<00:01, 21.34it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:07<00:01, 21.34it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:07<00:00, 30.72it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:07<00:00, 30.72it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:07<00:00, 30.72it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:07<00:00, 30.72it/s]

    Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:07<00:00, 30.72it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:07<00:00, 30.72it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:07<00:00, 30.72it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:07<00:00, 37.70it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:07<00:00, 37.70it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:07<00:00, 37.70it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:07<00:00, 37.70it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:07<00:00, 37.70it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:07<00:00, 37.70it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:07<00:00, 37.70it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:07<00:00, 37.70it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:07<00:00, 37.70it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:07<00:00, 47.94it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:07<00:00, 47.94it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:07<00:00, 47.94it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:07<00:00, 47.94it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:07<00:00, 47.94it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:07<00:00, 47.94it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:07<00:00, 47.94it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:07<00:00, 47.94it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:07<00:00, 47.94it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:07<00:00, 54.61it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:07<00:00, 54.61it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:07<00:00, 54.61it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:07<00:00, 54.61it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:07<00:00, 54.61it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:07<00:00, 54.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=36.01 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=36.01 GB):   2%|▏         | 1/58 [00:00<00:17,  3.21it/s]Capturing num tokens (num_tokens=7680 avail_mem=29.57 GB):   2%|▏         | 1/58 [00:00<00:17,  3.21it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=29.57 GB):   3%|▎         | 2/58 [00:00<00:16,  3.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=29.56 GB):   3%|▎         | 2/58 [00:00<00:16,  3.45it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=29.56 GB):   5%|▌         | 3/58 [00:00<00:14,  3.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=29.56 GB):   5%|▌         | 3/58 [00:00<00:14,  3.68it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=29.56 GB):   7%|▋         | 4/58 [00:01<00:13,  3.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=29.56 GB):   7%|▋         | 4/58 [00:01<00:13,  3.98it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=29.56 GB):   9%|▊         | 5/58 [00:01<00:12,  4.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=29.56 GB):   9%|▊         | 5/58 [00:01<00:12,  4.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=29.56 GB):  10%|█         | 6/58 [00:01<00:11,  4.57it/s]Capturing num tokens (num_tokens=5120 avail_mem=29.56 GB):  10%|█         | 6/58 [00:01<00:11,  4.57it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=29.56 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.95it/s]Capturing num tokens (num_tokens=4608 avail_mem=29.56 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.95it/s]Capturing num tokens (num_tokens=4608 avail_mem=29.56 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=29.56 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.45it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=29.56 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=29.56 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=29.56 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=29.56 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.36it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=29.56 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.91it/s]Capturing num tokens (num_tokens=3328 avail_mem=29.56 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.91it/s]Capturing num tokens (num_tokens=3328 avail_mem=29.56 GB):  21%|██        | 12/58 [00:02<00:06,  7.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=29.55 GB):  21%|██        | 12/58 [00:02<00:06,  7.51it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=29.55 GB):  21%|██        | 12/58 [00:02<00:06,  7.51it/s]Capturing num tokens (num_tokens=2816 avail_mem=29.55 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=29.55 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=29.55 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.70it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=29.55 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.06it/s]Capturing num tokens (num_tokens=2048 avail_mem=29.55 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=29.55 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=29.55 GB):  31%|███       | 18/58 [00:02<00:03, 11.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=29.54 GB):  31%|███       | 18/58 [00:02<00:03, 11.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=29.54 GB):  31%|███       | 18/58 [00:02<00:03, 11.61it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=29.53 GB):  31%|███       | 18/58 [00:02<00:03, 11.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=29.53 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.50it/s]Capturing num tokens (num_tokens=960 avail_mem=29.52 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.50it/s] Capturing num tokens (num_tokens=896 avail_mem=29.52 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.50it/s]Capturing num tokens (num_tokens=832 avail_mem=29.52 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.50it/s]Capturing num tokens (num_tokens=832 avail_mem=29.52 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.44it/s]Capturing num tokens (num_tokens=768 avail_mem=29.51 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.44it/s]Capturing num tokens (num_tokens=704 avail_mem=29.51 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.44it/s]

    Capturing num tokens (num_tokens=640 avail_mem=29.51 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.44it/s]Capturing num tokens (num_tokens=640 avail_mem=29.51 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.30it/s]Capturing num tokens (num_tokens=576 avail_mem=29.50 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.30it/s]Capturing num tokens (num_tokens=512 avail_mem=29.50 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.30it/s]Capturing num tokens (num_tokens=480 avail_mem=29.49 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.30it/s]Capturing num tokens (num_tokens=448 avail_mem=29.49 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.30it/s]Capturing num tokens (num_tokens=448 avail_mem=29.49 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.56it/s]Capturing num tokens (num_tokens=416 avail_mem=29.49 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.56it/s]Capturing num tokens (num_tokens=384 avail_mem=29.48 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.56it/s]

    Capturing num tokens (num_tokens=352 avail_mem=29.48 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.56it/s]Capturing num tokens (num_tokens=320 avail_mem=29.48 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.56it/s]Capturing num tokens (num_tokens=320 avail_mem=29.48 GB):  60%|██████    | 35/58 [00:03<00:00, 26.81it/s]Capturing num tokens (num_tokens=288 avail_mem=29.48 GB):  60%|██████    | 35/58 [00:03<00:00, 26.81it/s]Capturing num tokens (num_tokens=256 avail_mem=29.48 GB):  60%|██████    | 35/58 [00:03<00:00, 26.81it/s]Capturing num tokens (num_tokens=240 avail_mem=29.47 GB):  60%|██████    | 35/58 [00:03<00:00, 26.81it/s]Capturing num tokens (num_tokens=224 avail_mem=29.47 GB):  60%|██████    | 35/58 [00:03<00:00, 26.81it/s]Capturing num tokens (num_tokens=224 avail_mem=29.47 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.60it/s]Capturing num tokens (num_tokens=208 avail_mem=29.46 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.60it/s]Capturing num tokens (num_tokens=192 avail_mem=29.46 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.60it/s]

    Capturing num tokens (num_tokens=176 avail_mem=29.46 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.60it/s]Capturing num tokens (num_tokens=160 avail_mem=29.46 GB):  67%|██████▋   | 39/58 [00:03<00:00, 29.60it/s]Capturing num tokens (num_tokens=160 avail_mem=29.46 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.74it/s]Capturing num tokens (num_tokens=144 avail_mem=29.45 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.74it/s]Capturing num tokens (num_tokens=128 avail_mem=28.35 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.74it/s]

    Capturing num tokens (num_tokens=112 avail_mem=28.35 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.74it/s]Capturing num tokens (num_tokens=96 avail_mem=28.34 GB):  74%|███████▍  | 43/58 [00:03<00:00, 31.74it/s] Capturing num tokens (num_tokens=96 avail_mem=28.34 GB):  81%|████████  | 47/58 [00:03<00:00, 21.74it/s]Capturing num tokens (num_tokens=80 avail_mem=28.34 GB):  81%|████████  | 47/58 [00:03<00:00, 21.74it/s]Capturing num tokens (num_tokens=64 avail_mem=28.33 GB):  81%|████████  | 47/58 [00:03<00:00, 21.74it/s]Capturing num tokens (num_tokens=48 avail_mem=28.33 GB):  81%|████████  | 47/58 [00:03<00:00, 21.74it/s]

    Capturing num tokens (num_tokens=32 avail_mem=28.33 GB):  81%|████████  | 47/58 [00:03<00:00, 21.74it/s]Capturing num tokens (num_tokens=32 avail_mem=28.33 GB):  88%|████████▊ | 51/58 [00:03<00:00, 25.24it/s]Capturing num tokens (num_tokens=28 avail_mem=28.33 GB):  88%|████████▊ | 51/58 [00:03<00:00, 25.24it/s]Capturing num tokens (num_tokens=24 avail_mem=28.32 GB):  88%|████████▊ | 51/58 [00:03<00:00, 25.24it/s]Capturing num tokens (num_tokens=20 avail_mem=28.32 GB):  88%|████████▊ | 51/58 [00:04<00:00, 25.24it/s]Capturing num tokens (num_tokens=16 avail_mem=28.32 GB):  88%|████████▊ | 51/58 [00:04<00:00, 25.24it/s]Capturing num tokens (num_tokens=16 avail_mem=28.32 GB):  95%|█████████▍| 55/58 [00:04<00:00, 28.26it/s]Capturing num tokens (num_tokens=12 avail_mem=28.25 GB):  95%|█████████▍| 55/58 [00:04<00:00, 28.26it/s]Capturing num tokens (num_tokens=8 avail_mem=28.24 GB):  95%|█████████▍| 55/58 [00:04<00:00, 28.26it/s] Capturing num tokens (num_tokens=4 avail_mem=28.24 GB):  95%|█████████▍| 55/58 [00:04<00:00, 28.26it/s]

    Capturing num tokens (num_tokens=4 avail_mem=28.24 GB): 100%|██████████| 58/58 [00:04<00:00, 14.03it/s]


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
    
    Generated text: Okay, the user asked for the information and population of the capital of France in JSON format. First, I need to identify who the user is and what they need. They're probably looking for a structured data format, which is common for programming or data analysis purposes.
    
    I know that the capital of France is Paris. So, the key pieces of information would be the name of the city, its population, and maybe some additional relevant details like the country it's in. The population is a bit tricky because it can change every year. I should probably provide the most recent data available, which I think is from 2021. Let me double-check that number to make sure it's accurate.
    
    The population figure I remember is around 9 million, but I should confirm if that's the correct and current number. Maybe I can recall that it's approximately 8,780,000 as of 2021. It's important to be precise here to avoid giving incorrect information.
    
    So, putting it all together, the JSON structure should have a key "capital" with a "name" and "population". The population should be a number data type, and the name should be a string. I'll format it correctly, making sure commas are placed properly to avoid any syntax errors.
    
    I also need to consider if the user might be using this data for something else, like mapping or further analysis, so accuracy is crucial. Since I'm providing an example, I'll make sure to include all necessary fields without overcomplicating it.
    
    Alright, I think that's all. Time to put the JSON together.
    </think>
    
    Here is the information about the capital of France in JSON format:
    
    ```json
    {
      "capital": {
        "name": "Paris",
        "population": 9705000  // As of the latest available data (2021)
      }
    }
    ```



```python
llm.shutdown()
```
