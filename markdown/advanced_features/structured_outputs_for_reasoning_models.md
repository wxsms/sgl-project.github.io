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

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]


    2026-05-03 23:04:40,081 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-03 23:04:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:12,  2.37s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:12,  2.37s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:21,  1.48s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:21,  1.48s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:56,  1.05s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:56,  1.05s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:42,  1.24it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:42,  1.24it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:36,  1.42it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:36,  1.42it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:29,  1.74it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:29,  1.74it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:23,  2.10it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:23,  2.10it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:19,  2.51it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:19,  2.51it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:16,  2.95it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:16,  2.95it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:13,  3.38it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:13,  3.38it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:11,  3.87it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:11,  3.87it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:10,  4.36it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:10,  4.36it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:09,  4.60it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:09,  4.60it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:09,  4.66it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:09,  4.66it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:08,  4.82it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:08,  4.82it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:07,  5.19it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:07,  5.19it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:07,  5.55it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:07,  5.55it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:09<00:06,  6.00it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:09<00:06,  6.00it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:05,  6.57it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:05,  6.57it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:10<00:05,  7.18it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:10<00:05,  7.18it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:10<00:05,  7.18it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:10<00:03,  9.01it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:10<00:03,  9.01it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:10<00:03,  9.01it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:10<00:03, 10.10it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:10<00:03, 10.10it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:10<00:03, 10.10it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:10<00:02, 11.30it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:10<00:02, 11.30it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:10<00:02, 11.30it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:10<00:02, 13.17it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:10<00:02, 13.17it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:10<00:02, 13.17it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:10<00:01, 14.54it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:10<00:01, 14.54it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:10<00:01, 14.54it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:10<00:01, 14.99it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:10<00:01, 14.99it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:10<00:01, 14.99it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:10<00:01, 16.24it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:10<00:01, 16.24it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:11<00:01, 16.24it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:11<00:01, 16.24it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:11<00:01, 17.99it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:11<00:01, 17.99it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:11<00:01, 17.99it/s]

    Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:11<00:01, 17.99it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:11<00:00, 20.08it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:11<00:00, 20.08it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:11<00:00, 20.08it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:11<00:00, 20.08it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:11<00:00, 20.80it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:11<00:00, 20.80it/s]

    Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:11<00:00, 20.80it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:11<00:00, 20.80it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:11<00:00, 22.54it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:11<00:00, 22.54it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:11<00:00, 22.54it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:11<00:00, 22.54it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:11<00:00, 22.28it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:11<00:00, 22.28it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:11<00:00, 22.28it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:11<00:00, 22.28it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:11<00:00, 23.89it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:11<00:00, 23.89it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:11<00:00, 23.89it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:11<00:00, 23.89it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:11<00:00, 23.89it/s] 

    Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:11<00:00, 27.15it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:11<00:00, 27.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  4.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   2%|▏         | 1/58 [00:00<00:52,  1.09it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.74 GB):   2%|▏         | 1/58 [00:00<00:52,  1.09it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.74 GB):   3%|▎         | 2/58 [00:01<00:46,  1.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.80 GB):   3%|▎         | 2/58 [00:01<00:46,  1.21it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.80 GB):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.86 GB):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.86 GB):   7%|▋         | 4/58 [00:02<00:35,  1.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.93 GB):   7%|▋         | 4/58 [00:02<00:35,  1.51it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.93 GB):   9%|▊         | 5/58 [00:03<00:31,  1.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.99 GB):   9%|▊         | 5/58 [00:03<00:31,  1.68it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.99 GB):  10%|█         | 6/58 [00:03<00:27,  1.91it/s]Capturing num tokens (num_tokens=5120 avail_mem=26.05 GB):  10%|█         | 6/58 [00:03<00:27,  1.91it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=26.05 GB):  12%|█▏        | 7/58 [00:04<00:25,  1.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=24.99 GB):  12%|█▏        | 7/58 [00:04<00:25,  1.99it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=24.99 GB):  14%|█▍        | 8/58 [00:04<00:23,  2.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.30 GB):  14%|█▍        | 8/58 [00:04<00:23,  2.09it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.30 GB):  16%|█▌        | 9/58 [00:05<00:22,  2.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.69 GB):  16%|█▌        | 9/58 [00:05<00:22,  2.20it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=25.69 GB):  17%|█▋        | 10/58 [00:05<00:21,  2.26it/s]Capturing num tokens (num_tokens=3584 avail_mem=26.67 GB):  17%|█▋        | 10/58 [00:05<00:21,  2.26it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=26.67 GB):  19%|█▉        | 11/58 [00:05<00:19,  2.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=25.75 GB):  19%|█▉        | 11/58 [00:05<00:19,  2.36it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=25.75 GB):  21%|██        | 12/58 [00:06<00:18,  2.54it/s]Capturing num tokens (num_tokens=3072 avail_mem=25.81 GB):  21%|██        | 12/58 [00:06<00:18,  2.54it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=25.81 GB):  22%|██▏       | 13/58 [00:06<00:17,  2.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.66 GB):  22%|██▏       | 13/58 [00:06<00:17,  2.64it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.66 GB):  24%|██▍       | 14/58 [00:06<00:15,  2.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=25.88 GB):  24%|██▍       | 14/58 [00:06<00:15,  2.88it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=25.88 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.66 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.09it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=26.66 GB):  28%|██▊       | 16/58 [00:07<00:13,  3.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=25.92 GB):  28%|██▊       | 16/58 [00:07<00:13,  3.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=25.92 GB):  29%|██▉       | 17/58 [00:07<00:11,  3.58it/s]Capturing num tokens (num_tokens=1792 avail_mem=25.99 GB):  29%|██▉       | 17/58 [00:07<00:11,  3.58it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=25.99 GB):  31%|███       | 18/58 [00:07<00:10,  3.80it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.64 GB):  31%|███       | 18/58 [00:07<00:10,  3.80it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.64 GB):  33%|███▎      | 19/58 [00:07<00:09,  4.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.04 GB):  33%|███▎      | 19/58 [00:07<00:09,  4.27it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=26.04 GB):  34%|███▍      | 20/58 [00:08<00:08,  4.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.61 GB):  34%|███▍      | 20/58 [00:08<00:08,  4.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.61 GB):  36%|███▌      | 21/58 [00:08<00:06,  5.39it/s]Capturing num tokens (num_tokens=960 avail_mem=26.08 GB):  36%|███▌      | 21/58 [00:08<00:06,  5.39it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=26.08 GB):  38%|███▊      | 22/58 [00:08<00:06,  5.78it/s]Capturing num tokens (num_tokens=896 avail_mem=26.60 GB):  38%|███▊      | 22/58 [00:08<00:06,  5.78it/s]Capturing num tokens (num_tokens=896 avail_mem=26.60 GB):  40%|███▉      | 23/58 [00:08<00:05,  6.27it/s]Capturing num tokens (num_tokens=832 avail_mem=26.10 GB):  40%|███▉      | 23/58 [00:08<00:05,  6.27it/s]

    Capturing num tokens (num_tokens=832 avail_mem=26.10 GB):  41%|████▏     | 24/58 [00:08<00:05,  6.70it/s]Capturing num tokens (num_tokens=768 avail_mem=26.58 GB):  41%|████▏     | 24/58 [00:08<00:05,  6.70it/s]Capturing num tokens (num_tokens=768 avail_mem=26.58 GB):  43%|████▎     | 25/58 [00:08<00:04,  6.97it/s]Capturing num tokens (num_tokens=704 avail_mem=26.11 GB):  43%|████▎     | 25/58 [00:08<00:04,  6.97it/s]

    Capturing num tokens (num_tokens=640 avail_mem=26.25 GB):  43%|████▎     | 25/58 [00:08<00:04,  6.97it/s]Capturing num tokens (num_tokens=640 avail_mem=26.25 GB):  47%|████▋     | 27/58 [00:08<00:04,  7.74it/s]Capturing num tokens (num_tokens=576 avail_mem=26.56 GB):  47%|████▋     | 27/58 [00:08<00:04,  7.74it/s]

    Capturing num tokens (num_tokens=512 avail_mem=26.15 GB):  47%|████▋     | 27/58 [00:09<00:04,  7.74it/s]Capturing num tokens (num_tokens=512 avail_mem=26.15 GB):  50%|█████     | 29/58 [00:09<00:03,  8.55it/s]Capturing num tokens (num_tokens=480 avail_mem=26.54 GB):  50%|█████     | 29/58 [00:09<00:03,  8.55it/s]

    Capturing num tokens (num_tokens=480 avail_mem=26.54 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.81it/s]Capturing num tokens (num_tokens=448 avail_mem=26.17 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.81it/s]Capturing num tokens (num_tokens=416 avail_mem=26.19 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.81it/s]Capturing num tokens (num_tokens=416 avail_mem=26.19 GB):  55%|█████▌    | 32/58 [00:09<00:02,  9.47it/s]Capturing num tokens (num_tokens=384 avail_mem=26.52 GB):  55%|█████▌    | 32/58 [00:09<00:02,  9.47it/s]

    Capturing num tokens (num_tokens=352 avail_mem=26.21 GB):  55%|█████▌    | 32/58 [00:09<00:02,  9.47it/s]Capturing num tokens (num_tokens=352 avail_mem=26.21 GB):  59%|█████▊    | 34/58 [00:09<00:02, 10.58it/s]Capturing num tokens (num_tokens=320 avail_mem=26.23 GB):  59%|█████▊    | 34/58 [00:09<00:02, 10.58it/s]Capturing num tokens (num_tokens=288 avail_mem=26.50 GB):  59%|█████▊    | 34/58 [00:09<00:02, 10.58it/s]

    Capturing num tokens (num_tokens=288 avail_mem=26.50 GB):  62%|██████▏   | 36/58 [00:09<00:02, 10.97it/s]Capturing num tokens (num_tokens=256 avail_mem=26.49 GB):  62%|██████▏   | 36/58 [00:09<00:02, 10.97it/s]Capturing num tokens (num_tokens=240 avail_mem=26.28 GB):  62%|██████▏   | 36/58 [00:09<00:02, 10.97it/s]Capturing num tokens (num_tokens=240 avail_mem=26.28 GB):  66%|██████▌   | 38/58 [00:09<00:01, 12.42it/s]Capturing num tokens (num_tokens=224 avail_mem=26.47 GB):  66%|██████▌   | 38/58 [00:09<00:01, 12.42it/s]Capturing num tokens (num_tokens=208 avail_mem=26.46 GB):  66%|██████▌   | 38/58 [00:09<00:01, 12.42it/s]

    Capturing num tokens (num_tokens=208 avail_mem=26.46 GB):  69%|██████▉   | 40/58 [00:09<00:01, 13.29it/s]Capturing num tokens (num_tokens=192 avail_mem=26.44 GB):  69%|██████▉   | 40/58 [00:09<00:01, 13.29it/s]Capturing num tokens (num_tokens=176 avail_mem=26.43 GB):  69%|██████▉   | 40/58 [00:10<00:01, 13.29it/s]Capturing num tokens (num_tokens=176 avail_mem=26.43 GB):  72%|███████▏  | 42/58 [00:10<00:01, 14.17it/s]Capturing num tokens (num_tokens=160 avail_mem=26.43 GB):  72%|███████▏  | 42/58 [00:10<00:01, 14.17it/s]Capturing num tokens (num_tokens=144 avail_mem=26.30 GB):  72%|███████▏  | 42/58 [00:10<00:01, 14.17it/s]

    Capturing num tokens (num_tokens=144 avail_mem=26.30 GB):  76%|███████▌  | 44/58 [00:10<00:00, 14.94it/s]Capturing num tokens (num_tokens=128 avail_mem=26.39 GB):  76%|███████▌  | 44/58 [00:10<00:00, 14.94it/s]Capturing num tokens (num_tokens=112 avail_mem=26.33 GB):  76%|███████▌  | 44/58 [00:10<00:00, 14.94it/s]Capturing num tokens (num_tokens=112 avail_mem=26.33 GB):  79%|███████▉  | 46/58 [00:10<00:00, 15.52it/s]Capturing num tokens (num_tokens=96 avail_mem=26.39 GB):  79%|███████▉  | 46/58 [00:10<00:00, 15.52it/s] Capturing num tokens (num_tokens=80 avail_mem=26.37 GB):  79%|███████▉  | 46/58 [00:10<00:00, 15.52it/s]

    Capturing num tokens (num_tokens=80 avail_mem=26.37 GB):  83%|████████▎ | 48/58 [00:10<00:00, 16.41it/s]Capturing num tokens (num_tokens=64 avail_mem=26.36 GB):  83%|████████▎ | 48/58 [00:10<00:00, 16.41it/s]Capturing num tokens (num_tokens=48 avail_mem=26.35 GB):  83%|████████▎ | 48/58 [00:10<00:00, 16.41it/s]Capturing num tokens (num_tokens=48 avail_mem=26.35 GB):  86%|████████▌ | 50/58 [00:10<00:00, 17.16it/s]Capturing num tokens (num_tokens=32 avail_mem=26.34 GB):  86%|████████▌ | 50/58 [00:10<00:00, 17.16it/s]Capturing num tokens (num_tokens=28 avail_mem=26.33 GB):  86%|████████▌ | 50/58 [00:10<00:00, 17.16it/s]

    Capturing num tokens (num_tokens=24 avail_mem=26.32 GB):  86%|████████▌ | 50/58 [00:10<00:00, 17.16it/s]Capturing num tokens (num_tokens=24 avail_mem=26.32 GB):  91%|█████████▏| 53/58 [00:10<00:00, 18.42it/s]Capturing num tokens (num_tokens=20 avail_mem=26.31 GB):  91%|█████████▏| 53/58 [00:10<00:00, 18.42it/s]Capturing num tokens (num_tokens=16 avail_mem=26.31 GB):  91%|█████████▏| 53/58 [00:10<00:00, 18.42it/s]Capturing num tokens (num_tokens=12 avail_mem=26.29 GB):  91%|█████████▏| 53/58 [00:10<00:00, 18.42it/s]Capturing num tokens (num_tokens=12 avail_mem=26.29 GB):  97%|█████████▋| 56/58 [00:10<00:00, 19.19it/s]Capturing num tokens (num_tokens=8 avail_mem=26.28 GB):  97%|█████████▋| 56/58 [00:10<00:00, 19.19it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=26.27 GB):  97%|█████████▋| 56/58 [00:10<00:00, 19.19it/s]Capturing num tokens (num_tokens=4 avail_mem=26.27 GB): 100%|██████████| 58/58 [00:10<00:00,  5.31it/s]


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


<strong style='color: #00008B;'>reasoing_content: Alright, the user is in New York and wants to know the current date and time along with the weather. I need to figure out how to provide this using the allowed functions.<br><br>First, I should use the get_current_date function. The required parameter is timezone, which is 'America/New_York'. Since it's a required parameter, I don't need to provide a default. So the function call would be <function=get_current_date>{{"timezone": "America/New_York"}}</function>.<br><br>Next, for the weather, I'll use get_current_weather. The parameters needed are city, state, and unit. The city is 'New York', the state is 'NY', and the unit should be 'fahrenheit' since the user didn't specify. So the function call is <function=get_current_weather>{{"city": "New York", "state": "NY", "unit": "fahrenheit"}}</function>.<br><br>I should mention that I'm using these functions and maybe add a source note, like [TimezoneData](https://example.com/timezone) for the timezone and [WeatherAPI](https://example.com/weather) for the weather data. But since the user didn't specify a source, I'll just include the note as a reminder.<br><br>Putting it all together, I'll list each function call separately with their parameters and add the source note at the end to show where the information comes from.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>[TimezoneData](https://example.com/timezone) provided the timezone information, and [WeatherAPI](https://example.com/weather) provided the current weather data.</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'eb62042eea954ea38cea99f3b1f61375', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.748376398347318, 'response_sent_to_client_ts': 1777849543.6649425}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '22630a76a2794c0e9ac36aab408e837b', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.007744086906314, 'response_sent_to_client_ts': 1777849548.6814563}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '5ba81efbc4bb49569b70f381b90eca51', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.16584843210875988, 'response_sent_to_client_ts': 1777849548.8853557}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '5a5ec86562e24000a51f76f0c4f06e87', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.16565837897360325, 'response_sent_to_client_ts': 1777849548.8853703}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '7eb7bef8762d484c9bba3f0c2837d895', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1656025699339807, 'response_sent_to_client_ts': 1777849548.8853755}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'b5a000b94fbe4585bf67feff6dc83c8b', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 20.001234611030668, 'response_sent_to_client_ts': 1777849568.8941314}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so I need to figure out what to do for this query. The user asked for the information and population of the capital of France in JSON format. \n\nFirst, let me parse the request. They want a JSON object that includes the capital city\'s name and its population. Now, I need to make sure I get all the correct details. I know that the capital of France is Paris, so that\'s straightforward.\n\nNext, the population. The user didn\'t specify a year, so I think it\'s best to provide the most recent data I have. I recall that as of the latest estimates, Paris has a population around 21 million. I should double-check that, but since I don\'t have real-time data, I\'ll go with this number.\n\nNow, how to structure the JSON. It should have a key for the city and another for the population. Something like {"city": "Paris", "population": 21000000}. That seems simple enough.\n\nWait, should I include units for the population? The user didn\'t specify, but including \'thousand\' might make it clearer, so it could be {"city": "Paris", "population": "21 million"}. Hmm, but in JSON, numbers are typically integers or floats, not strings with units. Maybe the best way is to use the integer and let the user handle it if needed.\n\nI should also consider if the population is an approximate figure. Yes, since exact numbers vary and can be tricky to get accurately, especially for large cities. So using an approximate value is probably better.\n\nAnother thing to think about is if there are any alternative names for Paris, but I don\'t think so. It\'s commonly referred to as Paris regardless of the region or language.\n\nAlso, am I supposed to include other details? The query specifically asked for information and population, so keeping it to two keys is sufficient.\n\nAlright, I think I have a good structure and data. So the final JSON should correctly reflect the capital and its population.\n</think>\n\n```json\n{\n  "city": "Paris",\n  "population": 21000000\n}\n```', 'output_ids': [32313, 11, 773, 358, 1184, 311, 7071, 700, 1128, 311, 653, 369, 419, 3239, 13, 576, 1196, 4588, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 4710, 5338, 11, 1077, 752, 4715, 279, 1681, 13, 2379, 1366, 264, 4718, 1633, 429, 5646, 279, 6722, 3283, 594, 829, 323, 1181, 7042, 13, 4695, 11, 358, 1184, 311, 1281, 2704, 358, 633, 678, 279, 4396, 3565, 13, 358, 1414, 429, 279, 6722, 315, 9625, 374, 12095, 11, 773, 429, 594, 30339, 382, 5847, 11, 279, 7042, 13, 576, 1196, 3207, 944, 13837, 264, 1042, 11, 773, 358, 1744, 432, 594, 1850, 311, 3410, 279, 1429, 3213, 821, 358, 614, 13, 358, 19091, 429, 438, 315, 279, 5535, 17530, 11, 12095, 702, 264, 7042, 2163, 220, 17, 16, 3526, 13, 358, 1265, 1990, 15934, 429, 11, 714, 2474, 358, 1513, 944, 614, 1931, 7246, 821, 11, 358, 3278, 728, 448, 419, 1372, 382, 7039, 11, 1246, 311, 5944, 279, 4718, 13, 1084, 1265, 614, 264, 1376, 369, 279, 3283, 323, 2441, 369, 279, 7042, 13, 24656, 1075, 5212, 8926, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 15, 15, 15, 15, 15, 15, 7810, 2938, 4977, 4285, 3322, 382, 14190, 11, 1265, 358, 2924, 8153, 369, 279, 7042, 30, 576, 1196, 3207, 944, 13837, 11, 714, 2670, 364, 339, 51849, 6, 2578, 1281, 432, 48379, 11, 773, 432, 1410, 387, 5212, 8926, 788, 330, 59604, 497, 330, 44441, 788, 330, 17, 16, 3526, 1, 7810, 88190, 11, 714, 304, 4718, 11, 5109, 525, 11136, 25780, 476, 47902, 11, 537, 9069, 448, 8153, 13, 10696, 279, 1850, 1616, 374, 311, 990, 279, 7546, 323, 1077, 279, 1196, 3705, 432, 421, 4362, 382, 40, 1265, 1083, 2908, 421, 279, 7042, 374, 458, 44868, 7071, 13, 7414, 11, 2474, 4734, 5109, 13289, 323, 646, 387, 33453, 311, 633, 29257, 11, 5310, 369, 3460, 9720, 13, 2055, 1667, 458, 44868, 897, 374, 4658, 2664, 382, 14037, 3166, 311, 1744, 911, 374, 421, 1052, 525, 894, 10555, 5036, 369, 12095, 11, 714, 358, 1513, 944, 1744, 773, 13, 1084, 594, 16626, 13862, 311, 438, 12095, 15484, 315, 279, 5537, 476, 4128, 382, 13394, 11, 1079, 358, 9966, 311, 2924, 1008, 3565, 30, 576, 3239, 11689, 4588, 369, 1995, 323, 7042, 11, 773, 10282, 432, 311, 1378, 6894, 374, 14016, 382, 71486, 11, 358, 1744, 358, 614, 264, 1661, 5944, 323, 821, 13, 2055, 279, 1590, 4718, 1265, 12440, 8708, 279, 6722, 323, 1181, 7042, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 8926, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 15, 15, 15, 15, 15, 15, 198, 532, 73594, 151643], 'meta_info': {'id': '166ee9115f3f4a62af31cc714ce0cbe3', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 416, 'completion_tokens': 445, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.9581268089823425, 'response_sent_to_client_ts': 1777849572.8614655}}</strong>



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

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.20s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.21s/it]


    2026-05-03 23:06:28,532 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-03 23:06:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:44,  5.00s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:44,  5.00s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:02,  2.18s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:02,  2.18s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:10,  1.28s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:10,  1.28s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:31,  1.67it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:23,  2.24it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:23,  2.24it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.91it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.91it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.66it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.66it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.51it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.19it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.19it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.19it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.73it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.73it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.73it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.38it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.38it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.38it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.33it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.33it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.33it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:06<00:03, 11.33it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:02, 14.48it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:02, 14.48it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.48it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.48it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.48it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.74it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.74it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.74it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.74it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.74it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.74it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.91it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.91it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.91it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.91it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.91it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.91it/s]

    Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.91it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 35.12it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 35.12it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 35.12it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 35.12it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 35.12it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 35.12it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 35.12it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 35.12it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:07<00:00, 35.12it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 45.61it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 45.61it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:07<00:00, 45.61it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:07<00:00, 45.61it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:07<00:00, 45.61it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:07<00:00, 45.61it/s]

    Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:07<00:00, 45.61it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:07<00:00, 45.61it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:07<00:00, 45.61it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:07<00:00, 53.03it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:07<00:00, 53.03it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:07<00:00, 53.03it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:07<00:00, 53.03it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:07<00:00, 53.03it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:07<00:00, 53.03it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:07<00:00, 53.03it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:07<00:00, 53.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=45.39 GB):   2%|▏         | 1/58 [00:00<00:16,  3.41it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.35 GB):   2%|▏         | 1/58 [00:00<00:16,  3.41it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=45.35 GB):   3%|▎         | 2/58 [00:00<00:15,  3.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.35 GB):   3%|▎         | 2/58 [00:00<00:15,  3.51it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=45.35 GB):   5%|▌         | 3/58 [00:00<00:14,  3.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=45.35 GB):   5%|▌         | 3/58 [00:00<00:14,  3.78it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=45.35 GB):   7%|▋         | 4/58 [00:01<00:13,  4.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.35 GB):   7%|▋         | 4/58 [00:01<00:13,  4.09it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=45.35 GB):   9%|▊         | 5/58 [00:01<00:12,  4.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.35 GB):   9%|▊         | 5/58 [00:01<00:12,  4.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.35 GB):  10%|█         | 6/58 [00:01<00:10,  4.78it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.35 GB):  10%|█         | 6/58 [00:01<00:10,  4.78it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=45.35 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.35 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.35 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.35 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.71it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=45.35 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.35 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.35 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.68it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.35 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.68it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=45.35 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.22it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.35 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.22it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.35 GB):  21%|██        | 12/58 [00:02<00:05,  7.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=45.34 GB):  21%|██        | 12/58 [00:02<00:05,  7.81it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=45.34 GB):  21%|██        | 12/58 [00:02<00:05,  7.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.34 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.34 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.34 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.89it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=45.34 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=45.34 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.34 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.34 GB):  31%|███       | 18/58 [00:02<00:03, 11.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.33 GB):  31%|███       | 18/58 [00:02<00:03, 11.79it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.33 GB):  31%|███       | 18/58 [00:02<00:03, 11.79it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=45.32 GB):  31%|███       | 18/58 [00:02<00:03, 11.79it/s]Capturing num tokens (num_tokens=1024 avail_mem=45.32 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.77it/s]Capturing num tokens (num_tokens=960 avail_mem=45.31 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.77it/s] Capturing num tokens (num_tokens=896 avail_mem=45.31 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.77it/s]Capturing num tokens (num_tokens=832 avail_mem=45.31 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.77it/s]Capturing num tokens (num_tokens=832 avail_mem=45.31 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.87it/s]Capturing num tokens (num_tokens=768 avail_mem=45.30 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.87it/s]Capturing num tokens (num_tokens=704 avail_mem=45.30 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.87it/s]

    Capturing num tokens (num_tokens=640 avail_mem=45.30 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.87it/s]Capturing num tokens (num_tokens=576 avail_mem=45.29 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.87it/s]Capturing num tokens (num_tokens=576 avail_mem=45.29 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.78it/s]Capturing num tokens (num_tokens=512 avail_mem=45.29 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.78it/s]Capturing num tokens (num_tokens=480 avail_mem=45.29 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.78it/s]Capturing num tokens (num_tokens=448 avail_mem=45.28 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.78it/s]Capturing num tokens (num_tokens=416 avail_mem=45.28 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.78it/s]Capturing num tokens (num_tokens=416 avail_mem=45.28 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]

    Capturing num tokens (num_tokens=384 avail_mem=45.20 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]Capturing num tokens (num_tokens=352 avail_mem=44.15 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]

    Capturing num tokens (num_tokens=320 avail_mem=44.13 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]Capturing num tokens (num_tokens=320 avail_mem=44.13 GB):  60%|██████    | 35/58 [00:03<00:01, 16.37it/s]Capturing num tokens (num_tokens=288 avail_mem=43.65 GB):  60%|██████    | 35/58 [00:03<00:01, 16.37it/s]Capturing num tokens (num_tokens=256 avail_mem=43.49 GB):  60%|██████    | 35/58 [00:03<00:01, 16.37it/s]Capturing num tokens (num_tokens=240 avail_mem=43.49 GB):  60%|██████    | 35/58 [00:03<00:01, 16.37it/s]Capturing num tokens (num_tokens=224 avail_mem=43.48 GB):  60%|██████    | 35/58 [00:03<00:01, 16.37it/s]Capturing num tokens (num_tokens=224 avail_mem=43.48 GB):  67%|██████▋   | 39/58 [00:03<00:00, 20.61it/s]Capturing num tokens (num_tokens=208 avail_mem=43.48 GB):  67%|██████▋   | 39/58 [00:03<00:00, 20.61it/s]Capturing num tokens (num_tokens=192 avail_mem=43.48 GB):  67%|██████▋   | 39/58 [00:03<00:00, 20.61it/s]

    Capturing num tokens (num_tokens=176 avail_mem=43.47 GB):  67%|██████▋   | 39/58 [00:03<00:00, 20.61it/s]Capturing num tokens (num_tokens=160 avail_mem=43.47 GB):  67%|██████▋   | 39/58 [00:03<00:00, 20.61it/s]Capturing num tokens (num_tokens=144 avail_mem=43.47 GB):  67%|██████▋   | 39/58 [00:03<00:00, 20.61it/s]Capturing num tokens (num_tokens=144 avail_mem=43.47 GB):  76%|███████▌  | 44/58 [00:03<00:00, 25.42it/s]Capturing num tokens (num_tokens=128 avail_mem=43.47 GB):  76%|███████▌  | 44/58 [00:03<00:00, 25.42it/s]Capturing num tokens (num_tokens=112 avail_mem=43.47 GB):  76%|███████▌  | 44/58 [00:03<00:00, 25.42it/s]Capturing num tokens (num_tokens=96 avail_mem=43.46 GB):  76%|███████▌  | 44/58 [00:03<00:00, 25.42it/s] Capturing num tokens (num_tokens=80 avail_mem=43.46 GB):  76%|███████▌  | 44/58 [00:03<00:00, 25.42it/s]Capturing num tokens (num_tokens=80 avail_mem=43.46 GB):  83%|████████▎ | 48/58 [00:03<00:00, 28.44it/s]Capturing num tokens (num_tokens=64 avail_mem=43.45 GB):  83%|████████▎ | 48/58 [00:03<00:00, 28.44it/s]

    Capturing num tokens (num_tokens=48 avail_mem=43.45 GB):  83%|████████▎ | 48/58 [00:03<00:00, 28.44it/s]Capturing num tokens (num_tokens=32 avail_mem=43.45 GB):  83%|████████▎ | 48/58 [00:03<00:00, 28.44it/s]Capturing num tokens (num_tokens=28 avail_mem=43.44 GB):  83%|████████▎ | 48/58 [00:03<00:00, 28.44it/s]Capturing num tokens (num_tokens=28 avail_mem=43.44 GB):  90%|████████▉ | 52/58 [00:03<00:00, 31.05it/s]Capturing num tokens (num_tokens=24 avail_mem=43.44 GB):  90%|████████▉ | 52/58 [00:03<00:00, 31.05it/s]Capturing num tokens (num_tokens=20 avail_mem=43.44 GB):  90%|████████▉ | 52/58 [00:03<00:00, 31.05it/s]Capturing num tokens (num_tokens=16 avail_mem=43.43 GB):  90%|████████▉ | 52/58 [00:03<00:00, 31.05it/s]Capturing num tokens (num_tokens=12 avail_mem=43.43 GB):  90%|████████▉ | 52/58 [00:04<00:00, 31.05it/s]

    Capturing num tokens (num_tokens=12 avail_mem=43.43 GB):  97%|█████████▋| 56/58 [00:04<00:00, 30.71it/s]Capturing num tokens (num_tokens=8 avail_mem=43.43 GB):  97%|█████████▋| 56/58 [00:04<00:00, 30.71it/s] Capturing num tokens (num_tokens=4 avail_mem=43.42 GB):  97%|█████████▋| 56/58 [00:04<00:00, 30.71it/s]Capturing num tokens (num_tokens=4 avail_mem=43.42 GB): 100%|██████████| 58/58 [00:04<00:00, 14.19it/s]


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
    
    Generated text: Okay, so the user asked for the information and population of the capital of France in JSON format. The capital is Paris, so I need to provide accurate data. I should make sure to include the most recent population figure. I think as of 2023, the population was around 2,100,000. It's important to note that population figures can change, so I'll include a note about the data being up to date as of that year. I'll structure the JSON with the country name, capital, population, and a note. Keeping it simple and clear for the user.
    </think>
    
    ```json
    {
      "country": "France",
      "capital": "Paris",
      "population": 2100000,
      "note": "Population figure is approximate and may vary slightly depending on the source and year of data."
    }
    ```



```python
llm.shutdown()
```
