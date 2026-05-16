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

    Multi-thread loading shards:  50% Completed | 1/2 [00:02<00:02,  2.51s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:04<00:00,  2.31s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:04<00:00,  2.34s/it]


    2026-05-16 14:07:36,770 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 14:07:36] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:35,  4.84s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:35,  4.84s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:00,  2.16s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:00,  2.16s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:19,  1.45s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:19,  1.45s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<01:00,  1.12s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<01:00,  1.12s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:47,  1.11it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:47,  1.11it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:39,  1.31it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:39,  1.31it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:33,  1.52it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:33,  1.52it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:28,  1.74it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:28,  1.74it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:24,  1.98it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:24,  1.98it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:21,  2.24it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:21,  2.24it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:18,  2.49it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:18,  2.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:16,  2.76it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:16,  2.76it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:14,  3.05it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:14,  3.05it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:13,  3.29it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:13,  3.29it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:10<00:11,  3.63it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:10<00:11,  3.63it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:10<00:10,  4.04it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:10<00:10,  4.04it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:10<00:09,  4.50it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:10<00:09,  4.50it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:10<00:08,  4.97it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:10<00:08,  4.97it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:10<00:07,  5.56it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:10<00:07,  5.56it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:10<00:06,  6.03it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:10<00:06,  6.03it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:11<00:05,  6.68it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:11<00:05,  6.68it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:11<00:05,  6.68it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:11<00:04,  8.48it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:11<00:04,  8.48it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:11<00:04,  8.48it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:11<00:03,  9.76it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:11<00:03,  9.76it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:11<00:03,  9.76it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:11<00:02, 11.27it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:11<00:02, 11.27it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:11<00:02, 11.27it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:11<00:02, 12.71it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:11<00:02, 12.71it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:11<00:02, 12.71it/s]

    Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:11<00:02, 12.71it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:11<00:01, 14.72it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:11<00:01, 14.72it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:11<00:01, 14.72it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:11<00:01, 14.72it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:11<00:01, 16.51it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:11<00:01, 16.51it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:11<00:01, 16.51it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:11<00:01, 16.51it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:12<00:01, 18.45it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:12<00:01, 18.45it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:12<00:01, 18.45it/s]

    Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:12<00:01, 18.45it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:12<00:00, 20.03it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:12<00:00, 20.03it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:12<00:00, 20.03it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:12<00:00, 20.03it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:12<00:00, 21.62it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:12<00:00, 21.62it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:12<00:00, 21.62it/s]

    Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:12<00:00, 21.62it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:12<00:00, 21.62it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:12<00:00, 24.50it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:12<00:00, 24.50it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:12<00:00, 24.50it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:12<00:00, 24.50it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:12<00:00, 24.50it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:12<00:00, 27.51it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:12<00:00, 27.51it/s]

    Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:12<00:00, 27.51it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:12<00:00, 27.51it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:12<00:00, 27.51it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:12<00:00, 27.51it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:12<00:00, 31.96it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:12<00:00, 31.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:12<00:00,  4.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=26.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=26.79 GB):   2%|▏         | 1/58 [00:00<00:41,  1.37it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.48 GB):   2%|▏         | 1/58 [00:00<00:41,  1.37it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.48 GB):   3%|▎         | 2/58 [00:01<00:41,  1.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=24.71 GB):   3%|▎         | 2/58 [00:01<00:41,  1.36it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=24.71 GB):   5%|▌         | 3/58 [00:02<00:40,  1.34it/s]Capturing num tokens (num_tokens=6656 avail_mem=24.84 GB):   5%|▌         | 3/58 [00:02<00:40,  1.34it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=24.84 GB):   7%|▋         | 4/58 [00:02<00:39,  1.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=24.97 GB):   7%|▋         | 4/58 [00:02<00:39,  1.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=24.97 GB):   9%|▊         | 5/58 [00:03<00:36,  1.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.11 GB):   9%|▊         | 5/58 [00:03<00:36,  1.44it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.11 GB):  10%|█         | 6/58 [00:04<00:33,  1.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.23 GB):  10%|█         | 6/58 [00:04<00:33,  1.55it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.23 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.36 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.66it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=25.36 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=25.49 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.82it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=25.49 GB):  16%|█▌        | 9/58 [00:05<00:25,  1.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.63 GB):  16%|█▌        | 9/58 [00:05<00:25,  1.93it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=25.63 GB):  17%|█▋        | 10/58 [00:05<00:22,  2.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=25.69 GB):  17%|█▋        | 10/58 [00:05<00:22,  2.10it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=25.69 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=25.75 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.31it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=25.75 GB):  21%|██        | 12/58 [00:06<00:18,  2.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=26.23 GB):  21%|██        | 12/58 [00:06<00:18,  2.50it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=26.23 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.26 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.74it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.26 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.29 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.01it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.29 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.32 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.32it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=26.32 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.35 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.35 GB):  29%|██▉       | 17/58 [00:07<00:10,  4.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.38 GB):  29%|██▉       | 17/58 [00:07<00:10,  4.07it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=26.38 GB):  31%|███       | 18/58 [00:07<00:08,  4.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.40 GB):  31%|███       | 18/58 [00:07<00:08,  4.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.40 GB):  33%|███▎      | 19/58 [00:08<00:07,  4.95it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.42 GB):  33%|███▎      | 19/58 [00:08<00:07,  4.95it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=26.42 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.36 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.36 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.36it/s]Capturing num tokens (num_tokens=960 avail_mem=26.41 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.36it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=26.41 GB):  38%|███▊      | 22/58 [00:08<00:05,  7.13it/s]Capturing num tokens (num_tokens=896 avail_mem=26.42 GB):  38%|███▊      | 22/58 [00:08<00:05,  7.13it/s]Capturing num tokens (num_tokens=896 avail_mem=26.42 GB):  40%|███▉      | 23/58 [00:08<00:04,  7.74it/s]Capturing num tokens (num_tokens=832 avail_mem=26.32 GB):  40%|███▉      | 23/58 [00:08<00:04,  7.74it/s]

    Capturing num tokens (num_tokens=768 avail_mem=26.44 GB):  40%|███▉      | 23/58 [00:08<00:04,  7.74it/s]Capturing num tokens (num_tokens=768 avail_mem=26.44 GB):  43%|████▎     | 25/58 [00:08<00:03,  8.99it/s]Capturing num tokens (num_tokens=704 avail_mem=26.43 GB):  43%|████▎     | 25/58 [00:08<00:03,  8.99it/s]Capturing num tokens (num_tokens=640 avail_mem=26.43 GB):  43%|████▎     | 25/58 [00:08<00:03,  8.99it/s]

    Capturing num tokens (num_tokens=640 avail_mem=26.43 GB):  47%|████▋     | 27/58 [00:08<00:03, 10.27it/s]Capturing num tokens (num_tokens=576 avail_mem=26.44 GB):  47%|████▋     | 27/58 [00:08<00:03, 10.27it/s]Capturing num tokens (num_tokens=512 avail_mem=26.44 GB):  47%|████▋     | 27/58 [00:08<00:03, 10.27it/s]Capturing num tokens (num_tokens=512 avail_mem=26.44 GB):  50%|█████     | 29/58 [00:08<00:02, 11.53it/s]Capturing num tokens (num_tokens=480 avail_mem=26.45 GB):  50%|█████     | 29/58 [00:08<00:02, 11.53it/s]Capturing num tokens (num_tokens=448 avail_mem=26.43 GB):  50%|█████     | 29/58 [00:09<00:02, 11.53it/s]

    Capturing num tokens (num_tokens=448 avail_mem=26.43 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.48it/s]Capturing num tokens (num_tokens=416 avail_mem=26.42 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.48it/s]Capturing num tokens (num_tokens=384 avail_mem=26.41 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.48it/s]Capturing num tokens (num_tokens=384 avail_mem=26.41 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.35it/s]Capturing num tokens (num_tokens=352 avail_mem=26.41 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.35it/s]Capturing num tokens (num_tokens=320 avail_mem=26.39 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.35it/s]

    Capturing num tokens (num_tokens=320 avail_mem=26.39 GB):  60%|██████    | 35/58 [00:09<00:01, 14.92it/s]Capturing num tokens (num_tokens=288 avail_mem=26.40 GB):  60%|██████    | 35/58 [00:09<00:01, 14.92it/s]Capturing num tokens (num_tokens=256 avail_mem=26.40 GB):  60%|██████    | 35/58 [00:09<00:01, 14.92it/s]Capturing num tokens (num_tokens=256 avail_mem=26.40 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.69it/s]Capturing num tokens (num_tokens=240 avail_mem=26.42 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.69it/s]Capturing num tokens (num_tokens=224 avail_mem=26.41 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.69it/s]

    Capturing num tokens (num_tokens=208 avail_mem=26.40 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.69it/s]Capturing num tokens (num_tokens=208 avail_mem=26.40 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.29it/s]Capturing num tokens (num_tokens=192 avail_mem=26.39 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.29it/s]Capturing num tokens (num_tokens=176 avail_mem=26.38 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.29it/s]Capturing num tokens (num_tokens=160 avail_mem=26.37 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.29it/s]Capturing num tokens (num_tokens=160 avail_mem=26.37 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.44it/s]Capturing num tokens (num_tokens=144 avail_mem=26.36 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.44it/s]

    Capturing num tokens (num_tokens=128 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.44it/s]Capturing num tokens (num_tokens=112 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.44it/s]Capturing num tokens (num_tokens=112 avail_mem=26.35 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.29it/s]Capturing num tokens (num_tokens=96 avail_mem=26.34 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.29it/s] Capturing num tokens (num_tokens=80 avail_mem=26.32 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.29it/s]Capturing num tokens (num_tokens=64 avail_mem=26.31 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.29it/s]

    Capturing num tokens (num_tokens=64 avail_mem=26.31 GB):  84%|████████▍ | 49/58 [00:09<00:00, 20.16it/s]Capturing num tokens (num_tokens=48 avail_mem=26.30 GB):  84%|████████▍ | 49/58 [00:09<00:00, 20.16it/s]Capturing num tokens (num_tokens=32 avail_mem=26.29 GB):  84%|████████▍ | 49/58 [00:10<00:00, 20.16it/s]Capturing num tokens (num_tokens=28 avail_mem=26.28 GB):  84%|████████▍ | 49/58 [00:10<00:00, 20.16it/s]Capturing num tokens (num_tokens=28 avail_mem=26.28 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.84it/s]Capturing num tokens (num_tokens=24 avail_mem=26.27 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.84it/s]Capturing num tokens (num_tokens=20 avail_mem=26.26 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.84it/s]

    Capturing num tokens (num_tokens=16 avail_mem=26.25 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.84it/s]Capturing num tokens (num_tokens=16 avail_mem=26.25 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.37it/s]Capturing num tokens (num_tokens=12 avail_mem=26.24 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.37it/s]Capturing num tokens (num_tokens=8 avail_mem=26.23 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.37it/s] Capturing num tokens (num_tokens=4 avail_mem=26.22 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.37it/s]Capturing num tokens (num_tokens=4 avail_mem=26.22 GB): 100%|██████████| 58/58 [00:10<00:00, 21.71it/s]Capturing num tokens (num_tokens=4 avail_mem=26.22 GB): 100%|██████████| 58/58 [00:10<00:00,  5.58it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out how to get the current date and time in New York and the weather there. Let me start by understanding what the user is asking for. They mentioned being in New York, so I should get the date and time for that location. Also, they want the weather, which means I'll probably need to use the get_current_weather function.<br><br>First, I'll tackle the date and time. I remember there's a function called get_current_date that requires a timezone parameter. The user is in New York, so I should use the 'America/New_York' timezone. I'll structure the function call with timezone set to that string.<br><br>Next, for the weather, I'll use get_current_weather. Since the user is in New York, I need to provide the city and state. New York's state is NY, so the city is 'New York' and state is 'NY'. The unit isn't specified, so I'll leave it out or set it to default, which is usually Fahrenheit, but the function allows 'celsius' or 'fahrenheit' as options.<br><br>I should make sure each function call is separate and follow the required format. I'll write each function call on a new line, using the correct syntax with the function name, parameters in JSON, and the appropriate start and end tags.<br><br>Wait, do I need to specify the unit for get_current_weather? The function description says it's optional, so maybe I can omit it. I'll check the parameters again. The unit is optional, so it's fine to leave it out. That way, the function will return the temperature in the user's default unit, which is probably Fahrenheit since they're in the US.<br><br>Putting it all together, I'll first call get_current_date with the timezone parameter, then get_current_weather with city and state. I'll make sure each is on its own line and formatted correctly with the start and end tags. Also, I should include the sources from the functions' descriptions to provide context for the user.<br><br>I think that's all. I'll structure the response with both function calls, each on their own line, and include the necessary parameters.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY"  ,"unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'fbd66fa025354a8987d782b7aba348f1', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.8583813533186913, 'response_sent_to_client_ts': 1778940533.7493348}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '7cd45fdf969b48cf971722e189d5aacc', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 21.183798940852284, 'response_sent_to_client_ts': 1778940554.9414642}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '49c47b8fef7848f38c3d855de2d5273f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.16894224099814892, 'response_sent_to_client_ts': 1778940555.154123}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '385d6896577b48499f3127903494ee89', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1688681785017252, 'response_sent_to_client_ts': 1778940555.1541317}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '6208830b240541c2929405146da7af63', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.16882266849279404, 'response_sent_to_client_ts': 1778940555.1541348}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '360f2eb9e83344978121d5a26d6ff229', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 17.377135816961527, 'response_sent_to_client_ts': 1778940572.5389705}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so I need to figure out the information and population of the capital of France, which is Paris. Let me start by recalling what I know about Paris. I know it\'s the most visited city in the world and the political center of France. But for the population, I\'m not exactly sure. I think it\'s a big city, so maybe millions of people live there, but I\'m not certain about the exact number.\n\nWait, I remember hearing that Paris has a population over 30 million. But when I think about it, that seems high. I might be confusing it with another major city. Maybe I should break this down. Paris is known for its attractions like the Eiffel Tower, the Louvre, and places like RATP, which I think is the public transportation system, pronounced \' liberal\' maybe? Not sure. Then there\'s Le Marais, a historic district. I\'m trying to link this to population.\n\nSo, if I\'m recalling correctly, Paris is the capital with a population over 30 million. But I think in reality, it\'s a bit less. I might be mixing it up with Tokyo, which is the most populous city in Japan but definitely not in France. Wait, Tokyo\'s population is around 30 million, but I think Paris might be a bit higher. Alternatively, I could be wrong.\n\nMaybe I should think about the data I remember. I think it was somewhere around 21 million. Hmm, so is it about 21.5 million? I\'m not sure. I should try to find a more accurate figure. I know that major cities in Europe vary widely. For example, Berlin has about 3.7 million, London has over 9 million in the metro area, but Paris is much bigger.\n\nI think I\'ve heard that the population of Paris alone is about 21 million, not the metro area. So, that\'s the city proper, right? The greater Paris area might have more people, including the suburbs, but the city itself is around 21 million.\n\nSo putting it all together, the capital of France is Paris, and its population is approximately 21.5 million. That sounds right based on what I remember. Now, I need to structure this into JSON format. So, the JSON should have a key like "capital" with the value being "Paris" and another key "population" with the value being "21500000".\n\nI should make sure the population number is formatted correctly in the JSON as an integer without commas for consistency, or should I use a number with commas? JSON accepts both, but for clarity, using a comma-separated number might be better. Alternatively, I can write it as 21500000.\n\nAlso, the capital for Paris is definitely the political center, so that\'s correct. Therefore, the final JSON object should reflect that Paris is the capital and its population is approximately 21.5 million.\n\nI should double-check to make sure I didn\'t mix up numbers with other cities. For example, Lyon has around 500,000, and Rouen is another major city, but their populations are much lower. So, Paris at over 20 million is in the right range.\n\nIn conclusion, Paris is the capital of France with a population of approximately 21.5 million people. The JSON format will have the keys "capital" and "population" with their respective values.\n</think>\n\n```json\n{\n  "capital": "Paris",\n  "population": 21500000\n}\n```', 'output_ids': [32313, 11, 773, 358, 1184, 311, 7071, 700, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 11, 892, 374, 12095, 13, 6771, 752, 1191, 553, 88646, 1128, 358, 1414, 911, 12095, 13, 358, 1414, 432, 594, 279, 1429, 11994, 3283, 304, 279, 1879, 323, 279, 4948, 4126, 315, 9625, 13, 1988, 369, 279, 7042, 11, 358, 2776, 537, 6896, 2704, 13, 358, 1744, 432, 594, 264, 2409, 3283, 11, 773, 7196, 11728, 315, 1251, 3887, 1052, 11, 714, 358, 2776, 537, 3654, 911, 279, 4734, 1372, 382, 14190, 11, 358, 6099, 10778, 429, 12095, 702, 264, 7042, 916, 220, 18, 15, 3526, 13, 1988, 979, 358, 1744, 911, 432, 11, 429, 4977, 1550, 13, 358, 2578, 387, 30615, 432, 448, 2441, 3598, 3283, 13, 10696, 358, 1265, 1438, 419, 1495, 13, 12095, 374, 3881, 369, 1181, 38491, 1075, 279, 468, 3092, 301, 21938, 11, 279, 9729, 48506, 11, 323, 7482, 1075, 97880, 47, 11, 892, 358, 1744, 374, 279, 584, 17903, 1849, 11, 37517, 364, 17775, 6, 7196, 30, 2806, 2704, 13, 5005, 1052, 594, 1967, 2876, 2782, 11, 264, 18035, 9290, 13, 358, 2776, 4460, 311, 2656, 419, 311, 7042, 382, 4416, 11, 421, 358, 2776, 88646, 12440, 11, 12095, 374, 279, 6722, 448, 264, 7042, 916, 220, 18, 15, 3526, 13, 1988, 358, 1744, 304, 8729, 11, 432, 594, 264, 2699, 2686, 13, 358, 2578, 387, 26792, 432, 705, 448, 26194, 11, 892, 374, 279, 1429, 94451, 3283, 304, 6323, 714, 8491, 537, 304, 9625, 13, 13824, 11, 26194, 594, 7042, 374, 2163, 220, 18, 15, 3526, 11, 714, 358, 1744, 12095, 2578, 387, 264, 2699, 5080, 13, 38478, 11, 358, 1410, 387, 4969, 382, 21390, 358, 1265, 1744, 911, 279, 821, 358, 6099, 13, 358, 1744, 432, 572, 14696, 2163, 220, 17, 16, 3526, 13, 88190, 11, 773, 374, 432, 911, 220, 17, 16, 13, 20, 3526, 30, 358, 2776, 537, 2704, 13, 358, 1265, 1430, 311, 1477, 264, 803, 13382, 7071, 13, 358, 1414, 429, 3598, 9720, 304, 4505, 13289, 13570, 13, 1752, 3110, 11, 19846, 702, 911, 220, 18, 13, 22, 3526, 11, 7148, 702, 916, 220, 24, 3526, 304, 279, 33482, 3082, 11, 714, 12095, 374, 1753, 11243, 382, 40, 1744, 358, 3003, 6617, 429, 279, 7042, 315, 12095, 7484, 374, 911, 220, 17, 16, 3526, 11, 537, 279, 33482, 3082, 13, 2055, 11, 429, 594, 279, 3283, 6169, 11, 1290, 30, 576, 7046, 12095, 3082, 2578, 614, 803, 1251, 11, 2670, 279, 46913, 11, 714, 279, 3283, 5086, 374, 2163, 220, 17, 16, 3526, 382, 4416, 10687, 432, 678, 3786, 11, 279, 6722, 315, 9625, 374, 12095, 11, 323, 1181, 7042, 374, 13187, 220, 17, 16, 13, 20, 3526, 13, 2938, 10362, 1290, 3118, 389, 1128, 358, 6099, 13, 4695, 11, 358, 1184, 311, 5944, 419, 1119, 4718, 3561, 13, 2055, 11, 279, 4718, 1265, 614, 264, 1376, 1075, 330, 65063, 1, 448, 279, 897, 1660, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 897, 1660, 330, 17, 16, 20, 15, 15, 15, 15, 15, 11436, 40, 1265, 1281, 2704, 279, 7042, 1372, 374, 23126, 12440, 304, 279, 4718, 438, 458, 7546, 2041, 76602, 369, 28137, 11, 476, 1265, 358, 990, 264, 1372, 448, 76602, 30, 4718, 26344, 2176, 11, 714, 369, 31273, 11, 1667, 264, 31683, 72692, 1372, 2578, 387, 2664, 13, 38478, 11, 358, 646, 3270, 432, 438, 220, 17, 16, 20, 15, 15, 15, 15, 15, 382, 13394, 11, 279, 6722, 369, 12095, 374, 8491, 279, 4948, 4126, 11, 773, 429, 594, 4396, 13, 15277, 11, 279, 1590, 4718, 1633, 1265, 8708, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 374, 13187, 220, 17, 16, 13, 20, 3526, 382, 40, 1265, 1990, 15934, 311, 1281, 2704, 358, 3207, 944, 6514, 705, 5109, 448, 1008, 9720, 13, 1752, 3110, 11, 55201, 702, 2163, 220, 20, 15, 15, 11, 15, 15, 15, 11, 323, 28522, 268, 374, 2441, 3598, 3283, 11, 714, 862, 21910, 525, 1753, 4722, 13, 2055, 11, 12095, 518, 916, 220, 17, 15, 3526, 374, 304, 279, 1290, 2088, 382, 641, 16688, 11, 12095, 374, 279, 6722, 315, 9625, 448, 264, 7042, 315, 13187, 220, 17, 16, 13, 20, 3526, 1251, 13, 576, 4718, 3561, 686, 614, 279, 6894, 330, 65063, 1, 323, 330, 44441, 1, 448, 862, 19511, 2750, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 15, 198, 532, 73594, 151643], 'meta_info': {'id': '0991abe267334363afe49f8c3acd2e8f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 717, 'completion_tokens': 746, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 6.2269023433327675, 'response_sent_to_client_ts': 1778940578.7735913}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.41s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.38s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.38s/it]


    2026-05-16 14:09:56,657 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 14:09:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:57,  5.23s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:57,  5.23s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:07,  2.28s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:07,  2.28s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:12,  1.33s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:12,  1.33s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.14it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.14it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.63it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.18it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.18it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.85it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.85it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.59it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.59it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.46it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.46it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.46it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.19it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.19it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.19it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.45it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.45it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.45it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.41it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.41it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.41it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.41it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.72it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.72it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.72it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.72it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.72it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 20.22it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 20.22it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 20.22it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 20.22it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.22it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.22it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.22it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 29.03it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:07<00:00, 31.65it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:07<00:00, 31.65it/s]

    Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:07<00:00, 31.65it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:07<00:00, 31.65it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:07<00:00, 31.65it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:07<00:00, 31.65it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:07<00:00, 31.65it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:07<00:00, 38.30it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:07<00:00, 38.30it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:07<00:00, 38.30it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:07<00:00, 38.30it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:07<00:00, 38.30it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:07<00:00, 38.30it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:07<00:00, 38.30it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:07<00:00, 38.30it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:07<00:00, 46.83it/s]

    Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:07<00:00, 46.83it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:07<00:00, 55.84it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:07<00:00, 55.84it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:07<00:00, 55.84it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:07<00:00, 55.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.03 GB):   2%|▏         | 1/58 [00:00<00:16,  3.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.00 GB):   2%|▏         | 1/58 [00:00<00:16,  3.47it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.00 GB):   3%|▎         | 2/58 [00:00<00:15,  3.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.00 GB):   3%|▎         | 2/58 [00:00<00:15,  3.63it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.00 GB):   5%|▌         | 3/58 [00:00<00:14,  3.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.00 GB):   5%|▌         | 3/58 [00:00<00:14,  3.86it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.00 GB):   7%|▋         | 4/58 [00:01<00:13,  4.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.00 GB):   7%|▋         | 4/58 [00:01<00:13,  4.15it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.00 GB):   9%|▊         | 5/58 [00:01<00:12,  4.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.00 GB):   9%|▊         | 5/58 [00:01<00:12,  4.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.00 GB):  10%|█         | 6/58 [00:01<00:10,  4.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.00 GB):  10%|█         | 6/58 [00:01<00:10,  4.79it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.00 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.00 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.00 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=25.00 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.65it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=25.00 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.00 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.00 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=25.00 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.64it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=25.00 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=24.99 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=24.99 GB):  21%|██        | 12/58 [00:02<00:05,  7.82it/s]Capturing num tokens (num_tokens=3072 avail_mem=24.99 GB):  21%|██        | 12/58 [00:02<00:05,  7.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=24.99 GB):  21%|██        | 12/58 [00:02<00:05,  7.82it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=24.99 GB):  24%|██▍       | 14/58 [00:02<00:04,  9.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=24.99 GB):  24%|██▍       | 14/58 [00:02<00:04,  9.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=24.99 GB):  24%|██▍       | 14/58 [00:02<00:04,  9.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=24.99 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.47it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.98 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.47it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=24.98 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=24.98 GB):  31%|███       | 18/58 [00:02<00:03, 12.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=24.98 GB):  31%|███       | 18/58 [00:02<00:03, 12.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=24.98 GB):  31%|███       | 18/58 [00:02<00:03, 12.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.97 GB):  31%|███       | 18/58 [00:02<00:03, 12.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.97 GB):  36%|███▌      | 21/58 [00:02<00:02, 15.17it/s]Capturing num tokens (num_tokens=960 avail_mem=24.96 GB):  36%|███▌      | 21/58 [00:02<00:02, 15.17it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=24.96 GB):  36%|███▌      | 21/58 [00:02<00:02, 15.17it/s]Capturing num tokens (num_tokens=832 avail_mem=24.96 GB):  36%|███▌      | 21/58 [00:02<00:02, 15.17it/s]Capturing num tokens (num_tokens=832 avail_mem=24.96 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.18it/s]Capturing num tokens (num_tokens=768 avail_mem=24.95 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.18it/s]Capturing num tokens (num_tokens=704 avail_mem=24.95 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.18it/s]Capturing num tokens (num_tokens=640 avail_mem=24.94 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.18it/s]Capturing num tokens (num_tokens=640 avail_mem=24.94 GB):  47%|████▋     | 27/58 [00:02<00:01, 21.04it/s]Capturing num tokens (num_tokens=576 avail_mem=24.94 GB):  47%|████▋     | 27/58 [00:02<00:01, 21.04it/s]

    Capturing num tokens (num_tokens=512 avail_mem=24.93 GB):  47%|████▋     | 27/58 [00:02<00:01, 21.04it/s]Capturing num tokens (num_tokens=480 avail_mem=24.93 GB):  47%|████▋     | 27/58 [00:03<00:01, 21.04it/s]Capturing num tokens (num_tokens=448 avail_mem=24.93 GB):  47%|████▋     | 27/58 [00:03<00:01, 21.04it/s]Capturing num tokens (num_tokens=448 avail_mem=24.93 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.47it/s]Capturing num tokens (num_tokens=416 avail_mem=24.92 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.47it/s]Capturing num tokens (num_tokens=384 avail_mem=24.92 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.47it/s]Capturing num tokens (num_tokens=352 avail_mem=24.91 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.47it/s]Capturing num tokens (num_tokens=320 avail_mem=24.91 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.47it/s]

    Capturing num tokens (num_tokens=320 avail_mem=24.91 GB):  60%|██████    | 35/58 [00:03<00:00, 27.73it/s]Capturing num tokens (num_tokens=288 avail_mem=24.92 GB):  60%|██████    | 35/58 [00:03<00:00, 27.73it/s]Capturing num tokens (num_tokens=256 avail_mem=24.91 GB):  60%|██████    | 35/58 [00:03<00:00, 27.73it/s]Capturing num tokens (num_tokens=240 avail_mem=24.91 GB):  60%|██████    | 35/58 [00:03<00:00, 27.73it/s]Capturing num tokens (num_tokens=224 avail_mem=24.90 GB):  60%|██████    | 35/58 [00:03<00:00, 27.73it/s]Capturing num tokens (num_tokens=224 avail_mem=24.90 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]Capturing num tokens (num_tokens=208 avail_mem=24.90 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]Capturing num tokens (num_tokens=192 avail_mem=24.89 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]Capturing num tokens (num_tokens=176 avail_mem=24.89 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]Capturing num tokens (num_tokens=160 avail_mem=24.89 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.25it/s]

    Capturing num tokens (num_tokens=160 avail_mem=24.89 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.73it/s]Capturing num tokens (num_tokens=144 avail_mem=24.88 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.73it/s]Capturing num tokens (num_tokens=128 avail_mem=24.88 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.73it/s]Capturing num tokens (num_tokens=112 avail_mem=24.88 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.73it/s]Capturing num tokens (num_tokens=96 avail_mem=24.88 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.73it/s] Capturing num tokens (num_tokens=96 avail_mem=24.88 GB):  81%|████████  | 47/58 [00:03<00:00, 34.32it/s]Capturing num tokens (num_tokens=80 avail_mem=24.87 GB):  81%|████████  | 47/58 [00:03<00:00, 34.32it/s]Capturing num tokens (num_tokens=64 avail_mem=24.87 GB):  81%|████████  | 47/58 [00:03<00:00, 34.32it/s]Capturing num tokens (num_tokens=48 avail_mem=24.86 GB):  81%|████████  | 47/58 [00:03<00:00, 34.32it/s]Capturing num tokens (num_tokens=32 avail_mem=24.86 GB):  81%|████████  | 47/58 [00:03<00:00, 34.32it/s]

    Capturing num tokens (num_tokens=28 avail_mem=24.86 GB):  81%|████████  | 47/58 [00:03<00:00, 34.32it/s]Capturing num tokens (num_tokens=28 avail_mem=24.86 GB):  90%|████████▉ | 52/58 [00:03<00:00, 36.26it/s]Capturing num tokens (num_tokens=24 avail_mem=24.86 GB):  90%|████████▉ | 52/58 [00:03<00:00, 36.26it/s]Capturing num tokens (num_tokens=20 avail_mem=24.85 GB):  90%|████████▉ | 52/58 [00:03<00:00, 36.26it/s]Capturing num tokens (num_tokens=16 avail_mem=24.85 GB):  90%|████████▉ | 52/58 [00:03<00:00, 36.26it/s]Capturing num tokens (num_tokens=12 avail_mem=24.81 GB):  90%|████████▉ | 52/58 [00:03<00:00, 36.26it/s]Capturing num tokens (num_tokens=8 avail_mem=24.81 GB):  90%|████████▉ | 52/58 [00:03<00:00, 36.26it/s] Capturing num tokens (num_tokens=8 avail_mem=24.81 GB):  98%|█████████▊| 57/58 [00:03<00:00, 37.55it/s]Capturing num tokens (num_tokens=4 avail_mem=24.81 GB):  98%|█████████▊| 57/58 [00:03<00:00, 37.55it/s]Capturing num tokens (num_tokens=4 avail_mem=24.81 GB): 100%|██████████| 58/58 [00:03<00:00, 15.36it/s]


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
    
    Generated text: Okay, so the user is asking for the information and population of the capital of France in JSON format. Hmm, the capital of France is definitely Paris. I remember that Paris is a major city there. Now, they want the population data. I think I should look up the latest data. But wait, population numbers can change every year, so I need to make sure I have the most recent figures.
    
    I recall that Paris is the largest city in France and one of the biggest in the world. Let me check some recent statistics. I think the population is around 2 million, but I'm not 100% sure. Maybe I should verify this from a reliable source. Oh, right, the World Bank or the latest census would have this information. 
    
    I should also consider whether they want just the population or other details. The user mentioned "information and population," so I should include other relevant data like area, nickname, and the main language spoken. That makes the JSON more comprehensive and useful for them. 
    
    So, structuring the JSON, I'll have an object with the key "capital" containing another object with the keys like "city", "country", "area", " Fronde Nickname", and "main language", each with their respective values. Then, another object for the population with subkeys like "current population", "year", and "approximation". 
    
    Wait, the user might be using this data for a project or maybe just curious. They probably need it in a structured format, hence JSON. They didn't specify if they need an exact number or an approximate figure. Since population data can fluctuate, providing the approximate with a year would be helpful. 
    
    I should make sure the JSON is properly formatted, with commas in the right places and the keys correctly spelled. Also, the population figure I have in mind is around 2.17 million as of 2023, so I'll include that with the year 2023 and mention it's approximate. 
    
    Double-checking everything, I think that's all the information needed. I'll present it clearly so the user can easily access the data. Maybe they'll need it for a presentation or a report, so accuracy is key. I hope I didn't mix up any details about Paris. Let me just confirm once more—capital is Paris, country is France, population around 2.17 million as of 2023. Yep, that seems right.
    </think>
    
    Here is the information about the capital of France in JSON format:
    
    ```json
    {
      "capital": {
        "city": "Paris",
        "country": "France",
        "area": "107.5 square kilometers",
        "fronde Nickname": "The City of Light and the City of Love",
        "main language": "French"
      },
      "population": {
        "current population": "2,170,000",
        "year": "2023",
        "approximation": "Approximate population"
      }
    }
    ```



```python
llm.shutdown()
```
