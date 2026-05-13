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


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.75s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.67s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.68s/it]


    2026-05-13 07:37:41,377 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 07:37:41] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:14,  2.40s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:14,  2.40s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:27,  1.60s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:27,  1.60s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<01:05,  1.20s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<01:05,  1.20s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:50,  1.05it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:50,  1.05it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:41,  1.24it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:41,  1.24it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:34,  1.47it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:34,  1.47it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:29,  1.68it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:29,  1.68it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:25,  1.93it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:25,  1.93it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:21,  2.20it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:21,  2.20it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:19,  2.46it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:19,  2.46it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:16,  2.73it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:16,  2.73it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:10<00:14,  3.02it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:10<00:14,  3.02it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:10<00:13,  3.32it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:10<00:13,  3.32it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:10<00:11,  3.63it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:10<00:11,  3.63it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:10<00:10,  4.01it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:10<00:10,  4.01it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:10<00:09,  4.46it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:10<00:09,  4.46it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:11<00:08,  4.95it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:11<00:08,  4.95it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:11<00:07,  5.50it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:11<00:07,  5.50it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:11<00:06,  6.18it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:11<00:06,  6.18it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:11<00:05,  6.96it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:11<00:05,  6.96it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:11<00:05,  6.96it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:11<00:03,  8.80it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:11<00:03,  8.80it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:11<00:03,  8.80it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:11<00:03, 10.15it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:11<00:03, 10.15it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:11<00:03, 10.15it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:11<00:02, 11.65it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:11<00:02, 11.65it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:11<00:02, 11.65it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:12<00:02, 13.03it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:12<00:02, 13.03it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:12<00:02, 13.03it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:12<00:01, 14.61it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:12<00:01, 14.61it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:12<00:01, 14.61it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:12<00:01, 14.61it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:12<00:01, 16.73it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:12<00:01, 16.73it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:12<00:01, 16.73it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:12<00:01, 16.73it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:12<00:01, 19.09it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:12<00:01, 19.09it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:12<00:01, 19.09it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:12<00:01, 19.09it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:12<00:01, 19.09it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:12<00:00, 22.71it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:12<00:00, 22.71it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:12<00:00, 22.71it/s]

    Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:12<00:00, 22.71it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:12<00:00, 22.71it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:12<00:00, 26.06it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:12<00:00, 26.06it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:12<00:00, 26.06it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:12<00:00, 26.06it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:12<00:00, 26.06it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:12<00:00, 27.93it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:12<00:00, 27.93it/s]

    Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:12<00:00, 27.93it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:12<00:00, 27.93it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:12<00:00, 27.93it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:12<00:00, 30.34it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:12<00:00, 30.34it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:12<00:00, 30.34it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:12<00:00, 30.34it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:12<00:00, 30.34it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:12<00:00, 30.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:12<00:00, 35.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:12<00:00,  4.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=26.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=26.79 GB):   2%|▏         | 1/58 [00:00<00:40,  1.42it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.58 GB):   2%|▏         | 1/58 [00:00<00:40,  1.42it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.58 GB):   3%|▎         | 2/58 [00:01<00:41,  1.34it/s]Capturing num tokens (num_tokens=7168 avail_mem=24.71 GB):   3%|▎         | 2/58 [00:01<00:41,  1.34it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=24.71 GB):   5%|▌         | 3/58 [00:02<00:41,  1.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=24.84 GB):   5%|▌         | 3/58 [00:02<00:41,  1.32it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=24.84 GB):   7%|▋         | 4/58 [00:02<00:39,  1.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.82 GB):   7%|▋         | 4/58 [00:02<00:39,  1.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.82 GB):   9%|▊         | 5/58 [00:03<00:36,  1.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.90 GB):   9%|▊         | 5/58 [00:03<00:36,  1.44it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.90 GB):  10%|█         | 6/58 [00:04<00:33,  1.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.96 GB):  10%|█         | 6/58 [00:04<00:33,  1.55it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.96 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=26.02 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.65it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=26.02 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=25.43 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.80it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=25.43 GB):  16%|█▌        | 9/58 [00:05<00:25,  1.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.56 GB):  16%|█▌        | 9/58 [00:05<00:25,  1.95it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=25.56 GB):  17%|█▋        | 10/58 [00:05<00:22,  2.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=25.66 GB):  17%|█▋        | 10/58 [00:05<00:22,  2.14it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=25.66 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=25.72 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.29it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=25.72 GB):  21%|██        | 12/58 [00:06<00:18,  2.47it/s]Capturing num tokens (num_tokens=3072 avail_mem=25.77 GB):  21%|██        | 12/58 [00:06<00:18,  2.47it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=25.77 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=25.83 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.67it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=25.83 GB):  24%|██▍       | 14/58 [00:07<00:14,  2.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=25.90 GB):  24%|██▍       | 14/58 [00:07<00:14,  2.95it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=25.90 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=25.96 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.26it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=25.96 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.01 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.01 GB):  29%|██▉       | 17/58 [00:07<00:10,  3.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.38 GB):  29%|██▉       | 17/58 [00:07<00:10,  3.98it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=26.38 GB):  31%|███       | 18/58 [00:07<00:09,  4.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.40 GB):  31%|███       | 18/58 [00:07<00:09,  4.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.40 GB):  33%|███▎      | 19/58 [00:08<00:07,  4.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.42 GB):  33%|███▎      | 19/58 [00:08<00:07,  4.92it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=26.42 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.43 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.43 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.21it/s]Capturing num tokens (num_tokens=960 avail_mem=26.46 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.21it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=26.46 GB):  38%|███▊      | 22/58 [00:08<00:05,  6.84it/s]Capturing num tokens (num_tokens=896 avail_mem=26.45 GB):  38%|███▊      | 22/58 [00:08<00:05,  6.84it/s]Capturing num tokens (num_tokens=896 avail_mem=26.45 GB):  40%|███▉      | 23/58 [00:08<00:04,  7.53it/s]Capturing num tokens (num_tokens=832 avail_mem=26.44 GB):  40%|███▉      | 23/58 [00:08<00:04,  7.53it/s]Capturing num tokens (num_tokens=768 avail_mem=26.44 GB):  40%|███▉      | 23/58 [00:08<00:04,  7.53it/s]

    Capturing num tokens (num_tokens=768 avail_mem=26.44 GB):  43%|████▎     | 25/58 [00:08<00:03,  8.83it/s]Capturing num tokens (num_tokens=704 avail_mem=26.44 GB):  43%|████▎     | 25/58 [00:08<00:03,  8.83it/s]Capturing num tokens (num_tokens=640 avail_mem=26.43 GB):  43%|████▎     | 25/58 [00:08<00:03,  8.83it/s]Capturing num tokens (num_tokens=640 avail_mem=26.43 GB):  47%|████▋     | 27/58 [00:08<00:03, 10.06it/s]Capturing num tokens (num_tokens=576 avail_mem=26.45 GB):  47%|████▋     | 27/58 [00:08<00:03, 10.06it/s]

    Capturing num tokens (num_tokens=512 avail_mem=26.44 GB):  47%|████▋     | 27/58 [00:08<00:03, 10.06it/s]Capturing num tokens (num_tokens=512 avail_mem=26.44 GB):  50%|█████     | 29/58 [00:09<00:02, 11.33it/s]Capturing num tokens (num_tokens=480 avail_mem=26.44 GB):  50%|█████     | 29/58 [00:09<00:02, 11.33it/s]Capturing num tokens (num_tokens=448 avail_mem=26.43 GB):  50%|█████     | 29/58 [00:09<00:02, 11.33it/s]

    Capturing num tokens (num_tokens=448 avail_mem=26.43 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.02it/s]Capturing num tokens (num_tokens=416 avail_mem=26.42 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.02it/s]Capturing num tokens (num_tokens=384 avail_mem=26.42 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.02it/s]Capturing num tokens (num_tokens=384 avail_mem=26.42 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.04it/s]Capturing num tokens (num_tokens=352 avail_mem=26.41 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.04it/s]Capturing num tokens (num_tokens=320 avail_mem=26.40 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.04it/s]

    Capturing num tokens (num_tokens=320 avail_mem=26.40 GB):  60%|██████    | 35/58 [00:09<00:01, 14.49it/s]Capturing num tokens (num_tokens=288 avail_mem=26.40 GB):  60%|██████    | 35/58 [00:09<00:01, 14.49it/s]Capturing num tokens (num_tokens=256 avail_mem=26.40 GB):  60%|██████    | 35/58 [00:09<00:01, 14.49it/s]Capturing num tokens (num_tokens=256 avail_mem=26.40 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.56it/s]Capturing num tokens (num_tokens=240 avail_mem=26.41 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.56it/s]Capturing num tokens (num_tokens=224 avail_mem=26.42 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.56it/s]

    Capturing num tokens (num_tokens=208 avail_mem=26.40 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.56it/s]Capturing num tokens (num_tokens=208 avail_mem=26.40 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.12it/s]Capturing num tokens (num_tokens=192 avail_mem=26.39 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.12it/s]Capturing num tokens (num_tokens=176 avail_mem=26.38 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.12it/s]Capturing num tokens (num_tokens=160 avail_mem=26.37 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.12it/s]Capturing num tokens (num_tokens=160 avail_mem=26.37 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.31it/s]Capturing num tokens (num_tokens=144 avail_mem=26.36 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.31it/s]

    Capturing num tokens (num_tokens=128 avail_mem=26.36 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.31it/s]Capturing num tokens (num_tokens=112 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.31it/s]Capturing num tokens (num_tokens=112 avail_mem=26.35 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.15it/s]Capturing num tokens (num_tokens=96 avail_mem=26.34 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.15it/s] Capturing num tokens (num_tokens=80 avail_mem=26.33 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.15it/s]Capturing num tokens (num_tokens=64 avail_mem=26.32 GB):  79%|███████▉  | 46/58 [00:10<00:00, 19.15it/s]

    Capturing num tokens (num_tokens=64 avail_mem=26.32 GB):  84%|████████▍ | 49/58 [00:10<00:00, 20.06it/s]Capturing num tokens (num_tokens=48 avail_mem=26.31 GB):  84%|████████▍ | 49/58 [00:10<00:00, 20.06it/s]Capturing num tokens (num_tokens=32 avail_mem=26.30 GB):  84%|████████▍ | 49/58 [00:10<00:00, 20.06it/s]Capturing num tokens (num_tokens=28 avail_mem=26.29 GB):  84%|████████▍ | 49/58 [00:10<00:00, 20.06it/s]Capturing num tokens (num_tokens=28 avail_mem=26.29 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.76it/s]Capturing num tokens (num_tokens=24 avail_mem=26.28 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.76it/s]Capturing num tokens (num_tokens=20 avail_mem=26.27 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.76it/s]

    Capturing num tokens (num_tokens=16 avail_mem=26.26 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.76it/s]Capturing num tokens (num_tokens=16 avail_mem=26.26 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.21it/s]Capturing num tokens (num_tokens=12 avail_mem=26.25 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.21it/s]Capturing num tokens (num_tokens=8 avail_mem=26.23 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.21it/s] Capturing num tokens (num_tokens=4 avail_mem=26.22 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.21it/s]Capturing num tokens (num_tokens=4 avail_mem=26.22 GB): 100%|██████████| 58/58 [00:10<00:00, 21.58it/s]Capturing num tokens (num_tokens=4 avail_mem=26.22 GB): 100%|██████████| 58/58 [00:10<00:00,  5.54it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants to know the current date and time along with the weather. I need to figure out how to respond using the functions provided. <br><br>First, I should determine which functions to use. The user mentioned two things: current date/time and weather. So, I'll need both 'get_current_date' and 'get_current_weather' functions.<br><br>Starting with the date and time. The function requires a timezone parameter. Since the user is in New York, I'll use 'America/New_York' as the timezone. I'll structure the function call as <function=date>{{"timezone": "America/New_York"}}</function>.<br><br>Next, for the weather, the function needs a city and state. New York is in NY state, so the city is 'New York' and the state is 'NY'. I'll call 'get_current_weather' with these parameters. The unit isn't specified, but since the user didn't mention it, I'll assume Celsius for temperature and humidity. So, unit is 'celsius'. The function call will be <function=weather>{{"city": "New York", "state": "NY", "unit": "celsius"}}</function>.<br><br>I should make sure each function call is on its own line and properly formatted. Also, I need to include the sources in the response. For the date, the source is the get_current_date function, and for the weather, it's get_current_weather. <br><br>Putting it all together, I'll format each function call separately, ensuring the parameters are correctly included. I'll also add a brief description in English explaining the responses.<br><br><br>content: <function:date>{"timezone": "America/New_York"}<br><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "celsius"}</function><br><br>The current date and time in New York is:<br><function=get_current_date>{"timezone": "America/New_York"}</function><br><br>The current weather in New York is:<br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "celsius"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '14019f3ed2cb489e9c173ec72041a12b', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 3.88453471660614, 'response_sent_to_client_ts': 1778657936.5416715}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': 'fa9b24e5c5a24facacbc851f4eed0c3d', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 26.562035029754043, 'response_sent_to_client_ts': 1778657963.1125371}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'f760d3fbbde04a75b1d1b5ef1b91f4ca', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.10693260934203863, 'response_sent_to_client_ts': 1778657963.257616}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '5c77fb683edc440487ec19a76df8d2db', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.10686742514371872, 'response_sent_to_client_ts': 1778657963.2576246}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '4009050f4bc74f5f8f1e483452059c6f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1068304693326354, 'response_sent_to_client_ts': 1778657963.2576282}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'bbd145d4e92e4a2c84066e37fa4ed7f4', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 18.793702092021704, 'response_sent_to_client_ts': 1778657982.0580354}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user asked for the information and population of the capital of France. Hmm, France\'s capital is Paris. I should confirm that first. Maybe it\'s Paris? Yeah, I\'m pretty sure that\'s correct.\n\nNow, the user wants the population in JSON format. JSON stands for JavaScript Object Notation, so I think they want a structured data format. They\'re probably looking to use this data in an application or maybe just for purposes like a report or a website.\n\nI should look up the population of Paris. I believe it\'s around 2 million, but I\'m not 100% sure. Let me double-check that. Yeah, Paris has a population of approximately 2,170,336 as of 2023. Okay, that\'s clear.\n\nMaybe the user isn\'t just looking for the population number. Sometimes people want more details, like the area or key landmarks to provide a complete picture. So, I\'ll include more fields like administrative region, area, the Eiffel Tower, the Louvre, and maybe the average income to give a well-rounded answer.\n\nWait, the user\'s query was just about the population, but perhaps including additional info isn\'t necessary unless they ask. But since they want a JSON response, I\'ll structure it to have all the relevant details to make it comprehensive yet concise.\n\nI should present this information in a JSON format, which means a key-value structure. I\'ll make sure it\'s valid JSON so it can be easily parsed if needed. Including population, population density, and maybe some key landmarks will add more value without complicating the response.\n\nAlso, considering the user\'s possible use case, they might be writing research papers or creating a database entry. So, providing accurate and detailed data will be helpful for them. I\'ll make sure the JSON is correctly formatted with proper capitalization and brackets.\n\nI think I\'ve covered all the bases. Paris is the capital, I included the population, and more details that could be useful. I should double-check the population number again to ensure it\'s accurate. Yes, 2,170,336 as of the latest data seems correct.\n\nAlright, putting it all together in JSON format and making sure it\'s correctly structured. I\'ll present it neatly so the user can copy it without issues. If they need more information, they can always ask again for a more detailed response.\n</think>\n\nHere is the information and population of the capital of France in JSON format:\n\n```json\n{\n  "capital": {\n    "country": "France",\n    "city": "Paris",\n    "population": 2170336,\n    "area": 1074,\n    "official_name": "Île de Paris",\n    "founded": "1554",\n    "location": " Île-de-France",\n    "(Climate)": " Mediterranean with influence from Atlantic and European Normandy regions",\n    "Key Landmarks": [\n      "Eiffel Tower",\n      "Louvre Museum",\n      "社会主义_left}")\n  }\n}\n```', 'output_ids': [32313, 11, 773, 279, 1196, 4588, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 13, 88190, 11, 9625, 594, 6722, 374, 12095, 13, 358, 1265, 7683, 429, 1156, 13, 10696, 432, 594, 12095, 30, 21607, 11, 358, 2776, 5020, 2704, 429, 594, 4396, 382, 7039, 11, 279, 1196, 6801, 279, 7042, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 773, 358, 1744, 807, 1366, 264, 32930, 821, 3561, 13, 2379, 2299, 4658, 3330, 311, 990, 419, 821, 304, 458, 3766, 476, 7196, 1101, 369, 9895, 1075, 264, 1895, 476, 264, 3910, 382, 40, 1265, 1401, 705, 279, 7042, 315, 12095, 13, 358, 4411, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 2704, 13, 6771, 752, 1990, 15934, 429, 13, 21607, 11, 12095, 702, 264, 7042, 315, 13187, 220, 17, 11, 16, 22, 15, 11, 18, 18, 21, 438, 315, 220, 17, 15, 17, 18, 13, 35439, 11, 429, 594, 2797, 382, 21390, 279, 1196, 4436, 944, 1101, 3330, 369, 279, 7042, 1372, 13, 17688, 1251, 1366, 803, 3565, 11, 1075, 279, 3082, 476, 1376, 59924, 311, 3410, 264, 4583, 6802, 13, 2055, 11, 358, 3278, 2924, 803, 5043, 1075, 22707, 5537, 11, 3082, 11, 279, 468, 3092, 301, 21938, 11, 279, 9729, 48506, 11, 323, 7196, 279, 5461, 7911, 311, 2968, 264, 1632, 64218, 4226, 382, 14190, 11, 279, 1196, 594, 3239, 572, 1101, 911, 279, 7042, 11, 714, 8365, 2670, 5107, 3546, 4436, 944, 5871, 7241, 807, 2548, 13, 1988, 2474, 807, 1366, 264, 4718, 2033, 11, 358, 3278, 5944, 432, 311, 614, 678, 279, 9760, 3565, 311, 1281, 432, 15817, 3602, 63594, 382, 40, 1265, 3042, 419, 1995, 304, 264, 4718, 3561, 11, 892, 3363, 264, 1376, 19083, 5944, 13, 358, 3278, 1281, 2704, 432, 594, 2697, 4718, 773, 432, 646, 387, 6707, 15676, 421, 4362, 13, 55121, 7042, 11, 7042, 17457, 11, 323, 7196, 1045, 1376, 59924, 686, 912, 803, 897, 2041, 68126, 1095, 279, 2033, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 990, 1142, 11, 807, 2578, 387, 4378, 3412, 15689, 476, 6825, 264, 4625, 4343, 13, 2055, 11, 8241, 13382, 323, 11682, 821, 686, 387, 10950, 369, 1105, 13, 358, 3278, 1281, 2704, 279, 4718, 374, 12440, 23126, 448, 6169, 6722, 2022, 323, 38929, 382, 40, 1744, 358, 3003, 9761, 678, 279, 23092, 13, 12095, 374, 279, 6722, 11, 358, 5230, 279, 7042, 11, 323, 803, 3565, 429, 1410, 387, 5390, 13, 358, 1265, 1990, 15934, 279, 7042, 1372, 1549, 311, 5978, 432, 594, 13382, 13, 7414, 11, 220, 17, 11, 16, 22, 15, 11, 18, 18, 21, 438, 315, 279, 5535, 821, 4977, 4396, 382, 71486, 11, 10687, 432, 678, 3786, 304, 4718, 3561, 323, 3259, 2704, 432, 594, 12440, 32930, 13, 358, 3278, 3042, 432, 62166, 773, 279, 1196, 646, 2975, 432, 2041, 4714, 13, 1416, 807, 1184, 803, 1995, 11, 807, 646, 2677, 2548, 1549, 369, 264, 803, 11682, 2033, 624, 151649, 271, 8420, 374, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 341, 262, 330, 11141, 788, 330, 49000, 756, 262, 330, 8926, 788, 330, 59604, 756, 262, 330, 44441, 788, 220, 17, 16, 22, 15, 18, 18, 21, 345, 262, 330, 4798, 788, 220, 16, 15, 22, 19, 345, 262, 330, 32812, 1269, 788, 330, 71807, 273, 409, 12095, 756, 262, 330, 69, 13082, 788, 330, 16, 20, 20, 19, 756, 262, 330, 2527, 788, 330, 59108, 273, 6810, 7276, 34106, 756, 262, 330, 43644, 3426, 84636, 330, 37685, 448, 10173, 504, 22375, 323, 7513, 20317, 13331, 13604, 756, 262, 330, 1592, 11426, 15544, 788, 2278, 414, 330, 36, 3092, 301, 21938, 756, 414, 330, 92806, 48506, 16328, 756, 414, 330, 104407, 9579, 14451, 220, 456, 532, 73594, 151643], 'meta_info': {'id': '1c665fb603d140838fdffd276373f916', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 495, 'completion_tokens': 637, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.755789885297418, 'response_sent_to_client_ts': 1778657987.821708}}</strong>



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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.55s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.53s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.53s/it]


    2026-05-13 07:40:03,581 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 07:40:03] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:42,  4.96s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:42,  4.96s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:02,  2.18s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:02,  2.18s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:10,  1.28s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:10,  1.28s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.62it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:24,  2.13it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:24,  2.13it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.72it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.72it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:15,  3.26it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:15,  3.26it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:13,  3.59it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:13,  3.59it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:12,  3.88it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:12,  3.88it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:11,  4.24it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:11,  4.24it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:11,  4.24it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:07,  6.11it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:07,  6.11it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:07,  6.11it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:05,  8.01it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:05,  8.01it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:05,  8.01it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:04, 10.10it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:04, 10.10it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:04, 10.10it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:04, 10.10it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 13.47it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 13.47it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 13.47it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 13.47it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 13.47it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 18.78it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 18.78it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 18.78it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 18.78it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 18.78it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 18.78it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 25.89it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 25.89it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 25.89it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 25.89it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 25.89it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 25.89it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 25.89it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 33.52it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 33.52it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 33.52it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 33.52it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 33.52it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 33.52it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 33.52it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 33.52it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:07<00:00, 42.16it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:07<00:00, 42.16it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:07<00:00, 42.16it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:07<00:00, 42.16it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:07<00:00, 42.16it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:07<00:00, 42.16it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:07<00:00, 42.16it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:07<00:00, 42.16it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 48.51it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 48.51it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 48.51it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 48.51it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 48.51it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:08<00:00, 48.51it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:08<00:00, 48.51it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:08<00:00, 48.51it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:08<00:00, 48.51it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:08<00:00, 48.51it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00, 58.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=42.59 GB):   2%|▏         | 1/58 [00:00<00:16,  3.45it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.56 GB):   2%|▏         | 1/58 [00:00<00:16,  3.45it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.56 GB):   3%|▎         | 2/58 [00:00<00:15,  3.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.55 GB):   3%|▎         | 2/58 [00:00<00:15,  3.60it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.55 GB):   5%|▌         | 3/58 [00:00<00:14,  3.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.55 GB):   5%|▌         | 3/58 [00:00<00:14,  3.84it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.55 GB):   7%|▋         | 4/58 [00:01<00:13,  4.13it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.55 GB):   7%|▋         | 4/58 [00:01<00:13,  4.13it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.55 GB):   9%|▊         | 5/58 [00:01<00:12,  4.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.55 GB):   9%|▊         | 5/58 [00:01<00:12,  4.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.55 GB):  10%|█         | 6/58 [00:01<00:10,  4.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.55 GB):  10%|█         | 6/58 [00:01<00:10,  4.79it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.55 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.55 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.55 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.55 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.33it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.55 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.55 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.55 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.55 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.29it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.55 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.83it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.55 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.83it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.55 GB):  21%|██        | 12/58 [00:02<00:06,  7.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.55 GB):  21%|██        | 12/58 [00:02<00:06,  7.21it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=42.55 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.55 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.55 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.63it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.52 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.63it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=42.52 GB):  26%|██▌       | 15/58 [00:02<00:05,  7.73it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.52 GB):  26%|██▌       | 15/58 [00:02<00:05,  7.73it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.52 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.68it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.52 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.68it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=42.52 GB):  29%|██▉       | 17/58 [00:02<00:05,  7.85it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.52 GB):  29%|██▉       | 17/58 [00:02<00:05,  7.85it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.52 GB):  31%|███       | 18/58 [00:02<00:05,  7.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.51 GB):  31%|███       | 18/58 [00:02<00:05,  7.98it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=42.51 GB):  33%|███▎      | 19/58 [00:03<00:04,  8.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.51 GB):  33%|███▎      | 19/58 [00:03<00:04,  8.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.50 GB):  33%|███▎      | 19/58 [00:03<00:04,  8.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.50 GB):  36%|███▌      | 21/58 [00:03<00:03,  9.78it/s]Capturing num tokens (num_tokens=960 avail_mem=42.50 GB):  36%|███▌      | 21/58 [00:03<00:03,  9.78it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=42.49 GB):  36%|███▌      | 21/58 [00:03<00:03,  9.78it/s]Capturing num tokens (num_tokens=896 avail_mem=42.49 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.13it/s]Capturing num tokens (num_tokens=832 avail_mem=42.49 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.13it/s]Capturing num tokens (num_tokens=768 avail_mem=42.48 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.13it/s]Capturing num tokens (num_tokens=704 avail_mem=42.48 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.13it/s]Capturing num tokens (num_tokens=704 avail_mem=42.48 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.22it/s]Capturing num tokens (num_tokens=640 avail_mem=42.48 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.22it/s]

    Capturing num tokens (num_tokens=576 avail_mem=61.59 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.22it/s]Capturing num tokens (num_tokens=576 avail_mem=61.59 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.88it/s]Capturing num tokens (num_tokens=512 avail_mem=61.58 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.88it/s]Capturing num tokens (num_tokens=480 avail_mem=61.58 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.88it/s]Capturing num tokens (num_tokens=448 avail_mem=61.58 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.88it/s]Capturing num tokens (num_tokens=416 avail_mem=61.57 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.88it/s]Capturing num tokens (num_tokens=416 avail_mem=61.57 GB):  55%|█████▌    | 32/58 [00:03<00:01, 19.82it/s]Capturing num tokens (num_tokens=384 avail_mem=61.57 GB):  55%|█████▌    | 32/58 [00:03<00:01, 19.82it/s]Capturing num tokens (num_tokens=352 avail_mem=61.56 GB):  55%|█████▌    | 32/58 [00:03<00:01, 19.82it/s]

    Capturing num tokens (num_tokens=320 avail_mem=61.56 GB):  55%|█████▌    | 32/58 [00:03<00:01, 19.82it/s]Capturing num tokens (num_tokens=288 avail_mem=61.56 GB):  55%|█████▌    | 32/58 [00:03<00:01, 19.82it/s]Capturing num tokens (num_tokens=288 avail_mem=61.56 GB):  62%|██████▏   | 36/58 [00:03<00:00, 24.16it/s]Capturing num tokens (num_tokens=256 avail_mem=61.56 GB):  62%|██████▏   | 36/58 [00:03<00:00, 24.16it/s]Capturing num tokens (num_tokens=240 avail_mem=61.55 GB):  62%|██████▏   | 36/58 [00:03<00:00, 24.16it/s]Capturing num tokens (num_tokens=224 avail_mem=61.55 GB):  62%|██████▏   | 36/58 [00:03<00:00, 24.16it/s]Capturing num tokens (num_tokens=208 avail_mem=61.55 GB):  62%|██████▏   | 36/58 [00:03<00:00, 24.16it/s]Capturing num tokens (num_tokens=208 avail_mem=61.55 GB):  69%|██████▉   | 40/58 [00:03<00:00, 28.00it/s]Capturing num tokens (num_tokens=192 avail_mem=61.54 GB):  69%|██████▉   | 40/58 [00:03<00:00, 28.00it/s]Capturing num tokens (num_tokens=176 avail_mem=61.54 GB):  69%|██████▉   | 40/58 [00:03<00:00, 28.00it/s]

    Capturing num tokens (num_tokens=160 avail_mem=61.54 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.00it/s]Capturing num tokens (num_tokens=144 avail_mem=61.53 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.00it/s]Capturing num tokens (num_tokens=144 avail_mem=61.53 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.96it/s]Capturing num tokens (num_tokens=128 avail_mem=61.53 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.96it/s]Capturing num tokens (num_tokens=112 avail_mem=61.53 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.96it/s]Capturing num tokens (num_tokens=96 avail_mem=61.53 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.96it/s] Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.96it/s]Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  83%|████████▎ | 48/58 [00:04<00:00, 33.39it/s]Capturing num tokens (num_tokens=64 avail_mem=61.52 GB):  83%|████████▎ | 48/58 [00:04<00:00, 33.39it/s]Capturing num tokens (num_tokens=48 avail_mem=61.51 GB):  83%|████████▎ | 48/58 [00:04<00:00, 33.39it/s]

    Capturing num tokens (num_tokens=32 avail_mem=61.51 GB):  83%|████████▎ | 48/58 [00:04<00:00, 33.39it/s]Capturing num tokens (num_tokens=28 avail_mem=61.51 GB):  83%|████████▎ | 48/58 [00:04<00:00, 33.39it/s]Capturing num tokens (num_tokens=24 avail_mem=61.51 GB):  83%|████████▎ | 48/58 [00:04<00:00, 33.39it/s]Capturing num tokens (num_tokens=24 avail_mem=61.51 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.58it/s]Capturing num tokens (num_tokens=20 avail_mem=61.50 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.58it/s]Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.58it/s]Capturing num tokens (num_tokens=12 avail_mem=61.50 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.58it/s]Capturing num tokens (num_tokens=8 avail_mem=61.49 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.58it/s] Capturing num tokens (num_tokens=4 avail_mem=61.49 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.58it/s]Capturing num tokens (num_tokens=4 avail_mem=61.49 GB): 100%|██████████| 58/58 [00:04<00:00, 37.10it/s]Capturing num tokens (num_tokens=4 avail_mem=61.49 GB): 100%|██████████| 58/58 [00:04<00:00, 13.13it/s]


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
    
    Generated text: Alright, the user asked for the information and population of the capital of France in JSON format. I know the capital is Paris, so I need to provide its population. I remember the approximate population is around 2 million, but it's good to double-check for accuracy.
    
    I should structure this in JSON, making sure to include the country as well. Maybe include fields like name, population, and country for clarity. Let me format it properly without any markdown. 
    
    I also want to make sure the population figure is up-to-date. I think it's roughly 2,178,137 as of 2023. It's better to mention that the number can vary slightly depending on the source and year. That way, the user knows it's an estimate and might change over time.
    
    Putting it all together, I'll create a JSON object with the necessary details. I hope this meets the user's needs and provides accurate information.
    </think>
    
    ```json
    {
      "name": "Paris",
      "population": 2178137,
      "country": "France"
    }
    ```



```python
llm.shutdown()
```
