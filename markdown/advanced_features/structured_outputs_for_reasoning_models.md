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

    [2026-03-03 20:02:28] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-03 20:02:28] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-03 20:02:28] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-03 20:02:32] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 20:02:32] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 20:02:32] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:41: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-03 20:02:34] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.
    [2026-03-03 20:02:34] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 20:02:35] INFO utils.py:456: Successfully reserved port 32891 on host '0.0.0.0'


    [2026-03-03 20:02:39] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 20:02:39] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 20:02:39] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-03 20:02:39] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 20:02:39] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 20:02:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-03 20:02:43] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-03 20:02:43] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-03 20:02:43] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.05s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.42s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.37s/it]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=61.44 GB):   0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=61.44 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.60it/s]Capturing batches (bs=2 avail_mem=61.39 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.60it/s]Capturing batches (bs=1 avail_mem=61.39 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.60it/s]

    Capturing batches (bs=1 avail_mem=61.39 GB): 100%|██████████| 3/3 [00:00<00:00, 13.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:19,  2.45s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:19,  2.45s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:10,  1.25s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:10,  1.25s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:42,  1.29it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:42,  1.29it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:29,  1.85it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:29,  1.85it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:21,  2.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:21,  2.48it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:16,  3.12it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:16,  3.12it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:13,  3.87it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:13,  3.87it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:10,  4.64it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:10,  4.64it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:09,  5.43it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:09,  5.43it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:07,  6.24it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:07,  6.24it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:07,  6.61it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:07,  6.61it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:07,  6.61it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.49it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.49it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.49it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  9.98it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  9.98it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  9.98it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 12.06it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 12.06it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 12.06it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 12.06it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:02, 15.42it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:02, 15.42it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:02, 15.42it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:02, 15.42it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:02, 15.42it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:04<00:01, 20.38it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:04<00:01, 20.38it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:04<00:01, 20.38it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:04<00:01, 20.38it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:04<00:01, 20.38it/s]

    Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:04<00:01, 20.38it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:04<00:01, 20.38it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:00, 29.34it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:00, 29.34it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:00, 29.34it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:00, 29.34it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 36.89it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 36.89it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 36.89it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 36.89it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 36.89it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 36.89it/s]

    Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 36.89it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:00, 36.89it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:00, 36.89it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 46.52it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 46.52it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 46.52it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 46.52it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 46.52it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 46.52it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 46.52it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 46.52it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 46.52it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 54.48it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 54.48it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 54.48it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 54.48it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 54.48it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 54.48it/s] 

    Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 54.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=26.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=26.11 GB):   2%|▏         | 1/58 [00:00<00:19,  2.87it/s]Capturing num tokens (num_tokens=7680 avail_mem=26.07 GB):   2%|▏         | 1/58 [00:00<00:19,  2.87it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=26.07 GB):   3%|▎         | 2/58 [00:00<00:16,  3.33it/s]Capturing num tokens (num_tokens=7168 avail_mem=26.07 GB):   3%|▎         | 2/58 [00:00<00:16,  3.33it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=26.07 GB):   5%|▌         | 3/58 [00:00<00:14,  3.67it/s]Capturing num tokens (num_tokens=6656 avail_mem=26.08 GB):   5%|▌         | 3/58 [00:00<00:14,  3.67it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=26.08 GB):   7%|▋         | 4/58 [00:01<00:15,  3.52it/s]Capturing num tokens (num_tokens=6144 avail_mem=26.08 GB):   7%|▋         | 4/58 [00:01<00:15,  3.52it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=26.08 GB):   9%|▊         | 5/58 [00:01<00:13,  3.93it/s]Capturing num tokens (num_tokens=5632 avail_mem=26.08 GB):   9%|▊         | 5/58 [00:01<00:13,  3.93it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=26.08 GB):  10%|█         | 6/58 [00:01<00:14,  3.47it/s]Capturing num tokens (num_tokens=5120 avail_mem=24.93 GB):  10%|█         | 6/58 [00:01<00:14,  3.47it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=24.93 GB):  12%|█▏        | 7/58 [00:02<00:16,  3.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=24.93 GB):  12%|█▏        | 7/58 [00:02<00:16,  3.13it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=24.93 GB):  14%|█▍        | 8/58 [00:02<00:15,  3.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=24.94 GB):  14%|█▍        | 8/58 [00:02<00:15,  3.13it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=24.94 GB):  16%|█▌        | 9/58 [00:02<00:15,  3.16it/s]Capturing num tokens (num_tokens=3840 avail_mem=24.94 GB):  16%|█▌        | 9/58 [00:02<00:15,  3.16it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=24.94 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=24.90 GB):  17%|█▋        | 10/58 [00:03<00:15,  3.02it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=24.90 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=24.90 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.18it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=24.90 GB):  21%|██        | 12/58 [00:03<00:13,  3.37it/s]Capturing num tokens (num_tokens=3072 avail_mem=24.90 GB):  21%|██        | 12/58 [00:03<00:13,  3.37it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=24.90 GB):  22%|██▏       | 13/58 [00:03<00:12,  3.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=24.90 GB):  22%|██▏       | 13/58 [00:03<00:12,  3.54it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=24.90 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=24.90 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.79it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=24.90 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.05it/s]Capturing num tokens (num_tokens=2304 avail_mem=24.90 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.05it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=24.90 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.90 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.90 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=24.90 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.54it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=24.90 GB):  31%|███       | 18/58 [00:04<00:08,  4.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=24.90 GB):  31%|███       | 18/58 [00:04<00:08,  4.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=24.90 GB):  33%|███▎      | 19/58 [00:05<00:07,  5.18it/s]Capturing num tokens (num_tokens=1280 avail_mem=24.90 GB):  33%|███▎      | 19/58 [00:05<00:07,  5.18it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=24.90 GB):  34%|███▍      | 20/58 [00:05<00:06,  5.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.90 GB):  34%|███▍      | 20/58 [00:05<00:06,  5.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.90 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.28it/s]Capturing num tokens (num_tokens=960 avail_mem=24.90 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.28it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=24.90 GB):  38%|███▊      | 22/58 [00:05<00:05,  6.68it/s]Capturing num tokens (num_tokens=896 avail_mem=24.89 GB):  38%|███▊      | 22/58 [00:05<00:05,  6.68it/s]Capturing num tokens (num_tokens=832 avail_mem=24.89 GB):  38%|███▊      | 22/58 [00:05<00:05,  6.68it/s]Capturing num tokens (num_tokens=832 avail_mem=24.89 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.18it/s]Capturing num tokens (num_tokens=768 avail_mem=24.89 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.18it/s]

    Capturing num tokens (num_tokens=704 avail_mem=24.88 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.18it/s]Capturing num tokens (num_tokens=704 avail_mem=24.88 GB):  45%|████▍     | 26/58 [00:05<00:03,  9.76it/s]Capturing num tokens (num_tokens=640 avail_mem=24.88 GB):  45%|████▍     | 26/58 [00:05<00:03,  9.76it/s]Capturing num tokens (num_tokens=576 avail_mem=24.87 GB):  45%|████▍     | 26/58 [00:05<00:03,  9.76it/s]Capturing num tokens (num_tokens=576 avail_mem=24.87 GB):  48%|████▊     | 28/58 [00:05<00:02, 11.48it/s]Capturing num tokens (num_tokens=512 avail_mem=24.87 GB):  48%|████▊     | 28/58 [00:05<00:02, 11.48it/s]

    Capturing num tokens (num_tokens=480 avail_mem=24.87 GB):  48%|████▊     | 28/58 [00:05<00:02, 11.48it/s]Capturing num tokens (num_tokens=448 avail_mem=24.86 GB):  48%|████▊     | 28/58 [00:05<00:02, 11.48it/s]Capturing num tokens (num_tokens=448 avail_mem=24.86 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.07it/s]Capturing num tokens (num_tokens=416 avail_mem=24.85 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.07it/s]Capturing num tokens (num_tokens=384 avail_mem=24.85 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.07it/s]Capturing num tokens (num_tokens=352 avail_mem=24.85 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.07it/s]

    Capturing num tokens (num_tokens=352 avail_mem=24.85 GB):  59%|█████▊    | 34/58 [00:06<00:01, 16.44it/s]Capturing num tokens (num_tokens=320 avail_mem=24.84 GB):  59%|█████▊    | 34/58 [00:06<00:01, 16.44it/s]Capturing num tokens (num_tokens=288 avail_mem=24.84 GB):  59%|█████▊    | 34/58 [00:06<00:01, 16.44it/s]Capturing num tokens (num_tokens=288 avail_mem=24.84 GB):  62%|██████▏   | 36/58 [00:06<00:01, 17.09it/s]Capturing num tokens (num_tokens=256 avail_mem=24.83 GB):  62%|██████▏   | 36/58 [00:06<00:01, 17.09it/s]Capturing num tokens (num_tokens=240 avail_mem=24.83 GB):  62%|██████▏   | 36/58 [00:06<00:01, 17.09it/s]Capturing num tokens (num_tokens=224 avail_mem=24.83 GB):  62%|██████▏   | 36/58 [00:06<00:01, 17.09it/s]Capturing num tokens (num_tokens=208 avail_mem=24.82 GB):  62%|██████▏   | 36/58 [00:06<00:01, 17.09it/s]

    Capturing num tokens (num_tokens=192 avail_mem=24.82 GB):  62%|██████▏   | 36/58 [00:06<00:01, 17.09it/s]Capturing num tokens (num_tokens=192 avail_mem=24.82 GB):  71%|███████   | 41/58 [00:06<00:00, 23.89it/s]Capturing num tokens (num_tokens=176 avail_mem=24.82 GB):  71%|███████   | 41/58 [00:06<00:00, 23.89it/s]Capturing num tokens (num_tokens=160 avail_mem=24.81 GB):  71%|███████   | 41/58 [00:06<00:00, 23.89it/s]Capturing num tokens (num_tokens=144 avail_mem=24.81 GB):  71%|███████   | 41/58 [00:06<00:00, 23.89it/s]Capturing num tokens (num_tokens=128 avail_mem=24.82 GB):  71%|███████   | 41/58 [00:06<00:00, 23.89it/s]Capturing num tokens (num_tokens=112 avail_mem=24.82 GB):  71%|███████   | 41/58 [00:06<00:00, 23.89it/s]Capturing num tokens (num_tokens=112 avail_mem=24.82 GB):  79%|███████▉  | 46/58 [00:06<00:00, 29.07it/s]Capturing num tokens (num_tokens=96 avail_mem=24.81 GB):  79%|███████▉  | 46/58 [00:06<00:00, 29.07it/s] Capturing num tokens (num_tokens=80 avail_mem=24.79 GB):  79%|███████▉  | 46/58 [00:06<00:00, 29.07it/s]Capturing num tokens (num_tokens=64 avail_mem=24.78 GB):  79%|███████▉  | 46/58 [00:06<00:00, 29.07it/s]

    Capturing num tokens (num_tokens=48 avail_mem=23.59 GB):  79%|███████▉  | 46/58 [00:06<00:00, 29.07it/s]Capturing num tokens (num_tokens=48 avail_mem=23.59 GB):  86%|████████▌ | 50/58 [00:06<00:00, 25.25it/s]Capturing num tokens (num_tokens=32 avail_mem=23.59 GB):  86%|████████▌ | 50/58 [00:06<00:00, 25.25it/s]

    Capturing num tokens (num_tokens=28 avail_mem=23.58 GB):  86%|████████▌ | 50/58 [00:06<00:00, 25.25it/s]Capturing num tokens (num_tokens=24 avail_mem=24.74 GB):  86%|████████▌ | 50/58 [00:06<00:00, 25.25it/s]Capturing num tokens (num_tokens=24 avail_mem=24.74 GB):  91%|█████████▏| 53/58 [00:06<00:00, 19.89it/s]Capturing num tokens (num_tokens=20 avail_mem=24.74 GB):  91%|█████████▏| 53/58 [00:06<00:00, 19.89it/s]

    Capturing num tokens (num_tokens=16 avail_mem=23.74 GB):  91%|█████████▏| 53/58 [00:07<00:00, 19.89it/s]Capturing num tokens (num_tokens=12 avail_mem=23.74 GB):  91%|█████████▏| 53/58 [00:07<00:00, 19.89it/s]

    Capturing num tokens (num_tokens=12 avail_mem=23.74 GB):  97%|█████████▋| 56/58 [00:07<00:00, 12.54it/s]Capturing num tokens (num_tokens=8 avail_mem=24.72 GB):  97%|█████████▋| 56/58 [00:07<00:00, 12.54it/s] Capturing num tokens (num_tokens=4 avail_mem=23.80 GB):  97%|█████████▋| 56/58 [00:07<00:00, 12.54it/s]

    Capturing num tokens (num_tokens=4 avail_mem=23.80 GB): 100%|██████████| 58/58 [00:07<00:00, 10.84it/s]Capturing num tokens (num_tokens=4 avail_mem=23.80 GB): 100%|██████████| 58/58 [00:07<00:00,  7.53it/s]



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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably double-check that. <br><br>Wait, I think I heard somewhere that Paris has a population over 21 million. Maybe 21.6 million? I'm not sure if that's accurate. I should look up the latest data to confirm. Also, I wonder if the population includes just the city proper or the entire metropolitan area. I think sometimes population counts include the broader area, so maybe that's why the number is higher. <br><br>I should make sure to present this information in JSON format as the user requested. So, the key would be "capital" with the value "Paris" and another key "population" with the number. I need to decide whether to include the metropolitan area or just the city limits. Since the user didn't specify, I'll go with the metropolitan area population, which I think is around 21.6 million. <br><br>I should also consider the source of this information to ensure accuracy. Maybe the World Bank or recent census data. I recall that the population has been growing steadily, so 21.6 million seems reasonable. I don't think it's too high or too low. <br><br>So, putting it all together, the JSON should have two keys: "capital" and "population". The value for "capital" is "Paris", and "population" is 21600000. I should format it correctly with proper syntax, using quotes and commas where necessary. <br><br>I think that's all. I just need to make sure the information is accurate and present it in the required format.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21600000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think the population has been growing over the years. I recall reading somewhere that it's over 21 million now. Maybe around 21.6 million? I'm not sure if that's the exact number or just an estimate. I should look it up to confirm. Also, I should make sure that Paris is indeed the capital and not another city like Lyon or Marseille.<br><br>I'm pretty confident that Paris is the capital, but just to be thorough, I'll double-check. The official capital of France is Paris, so that's correct. Now, for the population, I think it's a bit tricky because population numbers can change yearly. I believe the latest data from 2021 puts the population at approximately 21.6 million. But I'm not 100% sure if that's accurate or if it's an average. Maybe it's around 21.5 million? I should present it as an estimate since exact numbers can vary.<br><br>So, putting it all together, the JSON should have the key "capital" with the value "Paris" and another key "population" with the estimated number. I'll format it properly, making sure the syntax is correct with commas and quotes. I should also mention that the population is an estimate based on the latest available data to avoid any confusion.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21600000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population. I think it's a big city, maybe around 3 million? I remember hearing that Paris is one of the most populous cities in Europe, but I'm not certain about the exact number. Maybe I should check some sources or think about recent growth. I think the population has been increasing over the years, so perhaps it's now over 3.5 million? I'm a bit confused because sometimes I hear different numbers, so I should make sure. Maybe I can recall that Paris has a metropolitan area that's much larger, but the city proper is around 3.5 million. I think I'll go with that for now.<br><br><br>content: Paris is the capital of France</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light." People go there for museums, landmarks like the Eiffel Tower, and it's a cultural hub. But is it the capital?<br><br>Wait, I think the capital is the official seat of government, right? So maybe Paris is both the capital and the most famous city. But I'm not entirely certain. I recall that some countries have their capital in a different city than their main tourist attraction. For example, I think Brazil's capital is not Rio de Janeiro, which is more famous. So maybe France is like that too.<br><br>Let me try to remember any specific information. I think the French government declares Paris as the capital. I've heard that before. Also, I remember learning that the Eiffel Tower is in Paris, which is a symbol of the city, but not necessarily the government building. The government buildings are probably in another part of the city or in a different city altogether.<br><br>Wait, no, I think the government buildings are in Paris. Maybe the Palace of Consultation or something like that. I'm not sure about the exact name, but I'm pretty sure Paris is where the government offices are located. So that would make Paris the capital.<br><br>I also think that sometimes people confuse the capital with the administrative center, but in France, I believe the administrative center is in Toulouse, but the capital is still Paris. So even though Toulouse is the main hub for government agencies, Paris is where the president and prime minister are located.<br><br>So putting it all together, Paris is the capital of France because it's the seat of government, even though it's also the most well-known city in the country. I think that's correct, but I'm a bit fuzzy on the exact administrative details. Maybe I should double-check, but from what I remember, Paris is definitely the capital.<br><br><br>content: Paris</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants to know the current date and time, as well as the weather. I need to figure out how to get this information using the functions provided.<br><br>First, I should use the get_current_date function because it gives both the date and time for a specified timezone. The user is in New York, so I'll set the timezone parameter to 'America/New_York'. That should give me the accurate current date and time.<br><br>Next, for the weather, I'll use the get_current_weather function. The city is New York, and since the user didn't specify, I'll default to 'US' as the state. I'll choose Fahrenheit as the unit because that's a common preference, but the user could have another unit in mind. I should include this in the parameters.<br><br>I need to make sure each function call is separate and follows the required format. So, I'll call get_current_date first with the timezone parameter. Then, I'll call get_current_weather with city, state, and unit parameters.<br><br>I should also remember to include the sources in the response. For the date, it's from the get_current_date function, and for the weather, it's from get_current_weather. That way, the user knows where the information came from.<br><br>Putting it all together, I'll structure the response with each function call on its own line, making sure to format them correctly with the start and end tags. I'll also add a brief explanation to tie everything together.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "US", "unit": "fahrenheit"}</function>  <br><br>The current date and time in New York can be found using the get_current_date function with the timezone parameter set to 'America/New_York'. The current weather in New York can be found using the get_current_weather function with the city parameter set to 'New York', the state parameter set to 'US', and the unit parameter set to 'fahrenheit'.</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s straightforward. \n\nNext, I need to find the population. I remember that Paris is a major city, so its population is quite large. I think it\'s over 3 million, but I\'m not exactly sure of the exact number. Maybe I should double-check that. \n\nWait, I recall that the population figure can vary depending on the source and the year. The user didn\'t specify a particular year, so I should probably go with the most recent estimate. I believe the population is around 3,500,000 as of 2023. \n\nNow, I need to structure this information into a JSON format. JSON typically uses key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" since the user mentioned France. \n\nI should make sure the keys are in English to keep it clear. The city is Paris, the population is 3,500,000, and the country is France. I\'ll format this into a JSON object. \n\nI also need to present this in a way that\'s easy to read, so I\'ll use proper syntax with quotation marks and commas in the right places. No trailing commas to avoid errors. \n\nPutting it all together, the JSON should look something like this: a dictionary with the keys and the corresponding values. I\'ll make sure to test it to ensure it\'s valid, but since I\'m just writing it out, I\'ll assume it\'s correct based on my knowledge. \n\nI think that\'s all. The user just needs the information in JSON, so this should satisfy their request.\n</think>{\n\n"name": "Paris",\n"population": 3500000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4710, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 11, 773, 1181, 7042, 374, 5008, 3460, 13, 358, 1744, 432, 594, 916, 220, 18, 3526, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 10696, 358, 1265, 1990, 15934, 429, 13, 4710, 14190, 11, 358, 19091, 429, 279, 7042, 7071, 646, 13289, 11649, 389, 279, 2530, 323, 279, 1042, 13, 576, 1196, 3207, 944, 13837, 264, 3953, 1042, 11, 773, 358, 1265, 4658, 728, 448, 279, 1429, 3213, 16045, 13, 358, 4411, 279, 7042, 374, 2163, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 4710, 7039, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 11136, 5711, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 2474, 279, 1196, 9733, 9625, 13, 4710, 40, 1265, 1281, 2704, 279, 6894, 525, 304, 6364, 311, 2506, 432, 2797, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 13, 358, 3278, 3561, 419, 1119, 264, 4718, 1633, 13, 4710, 40, 1083, 1184, 311, 3042, 419, 304, 264, 1616, 429, 594, 4135, 311, 1349, 11, 773, 358, 3278, 990, 6169, 19482, 448, 54231, 15423, 323, 76602, 304, 279, 1290, 7482, 13, 2308, 27748, 76602, 311, 5648, 5975, 13, 4710, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 264, 10997, 448, 279, 6894, 323, 279, 12159, 2750, 13, 358, 3278, 1281, 2704, 311, 1273, 432, 311, 5978, 432, 594, 2697, 11, 714, 2474, 358, 2776, 1101, 4378, 432, 700, 11, 358, 3278, 9658, 432, 594, 4396, 3118, 389, 847, 6540, 13, 4710, 40, 1744, 429, 594, 678, 13, 576, 1196, 1101, 3880, 279, 1995, 304, 4718, 11, 773, 419, 1265, 26553, 862, 1681, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 18, 20, 15, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'ce7491247a564726b46f4c87c5323191', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 405, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.533169908914715, 'response_sent_to_client_ts': 1772568202.4519923}}



<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that's straightforward. <br><br>Next, I need to find the population. I remember that Paris is a major city, so its population is quite large. I think it's over 3 million, but I'm not exactly sure of the exact number. Maybe I should double-check that. <br><br>Wait, I recall that the population figure can vary depending on the source and the year. The user didn't specify a particular year, so I should probably go with the most recent estimate. I believe the population is around 3,500,000 as of 2023. <br><br>Now, I need to structure this information into a JSON format. JSON typically uses key-value pairs, so I'll create an object with keys like "city", "population", and maybe "country" since the user mentioned France. <br><br>I should make sure the keys are in English to keep it clear. The city is Paris, the population is 3,500,000, and the country is France. I'll format this into a JSON object. <br><br>I also need to present this in a way that's easy to read, so I'll use proper syntax with quotation marks and commas in the right places. No trailing commas to avoid errors. <br><br>Putting it all together, the JSON should look something like this: a dictionary with the keys and the corresponding values. I'll make sure to test it to ensure it's valid, but since I'm just writing it out, I'll assume it's correct based on my knowledge. <br><br>I think that's all. The user just needs the information in JSON, so this should satisfy their request.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 3500000}</strong>


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants both the general information and the population. So, I\'ll create an object with a "name" field for the capital, a "general_information" section that includes the administrative center, area, and government department, and a "population" section that includes the current population and a note about the data source.\n\nI should also add a "source" field to indicate where the population data comes from, which is the 2020 census. This makes the information more transparent and trustworthy.\n\nPutting it all together, I\'ll format the JSON with proper syntax, using double quotes for strings and ensuring that the keys are clear and descriptive. I\'ll make sure there are no typos and that the JSON is valid.\n\nFinally, I\'ll present the JSON in a code block so the user can easily copy and use it. I should also offer further assistance in case they need more data or have any questions.\n</think>{\n  "name": "Paris",\n  "population": 2174300\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 2176, 279, 4586, 1995, 323, 279, 7042, 13, 2055, 11, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 6722, 11, 264, 330, 24595, 35212, 1, 3772, 429, 5646, 279, 22707, 4126, 11, 3082, 11, 323, 3033, 9292, 11, 323, 264, 330, 44441, 1, 3772, 429, 5646, 279, 1482, 7042, 323, 264, 5185, 911, 279, 821, 2530, 382, 40, 1265, 1083, 912, 264, 330, 2427, 1, 2070, 311, 13216, 1380, 279, 7042, 821, 4041, 504, 11, 892, 374, 279, 220, 17, 15, 17, 15, 43602, 13, 1096, 3643, 279, 1995, 803, 17821, 323, 55942, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3561, 279, 4718, 448, 6169, 19482, 11, 1667, 1990, 17194, 369, 9069, 323, 22573, 429, 279, 6894, 525, 2797, 323, 52844, 13, 358, 3278, 1281, 2704, 1052, 525, 902, 13580, 966, 323, 429, 279, 4718, 374, 2697, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 304, 264, 2038, 2504, 773, 279, 1196, 646, 6707, 2975, 323, 990, 432, 13, 358, 1265, 1083, 3010, 4623, 12994, 304, 1142, 807, 1184, 803, 821, 476, 614, 894, 4755, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 198, 92, 151643], 'meta_info': {'id': '5f8167dc57c547e599c64ec91ded1f6f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 386, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.333856603130698, 'response_sent_to_client_ts': 1772568204.7957687}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '6c4988e51f544f349a48fe966fe971ca', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.14778147684410214, 'response_sent_to_client_ts': 1772568204.983755}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '2b38d86ac7e94bd490d8cbb5780bc45a', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.1461044796742499, 'response_sent_to_client_ts': 1772568204.983773}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'ab61dfff5e3047da98bab7ebfce5e206', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.14605701295658946, 'response_sent_to_client_ts': 1772568204.9837785}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '33ebfb2279c146e99851131dc2c6c7d6', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 15.033782253973186, 'response_sent_to_client_ts': 1772568220.025572}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so I need to figure out the population of the capital of France, which is Paris. I remember that Paris is the main city in France and one of the largest in Europe. But I\'m not exactly sure about the exact number of people living there. I think it\'s somewhere around 3 million. Maybe a bit more? I\'ve heard estimates that go up to 3.5 million or even 4 million. Wait, I think 3.5 million is a more common figure I\'ve come across before.\n\nLet me think about other major cities to compare. Tokyo, for instance, I believe has about 3.5 million people. Rome also has a similar population. So, considering those, Paris being a major cultural and economic hub, it\'s reasonable that it\'s around that number.\n\nI don\'t recall ever seeing an official figure that\'s much different from that. Maybe in recent years, the population has grown a bit, but I\'m not sure if it\'s significantly more than 3.5 million. I should also remember that population figures can vary depending on how they are measured and the time of year. For example, sometimes people are counted differently in holidays or major events, but generally, it\'s around that range.\n\nI think I\'m pretty confident that Paris\' population is approximately 3.5 million. So, putting that together, I can format it into JSON as the problem requested. I should make sure to include the correct city name and population.\n</think>\n\nThe population of the capital of France, Paris, is approximately 3.5 million.\n\n```json\n{\n  "capital": "Paris",\n  "population": 3500000\n}\n```', 'output_ids': [32313, 11, 773, 358, 1184, 311, 7071, 700, 279, 7042, 315, 279, 6722, 315, 9625, 11, 892, 374, 12095, 13, 358, 6099, 429, 12095, 374, 279, 1887, 3283, 304, 9625, 323, 825, 315, 279, 7772, 304, 4505, 13, 1988, 358, 2776, 537, 6896, 2704, 911, 279, 4734, 1372, 315, 1251, 5382, 1052, 13, 358, 1744, 432, 594, 14696, 2163, 220, 18, 3526, 13, 10696, 264, 2699, 803, 30, 358, 3003, 6617, 17530, 429, 728, 705, 311, 220, 18, 13, 20, 3526, 476, 1496, 220, 19, 3526, 13, 13824, 11, 358, 1744, 220, 18, 13, 20, 3526, 374, 264, 803, 4185, 7071, 358, 3003, 2525, 3941, 1573, 382, 10061, 752, 1744, 911, 1008, 3598, 9720, 311, 9429, 13, 26194, 11, 369, 2867, 11, 358, 4411, 702, 911, 220, 18, 13, 20, 3526, 1251, 13, 21718, 1083, 702, 264, 4428, 7042, 13, 2055, 11, 12831, 1846, 11, 12095, 1660, 264, 3598, 12752, 323, 6955, 18719, 11, 432, 594, 13276, 429, 432, 594, 2163, 429, 1372, 382, 40, 1513, 944, 19091, 3512, 9120, 458, 3946, 7071, 429, 594, 1753, 2155, 504, 429, 13, 10696, 304, 3213, 1635, 11, 279, 7042, 702, 14700, 264, 2699, 11, 714, 358, 2776, 537, 2704, 421, 432, 594, 11941, 803, 1091, 220, 18, 13, 20, 3526, 13, 358, 1265, 1083, 6099, 429, 7042, 12396, 646, 13289, 11649, 389, 1246, 807, 525, 16878, 323, 279, 882, 315, 1042, 13, 1752, 3110, 11, 7025, 1251, 525, 29994, 21303, 304, 24421, 476, 3598, 4357, 11, 714, 8789, 11, 432, 594, 2163, 429, 2088, 382, 40, 1744, 358, 2776, 5020, 16506, 429, 12095, 6, 7042, 374, 13187, 220, 18, 13, 20, 3526, 13, 2055, 11, 10687, 429, 3786, 11, 358, 646, 3561, 432, 1119, 4718, 438, 279, 3491, 11223, 13, 358, 1265, 1281, 2704, 311, 2924, 279, 4396, 3283, 829, 323, 7042, 624, 151649, 271, 785, 7042, 315, 279, 6722, 315, 9625, 11, 12095, 11, 374, 13187, 220, 18, 13, 20, 3526, 382, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 18, 20, 15, 15, 15, 15, 15, 198, 532, 73594, 151643], 'meta_info': {'id': '6dfa9d04ec5845f4af19c80cf9e9cefd', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 348, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.185510292649269, 'response_sent_to_client_ts': 1772568222.2230754}}</strong>



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

    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [2026-03-03 20:03:44] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.


    [2026-03-03 20:03:44] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 20:03:44] INFO engine.py:157: server_args=ServerArgs(model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=293972748, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser='deepseek-r1', tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.03s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.29s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.25s/it]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=40.22 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=40.22 GB):   5%|▌         | 1/20 [00:00<00:02,  6.72it/s]Capturing batches (bs=120 avail_mem=40.10 GB):   5%|▌         | 1/20 [00:00<00:02,  6.72it/s]Capturing batches (bs=112 avail_mem=40.10 GB):   5%|▌         | 1/20 [00:00<00:02,  6.72it/s]

    Capturing batches (bs=104 avail_mem=40.10 GB):   5%|▌         | 1/20 [00:00<00:02,  6.72it/s]Capturing batches (bs=96 avail_mem=40.10 GB):   5%|▌         | 1/20 [00:00<00:02,  6.72it/s] Capturing batches (bs=96 avail_mem=40.10 GB):  25%|██▌       | 5/20 [00:00<00:00, 21.21it/s]Capturing batches (bs=88 avail_mem=40.10 GB):  25%|██▌       | 5/20 [00:00<00:00, 21.21it/s]Capturing batches (bs=80 avail_mem=40.10 GB):  25%|██▌       | 5/20 [00:00<00:00, 21.21it/s]Capturing batches (bs=72 avail_mem=40.10 GB):  25%|██▌       | 5/20 [00:00<00:00, 21.21it/s]Capturing batches (bs=64 avail_mem=40.10 GB):  25%|██▌       | 5/20 [00:00<00:00, 21.21it/s]Capturing batches (bs=64 avail_mem=40.10 GB):  45%|████▌     | 9/20 [00:00<00:00, 26.48it/s]Capturing batches (bs=56 avail_mem=40.09 GB):  45%|████▌     | 9/20 [00:00<00:00, 26.48it/s]

    Capturing batches (bs=48 avail_mem=40.09 GB):  45%|████▌     | 9/20 [00:00<00:00, 26.48it/s]Capturing batches (bs=40 avail_mem=40.09 GB):  45%|████▌     | 9/20 [00:00<00:00, 26.48it/s]Capturing batches (bs=32 avail_mem=40.09 GB):  45%|████▌     | 9/20 [00:00<00:00, 26.48it/s]Capturing batches (bs=32 avail_mem=40.09 GB):  65%|██████▌   | 13/20 [00:00<00:00, 29.32it/s]Capturing batches (bs=24 avail_mem=40.09 GB):  65%|██████▌   | 13/20 [00:00<00:00, 29.32it/s]Capturing batches (bs=16 avail_mem=40.09 GB):  65%|██████▌   | 13/20 [00:00<00:00, 29.32it/s]Capturing batches (bs=12 avail_mem=40.09 GB):  65%|██████▌   | 13/20 [00:00<00:00, 29.32it/s]

    Capturing batches (bs=8 avail_mem=40.08 GB):  65%|██████▌   | 13/20 [00:00<00:00, 29.32it/s] Capturing batches (bs=8 avail_mem=40.08 GB):  85%|████████▌ | 17/20 [00:00<00:00, 28.54it/s]Capturing batches (bs=4 avail_mem=40.08 GB):  85%|████████▌ | 17/20 [00:00<00:00, 28.54it/s]Capturing batches (bs=2 avail_mem=40.08 GB):  85%|████████▌ | 17/20 [00:00<00:00, 28.54it/s]Capturing batches (bs=1 avail_mem=40.08 GB):  85%|████████▌ | 17/20 [00:00<00:00, 28.54it/s]Capturing batches (bs=1 avail_mem=40.08 GB): 100%|██████████| 20/20 [00:00<00:00, 28.01it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:24,  2.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:24,  2.54s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:12,  1.30s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:12,  1.30s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:44,  1.23it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:44,  1.23it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:30,  1.75it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:30,  1.75it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:22,  2.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:22,  2.32it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:18,  2.89it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:18,  2.89it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:14,  3.52it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:14,  3.52it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:10,  4.63it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:10,  4.63it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.67it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.67it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:09,  4.77it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.96it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.96it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.96it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:06,  7.04it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:06,  7.04it/s]

    Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:06,  7.04it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:04,  9.12it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:04,  9.12it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:04,  9.12it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:04,  9.12it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:03, 12.44it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:03, 12.44it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:03, 12.44it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:03, 12.44it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:03, 12.44it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:01, 17.56it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 24.63it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 24.63it/s]

    Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 24.63it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 24.63it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 24.63it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 24.63it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 24.63it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 33.20it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 33.20it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 33.20it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 33.20it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 33.20it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:00, 33.20it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:00, 33.20it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:00, 33.20it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:05<00:00, 33.20it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s]

    Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s]Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s]Compiling num tokens (num_tokens=24):  72%|███████▏  | 42/58 [00:05<00:00, 43.93it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 60.86it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 60.86it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 60.86it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 60.86it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 60.86it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 60.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=61.35 GB):   2%|▏         | 1/58 [00:00<00:17,  3.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.32 GB):   2%|▏         | 1/58 [00:00<00:17,  3.34it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=61.32 GB):   3%|▎         | 2/58 [00:00<00:15,  3.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.32 GB):   3%|▎         | 2/58 [00:00<00:15,  3.52it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=61.32 GB):   5%|▌         | 3/58 [00:00<00:14,  3.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.32 GB):   5%|▌         | 3/58 [00:00<00:14,  3.75it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=61.32 GB):   7%|▋         | 4/58 [00:01<00:13,  4.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.33 GB):   7%|▋         | 4/58 [00:01<00:13,  4.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.33 GB):   9%|▊         | 5/58 [00:01<00:12,  4.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.33 GB):   9%|▊         | 5/58 [00:01<00:12,  4.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.33 GB):  10%|█         | 6/58 [00:01<00:11,  4.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.34 GB):  10%|█         | 6/58 [00:01<00:11,  4.69it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=61.34 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.34 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.34 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.34 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.53it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=61.34 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.35 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.35 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.34 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.49it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=61.34 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.34 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.34 GB):  21%|██        | 12/58 [00:02<00:05,  7.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.34 GB):  21%|██        | 12/58 [00:02<00:05,  7.68it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=61.34 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.34 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.34 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.54it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.34 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.54it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=61.34 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.34 GB):  28%|██▊       | 16/58 [00:02<00:04,  8.63it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.34 GB):  28%|██▊       | 16/58 [00:02<00:04,  8.63it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.34 GB):  28%|██▊       | 16/58 [00:02<00:04,  8.63it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.34 GB):  31%|███       | 18/58 [00:02<00:03, 10.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.34 GB):  31%|███       | 18/58 [00:02<00:03, 10.67it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=61.34 GB):  31%|███       | 18/58 [00:02<00:03, 10.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.34 GB):  31%|███       | 18/58 [00:02<00:03, 10.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.34 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.09it/s]Capturing num tokens (num_tokens=960 avail_mem=61.34 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.09it/s] Capturing num tokens (num_tokens=896 avail_mem=61.33 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.09it/s]Capturing num tokens (num_tokens=832 avail_mem=61.33 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.09it/s]Capturing num tokens (num_tokens=832 avail_mem=61.33 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.21it/s]Capturing num tokens (num_tokens=768 avail_mem=61.33 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.21it/s]

    Capturing num tokens (num_tokens=704 avail_mem=61.32 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.21it/s]Capturing num tokens (num_tokens=640 avail_mem=61.32 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.21it/s]Capturing num tokens (num_tokens=576 avail_mem=61.31 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.21it/s]Capturing num tokens (num_tokens=576 avail_mem=61.31 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.24it/s]Capturing num tokens (num_tokens=512 avail_mem=61.31 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.24it/s]Capturing num tokens (num_tokens=480 avail_mem=61.31 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.24it/s]Capturing num tokens (num_tokens=448 avail_mem=61.30 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.24it/s]Capturing num tokens (num_tokens=416 avail_mem=61.30 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.24it/s]

    Capturing num tokens (num_tokens=416 avail_mem=61.30 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Capturing num tokens (num_tokens=384 avail_mem=61.29 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Capturing num tokens (num_tokens=352 avail_mem=61.29 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Capturing num tokens (num_tokens=320 avail_mem=61.29 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Capturing num tokens (num_tokens=288 avail_mem=61.28 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Capturing num tokens (num_tokens=288 avail_mem=61.28 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.47it/s]Capturing num tokens (num_tokens=256 avail_mem=61.28 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.47it/s]Capturing num tokens (num_tokens=240 avail_mem=61.28 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.47it/s]Capturing num tokens (num_tokens=224 avail_mem=61.27 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.47it/s]Capturing num tokens (num_tokens=208 avail_mem=61.27 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.47it/s]

    Capturing num tokens (num_tokens=192 avail_mem=61.26 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.47it/s]Capturing num tokens (num_tokens=192 avail_mem=61.26 GB):  71%|███████   | 41/58 [00:03<00:00, 32.30it/s]Capturing num tokens (num_tokens=176 avail_mem=61.26 GB):  71%|███████   | 41/58 [00:03<00:00, 32.30it/s]Capturing num tokens (num_tokens=160 avail_mem=61.26 GB):  71%|███████   | 41/58 [00:03<00:00, 32.30it/s]Capturing num tokens (num_tokens=144 avail_mem=61.25 GB):  71%|███████   | 41/58 [00:03<00:00, 32.30it/s]Capturing num tokens (num_tokens=128 avail_mem=61.26 GB):  71%|███████   | 41/58 [00:03<00:00, 32.30it/s]Capturing num tokens (num_tokens=128 avail_mem=61.26 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.11it/s]Capturing num tokens (num_tokens=112 avail_mem=61.24 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.11it/s]Capturing num tokens (num_tokens=96 avail_mem=61.24 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.11it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=61.23 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.11it/s]Capturing num tokens (num_tokens=64 avail_mem=61.23 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.11it/s]Capturing num tokens (num_tokens=64 avail_mem=61.23 GB):  84%|████████▍ | 49/58 [00:03<00:00, 31.62it/s]Capturing num tokens (num_tokens=48 avail_mem=61.23 GB):  84%|████████▍ | 49/58 [00:03<00:00, 31.62it/s]Capturing num tokens (num_tokens=32 avail_mem=61.21 GB):  84%|████████▍ | 49/58 [00:03<00:00, 31.62it/s]Capturing num tokens (num_tokens=28 avail_mem=60.73 GB):  84%|████████▍ | 49/58 [00:03<00:00, 31.62it/s]Capturing num tokens (num_tokens=24 avail_mem=60.63 GB):  84%|████████▍ | 49/58 [00:03<00:00, 31.62it/s]Capturing num tokens (num_tokens=24 avail_mem=60.63 GB):  91%|█████████▏| 53/58 [00:03<00:00, 32.97it/s]Capturing num tokens (num_tokens=20 avail_mem=60.56 GB):  91%|█████████▏| 53/58 [00:03<00:00, 32.97it/s]Capturing num tokens (num_tokens=16 avail_mem=60.55 GB):  91%|█████████▏| 53/58 [00:03<00:00, 32.97it/s]

    Capturing num tokens (num_tokens=12 avail_mem=60.55 GB):  91%|█████████▏| 53/58 [00:03<00:00, 32.97it/s]Capturing num tokens (num_tokens=8 avail_mem=60.55 GB):  91%|█████████▏| 53/58 [00:03<00:00, 32.97it/s] Capturing num tokens (num_tokens=4 avail_mem=60.54 GB):  91%|█████████▏| 53/58 [00:04<00:00, 32.97it/s]Capturing num tokens (num_tokens=4 avail_mem=60.54 GB): 100%|██████████| 58/58 [00:04<00:00, 35.83it/s]Capturing num tokens (num_tokens=4 avail_mem=60.54 GB): 100%|██████████| 58/58 [00:04<00:00, 14.37it/s]


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
    Generated text: Rome is the capital of Italy
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
    
    Generated text: Alright, so the user asked for the information and population of the capital of France in JSON format. Hmm, first off, I know that the capital of France is Paris. That's pretty straightforward, right? But wait, maybe I should double-check that to be sure. Yeah, Paris is definitely the capital city.
    
    Now, the user specifically wants the population data. I should make sure to find the most recent numbers. I recall that the population figures for Paris can vary depending on the source and the year. I think as of the latest estimates, around 2023, the population is somewhere between 2 million and 2.1 million. I should probably look up the exact number to provide the most accurate information.
    
    I also need to consider how to format this response. They asked for JSON, so I'll structure it with a key for the city name and another for the population. Making it clear and concise is important. Maybe I can present it like this: {"City": "Paris", "Population": 2075000}. That seems straightforward.
    
    Wait, should I mention that this is an approximate figure or provide a range? It might be helpful to note that population figures can fluctuate due to births, deaths, and migration. But since the user just asked for the information and population, I'll stick to a single number for simplicity unless they ask for more details.
    
    Also, I should ensure the JSON is correctly formatted without any typos. Let me make sure the commas and brackets are in the right places. No, looks good. The city name and population are the main points here, so I don't need to add extra fields unless specified.
    
    I think that's it. The user probably needs this for a quick reference or maybe a report. They might not be looking for an in-depth analysis, just the facts. So providing the JSON with the correct city and population should meet their needs.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "City": "Paris",
      "Population": 2075000
    }
    ```



```python
llm.shutdown()
```
