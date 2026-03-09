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

    [2026-03-09 15:01:20] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-09 15:01:20] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-09 15:01:20] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-09 15:01:24] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-09 15:01:24] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-09 15:01:24] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-09 15:01:26] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.
    [2026-03-09 15:01:26] INFO server_args.py:3223: Set soft_watchdog_timeout since in CI


    [2026-03-09 15:01:31] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-09 15:01:31] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-09 15:01:31] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-09 15:01:31] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-09 15:01:31] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-09 15:01:31] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-09 15:01:36] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-09 15:01:36] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-09 15:01:36] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.36s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:03<00:00,  1.81s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:03<00:00,  1.74s/it]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=30.06 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=30.06 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.89it/s]Capturing batches (bs=2 avail_mem=29.92 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.89it/s]Capturing batches (bs=1 avail_mem=29.92 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.89it/s]Capturing batches (bs=1 avail_mem=29.92 GB): 100%|██████████| 3/3 [00:00<00:00, 10.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:24,  2.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:24,  2.54s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:42,  1.84s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:42,  1.84s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:11,  1.30s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:11,  1.30s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:55,  1.02s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:55,  1.02s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:44,  1.19it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:44,  1.19it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:32,  1.57it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:32,  1.57it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:28,  1.77it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:28,  1.77it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:24,  2.00it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:24,  2.00it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:21,  2.25it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:21,  2.25it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:18,  2.48it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:18,  2.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:16,  2.74it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:16,  2.74it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:14,  3.01it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:14,  3.01it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:13,  3.30it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:13,  3.30it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:11,  3.61it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:11,  3.61it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:10,  4.04it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:10,  4.04it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:09,  4.50it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:09,  4.50it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:07,  5.02it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:07,  5.02it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:09<00:06,  5.63it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:09<00:06,  5.63it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:06,  6.28it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:06,  6.28it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:06,  6.28it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:09<00:04,  8.06it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:09<00:04,  8.06it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:09<00:04,  8.06it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:10<00:03, 10.00it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:10<00:03, 10.00it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:10<00:03, 10.00it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:10<00:02, 11.31it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:10<00:02, 11.31it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:10<00:02, 11.31it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:10<00:02, 12.46it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:10<00:02, 12.46it/s]

    Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:10<00:02, 12.46it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:10<00:02, 12.46it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:10<00:01, 14.87it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:10<00:01, 14.87it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:10<00:01, 14.87it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:10<00:01, 15.68it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:10<00:01, 15.68it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:10<00:01, 15.68it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:10<00:01, 15.68it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:10<00:01, 18.30it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:10<00:01, 18.30it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:10<00:01, 18.30it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:10<00:01, 18.30it/s]

    Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:10<00:00, 21.18it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:10<00:00, 21.18it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:10<00:00, 21.18it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:10<00:00, 21.18it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:10<00:00, 21.18it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:10<00:00, 23.91it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:10<00:00, 23.91it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:10<00:00, 23.91it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:10<00:00, 23.91it/s]

    Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:10<00:00, 23.91it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:11<00:00, 26.05it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:11<00:00, 26.05it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:11<00:00, 26.05it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:11<00:00, 26.05it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:11<00:00, 26.05it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:11<00:00, 28.33it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:11<00:00, 28.33it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:11<00:00, 28.33it/s]

    Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:11<00:00, 28.33it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:11<00:00, 28.33it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:11<00:00, 30.50it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:11<00:00, 30.50it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:11<00:00, 30.50it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:11<00:00, 30.50it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  5.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.24 GB):   2%|▏         | 1/58 [00:00<00:52,  1.08it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.35 GB):   2%|▏         | 1/58 [00:00<00:52,  1.08it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.35 GB):   3%|▎         | 2/58 [00:01<00:49,  1.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.41 GB):   3%|▎         | 2/58 [00:01<00:49,  1.12it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.41 GB):   5%|▌         | 3/58 [00:02<00:46,  1.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.49 GB):   5%|▌         | 3/58 [00:02<00:46,  1.19it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.49 GB):   7%|▋         | 4/58 [00:03<00:42,  1.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=24.79 GB):   7%|▋         | 4/58 [00:03<00:42,  1.28it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=24.79 GB):   9%|▊         | 5/58 [00:03<00:38,  1.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=24.92 GB):   9%|▊         | 5/58 [00:03<00:38,  1.36it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=24.92 GB):  10%|█         | 6/58 [00:04<00:34,  1.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.05 GB):  10%|█         | 6/58 [00:04<00:34,  1.49it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.05 GB):  12%|█▏        | 7/58 [00:05<00:32,  1.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.20 GB):  12%|█▏        | 7/58 [00:05<00:32,  1.56it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=25.20 GB):  14%|█▍        | 8/58 [00:05<00:28,  1.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=25.34 GB):  14%|█▍        | 8/58 [00:05<00:28,  1.74it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=25.34 GB):  16%|█▌        | 9/58 [00:05<00:25,  1.91it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.42 GB):  16%|█▌        | 9/58 [00:05<00:25,  1.91it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=25.42 GB):  17%|█▋        | 10/58 [00:06<00:23,  2.07it/s]Capturing num tokens (num_tokens=3584 avail_mem=25.47 GB):  17%|█▋        | 10/58 [00:06<00:23,  2.07it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=25.47 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=25.53 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.27it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=25.53 GB):  21%|██        | 12/58 [00:06<00:18,  2.49it/s]Capturing num tokens (num_tokens=3072 avail_mem=25.60 GB):  21%|██        | 12/58 [00:06<00:18,  2.49it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=25.60 GB):  22%|██▏       | 13/58 [00:07<00:16,  2.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.11 GB):  22%|██▏       | 13/58 [00:07<00:16,  2.70it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.11 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.40 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.02it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.40 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.37it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.40 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.37it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.40 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.76it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.40 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.76it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=26.40 GB):  29%|██▉       | 17/58 [00:08<00:09,  4.21it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.39 GB):  29%|██▉       | 17/58 [00:08<00:09,  4.21it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.39 GB):  31%|███       | 18/58 [00:08<00:08,  4.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.39 GB):  31%|███       | 18/58 [00:08<00:08,  4.72it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=26.39 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.39 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.39 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.26 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.14it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=26.26 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.83it/s]Capturing num tokens (num_tokens=960 avail_mem=26.25 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.83it/s] Capturing num tokens (num_tokens=960 avail_mem=26.25 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.55it/s]Capturing num tokens (num_tokens=896 avail_mem=26.25 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.55it/s]

    Capturing num tokens (num_tokens=896 avail_mem=26.25 GB):  40%|███▉      | 23/58 [00:08<00:04,  8.13it/s]Capturing num tokens (num_tokens=832 avail_mem=26.24 GB):  40%|███▉      | 23/58 [00:08<00:04,  8.13it/s]

    Capturing num tokens (num_tokens=832 avail_mem=26.24 GB):  41%|████▏     | 24/58 [00:08<00:05,  6.76it/s]Capturing num tokens (num_tokens=768 avail_mem=26.34 GB):  41%|████▏     | 24/58 [00:08<00:05,  6.76it/s]Capturing num tokens (num_tokens=704 avail_mem=26.23 GB):  41%|████▏     | 24/58 [00:09<00:05,  6.76it/s]Capturing num tokens (num_tokens=704 avail_mem=26.23 GB):  45%|████▍     | 26/58 [00:09<00:03,  8.44it/s]Capturing num tokens (num_tokens=640 avail_mem=26.23 GB):  45%|████▍     | 26/58 [00:09<00:03,  8.44it/s]

    Capturing num tokens (num_tokens=640 avail_mem=26.23 GB):  47%|████▋     | 27/58 [00:09<00:03,  8.65it/s]Capturing num tokens (num_tokens=576 avail_mem=26.30 GB):  47%|████▋     | 27/58 [00:09<00:03,  8.65it/s]Capturing num tokens (num_tokens=512 avail_mem=26.29 GB):  47%|████▋     | 27/58 [00:09<00:03,  8.65it/s]Capturing num tokens (num_tokens=512 avail_mem=26.29 GB):  50%|█████     | 29/58 [00:09<00:02, 10.05it/s]Capturing num tokens (num_tokens=480 avail_mem=26.28 GB):  50%|█████     | 29/58 [00:09<00:02, 10.05it/s]

    Capturing num tokens (num_tokens=448 avail_mem=26.29 GB):  50%|█████     | 29/58 [00:09<00:02, 10.05it/s]Capturing num tokens (num_tokens=448 avail_mem=26.29 GB):  53%|█████▎    | 31/58 [00:09<00:02, 11.00it/s]Capturing num tokens (num_tokens=416 avail_mem=26.26 GB):  53%|█████▎    | 31/58 [00:09<00:02, 11.00it/s]Capturing num tokens (num_tokens=384 avail_mem=26.25 GB):  53%|█████▎    | 31/58 [00:09<00:02, 11.00it/s]

    Capturing num tokens (num_tokens=384 avail_mem=26.25 GB):  57%|█████▋    | 33/58 [00:09<00:02, 11.90it/s]Capturing num tokens (num_tokens=352 avail_mem=26.21 GB):  57%|█████▋    | 33/58 [00:09<00:02, 11.90it/s]Capturing num tokens (num_tokens=320 avail_mem=26.23 GB):  57%|█████▋    | 33/58 [00:09<00:02, 11.90it/s]Capturing num tokens (num_tokens=320 avail_mem=26.23 GB):  60%|██████    | 35/58 [00:09<00:01, 13.17it/s]Capturing num tokens (num_tokens=288 avail_mem=26.21 GB):  60%|██████    | 35/58 [00:09<00:01, 13.17it/s]Capturing num tokens (num_tokens=256 avail_mem=26.19 GB):  60%|██████    | 35/58 [00:09<00:01, 13.17it/s]

    Capturing num tokens (num_tokens=256 avail_mem=26.19 GB):  64%|██████▍   | 37/58 [00:09<00:01, 14.58it/s]Capturing num tokens (num_tokens=240 avail_mem=26.19 GB):  64%|██████▍   | 37/58 [00:09<00:01, 14.58it/s]Capturing num tokens (num_tokens=224 avail_mem=26.16 GB):  64%|██████▍   | 37/58 [00:09<00:01, 14.58it/s]Capturing num tokens (num_tokens=224 avail_mem=26.16 GB):  67%|██████▋   | 39/58 [00:10<00:01, 15.14it/s]Capturing num tokens (num_tokens=208 avail_mem=26.15 GB):  67%|██████▋   | 39/58 [00:10<00:01, 15.14it/s]Capturing num tokens (num_tokens=192 avail_mem=26.14 GB):  67%|██████▋   | 39/58 [00:10<00:01, 15.14it/s]

    Capturing num tokens (num_tokens=192 avail_mem=26.14 GB):  71%|███████   | 41/58 [00:10<00:01, 15.74it/s]Capturing num tokens (num_tokens=176 avail_mem=26.15 GB):  71%|███████   | 41/58 [00:10<00:01, 15.74it/s]Capturing num tokens (num_tokens=160 avail_mem=26.15 GB):  71%|███████   | 41/58 [00:10<00:01, 15.74it/s]Capturing num tokens (num_tokens=144 avail_mem=26.13 GB):  71%|███████   | 41/58 [00:10<00:01, 15.74it/s]Capturing num tokens (num_tokens=144 avail_mem=26.13 GB):  76%|███████▌  | 44/58 [00:10<00:00, 17.72it/s]Capturing num tokens (num_tokens=128 avail_mem=26.15 GB):  76%|███████▌  | 44/58 [00:10<00:00, 17.72it/s]Capturing num tokens (num_tokens=112 avail_mem=26.12 GB):  76%|███████▌  | 44/58 [00:10<00:00, 17.72it/s]

    Capturing num tokens (num_tokens=96 avail_mem=26.11 GB):  76%|███████▌  | 44/58 [00:10<00:00, 17.72it/s] Capturing num tokens (num_tokens=96 avail_mem=26.11 GB):  81%|████████  | 47/58 [00:10<00:00, 19.14it/s]Capturing num tokens (num_tokens=80 avail_mem=26.10 GB):  81%|████████  | 47/58 [00:10<00:00, 19.14it/s]Capturing num tokens (num_tokens=64 avail_mem=26.09 GB):  81%|████████  | 47/58 [00:10<00:00, 19.14it/s]Capturing num tokens (num_tokens=48 avail_mem=26.08 GB):  81%|████████  | 47/58 [00:10<00:00, 19.14it/s]Capturing num tokens (num_tokens=48 avail_mem=26.08 GB):  86%|████████▌ | 50/58 [00:10<00:00, 19.96it/s]Capturing num tokens (num_tokens=32 avail_mem=26.07 GB):  86%|████████▌ | 50/58 [00:10<00:00, 19.96it/s]

    Capturing num tokens (num_tokens=28 avail_mem=26.08 GB):  86%|████████▌ | 50/58 [00:10<00:00, 19.96it/s]Capturing num tokens (num_tokens=24 avail_mem=26.05 GB):  86%|████████▌ | 50/58 [00:10<00:00, 19.96it/s]Capturing num tokens (num_tokens=24 avail_mem=26.05 GB):  91%|█████████▏| 53/58 [00:10<00:00, 20.98it/s]Capturing num tokens (num_tokens=20 avail_mem=26.04 GB):  91%|█████████▏| 53/58 [00:10<00:00, 20.98it/s]Capturing num tokens (num_tokens=16 avail_mem=26.04 GB):  91%|█████████▏| 53/58 [00:10<00:00, 20.98it/s]Capturing num tokens (num_tokens=12 avail_mem=26.03 GB):  91%|█████████▏| 53/58 [00:10<00:00, 20.98it/s]

    Capturing num tokens (num_tokens=12 avail_mem=26.03 GB):  97%|█████████▋| 56/58 [00:10<00:00, 21.35it/s]Capturing num tokens (num_tokens=8 avail_mem=26.02 GB):  97%|█████████▋| 56/58 [00:10<00:00, 21.35it/s] Capturing num tokens (num_tokens=4 avail_mem=26.01 GB):  97%|█████████▋| 56/58 [00:10<00:00, 21.35it/s]Capturing num tokens (num_tokens=4 avail_mem=26.01 GB): 100%|██████████| 58/58 [00:10<00:00,  5.33it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants the current date and time along with the weather. Let me figure out how to approach this. <br><br>First, I need to determine which functions to use. The user mentioned two possibilities: 'get_current_date' and 'get_current_weather'. Both are available, so I can use both functions. <br><br>I should structure the response by first getting the date and time using 'get_current_date' with the timezone set to 'America/New_York'. That will give me the current datetime. <br><br>Next, I'll call 'get_current_weather' with the city as 'New York' and the unit as 'fahrenheit' since the user didn't specify the unit. This will provide the temperature and weather conditions. <br><br>I need to make sure each function call is properly formatted as per the instructions. For 'get_current_date', the parameter is 'timezone' with the value 'America/New_York'. For 'get_current_weather', the parameters are 'city' and 'unit'. <br><br>I should also include the sources in the response. So, I'll add a note from the documentation for each function to indicate where the information comes from. <br><br>Putting it all together, I'll first reply with the date and time using the date function, then add the weather information. Each part will be in the specified format with the function calls and parameters included. <br><br>I think that covers everything the user asked for. I just need to make sure the functions are called correctly and the parameters are accurate.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>Note: The documentation for 'get_current_date' can be found here: [Documentation Link](#)  <br>Note: The documentation for 'get_current_weather' can be found here: [Documentation Link](#)</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s straightforward. \n\nNext, I need to find the population. I remember that Paris is a major city, so its population is quite large. I think it\'s over 3 million, but I\'m not exactly sure of the exact number. Maybe I should double-check that. \n\nWait, I recall that the population figure can vary depending on the source and the year. The user didn\'t specify a particular year, so I should probably go with the most recent estimate. I believe the population is around 3,500,000 as of 2023. \n\nNow, I need to structure this information into a JSON format. JSON typically uses key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" since the user mentioned France. \n\nI should make sure the keys are in English to keep it clear. The city is Paris, the population is 3,500,000, and the country is France. I\'ll format this into a JSON object. \n\nI also need to present this in a way that\'s easy to read, so I\'ll use proper syntax with quotation marks and commas in the right places. No trailing commas to avoid errors. \n\nPutting it all together, the JSON should look something like this: a dictionary with the keys and the corresponding values. I\'ll make sure to test it to ensure it\'s valid, but since I\'m just writing it out, I\'ll assume it\'s correct based on my knowledge. \n\nI think that\'s all. The user just needs the information in JSON, so this should satisfy their request.\n</think>{\n\n"name": "Paris",\n"population": 3500000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4710, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 11, 773, 1181, 7042, 374, 5008, 3460, 13, 358, 1744, 432, 594, 916, 220, 18, 3526, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 10696, 358, 1265, 1990, 15934, 429, 13, 4710, 14190, 11, 358, 19091, 429, 279, 7042, 7071, 646, 13289, 11649, 389, 279, 2530, 323, 279, 1042, 13, 576, 1196, 3207, 944, 13837, 264, 3953, 1042, 11, 773, 358, 1265, 4658, 728, 448, 279, 1429, 3213, 16045, 13, 358, 4411, 279, 7042, 374, 2163, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 4710, 7039, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 11136, 5711, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 2474, 279, 1196, 9733, 9625, 13, 4710, 40, 1265, 1281, 2704, 279, 6894, 525, 304, 6364, 311, 2506, 432, 2797, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 13, 358, 3278, 3561, 419, 1119, 264, 4718, 1633, 13, 4710, 40, 1083, 1184, 311, 3042, 419, 304, 264, 1616, 429, 594, 4135, 311, 1349, 11, 773, 358, 3278, 990, 6169, 19482, 448, 54231, 15423, 323, 76602, 304, 279, 1290, 7482, 13, 2308, 27748, 76602, 311, 5648, 5975, 13, 4710, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 264, 10997, 448, 279, 6894, 323, 279, 12159, 2750, 13, 358, 3278, 1281, 2704, 311, 1273, 432, 311, 5978, 432, 594, 2697, 11, 714, 2474, 358, 2776, 1101, 4378, 432, 700, 11, 358, 3278, 9658, 432, 594, 4396, 3118, 389, 847, 6540, 13, 4710, 40, 1744, 429, 594, 678, 13, 576, 1196, 1101, 3880, 279, 1995, 304, 4718, 11, 773, 419, 1265, 26553, 862, 1681, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 18, 20, 15, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'f342b644480e48f082705be316bd4d24', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 405, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.43498894572258, 'response_sent_to_client_ts': 1773068552.5292466}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants both the general information and the population. So, I\'ll create an object with a "name" field for the capital, a "general_information" section that includes the administrative center, area, and government department, and a "population" section that includes the current population and a note about the data source.\n\nI should also add a "source" field to indicate where the population data comes from, which is the 2020 census. This makes the information more transparent and trustworthy.\n\nPutting it all together, I\'ll format the JSON with proper syntax, using double quotes for strings and ensuring that the keys are clear and descriptive. I\'ll make sure there are no typos and that the JSON is valid.\n\nFinally, I\'ll present the JSON in a code block so the user can easily copy and use it. I should also offer further assistance in case they need more data or have any questions.\n</think>{\n  "name": "Paris",\n  "population": 2174300\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 2176, 279, 4586, 1995, 323, 279, 7042, 13, 2055, 11, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 6722, 11, 264, 330, 24595, 35212, 1, 3772, 429, 5646, 279, 22707, 4126, 11, 3082, 11, 323, 3033, 9292, 11, 323, 264, 330, 44441, 1, 3772, 429, 5646, 279, 1482, 7042, 323, 264, 5185, 911, 279, 821, 2530, 382, 40, 1265, 1083, 912, 264, 330, 2427, 1, 2070, 311, 13216, 1380, 279, 7042, 821, 4041, 504, 11, 892, 374, 279, 220, 17, 15, 17, 15, 43602, 13, 1096, 3643, 279, 1995, 803, 17821, 323, 55942, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3561, 279, 4718, 448, 6169, 19482, 11, 1667, 1990, 17194, 369, 9069, 323, 22573, 429, 279, 6894, 525, 2797, 323, 52844, 13, 358, 3278, 1281, 2704, 1052, 525, 902, 13580, 966, 323, 429, 279, 4718, 374, 2697, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 304, 264, 2038, 2504, 773, 279, 1196, 646, 6707, 2975, 323, 990, 432, 13, 358, 1265, 1083, 3010, 4623, 12994, 304, 1142, 807, 1184, 803, 821, 476, 614, 894, 4755, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 198, 92, 151643], 'meta_info': {'id': 'bb565682f5864f21bcb4d6a5d14838ea', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 386, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.4836543747223914, 'response_sent_to_client_ts': 1773068555.0233364}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'c595abcf08d14749aa99406c6b4d883e', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.5647215149365366, 'response_sent_to_client_ts': 1773068555.6213155}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '3d504c33d4d2423182d0ab03b2c9006b', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.5646434342488647, 'response_sent_to_client_ts': 1773068555.6213284}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '4dcfdf38915c4292b660fa6eb1957d17', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.5646021566353738, 'response_sent_to_client_ts': 1773068555.6213324}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '24d4c625d5f641a388bfdefc90539094', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 25.505884347017854, 'response_sent_to_client_ts': 1773068581.133791}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so I need to find the population of the capital of France and present it in JSON format. Hmm, let me start by recalling which city is the capital of France. I think it\'s Paris, but I\'m not 100% sure because sometimes capitals can change, although I don\'t remember any recent changes. So I\'ll go with Paris as the capital.\n\nNext, I need the population figure. I\'ve heard that Paris is a very populous city, probably the most populous in France. Let me think, I remember reading that the population was around 2 million, but I\'m not certain. Alternatively, sometimes sources give it as 2 million plus, maybe even 2.1 or 2.2 million. I should double-check to be accurate. But since I\'m supposed to imagine this as someone thinking through, I\'ll note that social media and recent estimates might say 2 million. It\'s also possible that the exact figure can vary per year due to births, deaths, and migrations.\n\nNow, structuring the information in JSON format. JSON typically uses key-value pairs, so the keys would be things like "country," "capital," and "population." The population needs to be a numerical value, probably an integer, since we\'re talking about whole numbers of people. I might include parentheses around the population number to indicate it\'s an estimate, which could be helpful for clarity.\n\nWait, should I make the country name lowercase? Sometimes in JSON, it\'s conventional to have keys in lowercase, so "country" would be lowercase. The capital\'s name should probably be title case or camel case, so "Paris" makes sense. The population key should be something descriptive, maybe "population_of_paris." So putting it all together, the JSON structure would have a "country" key with "France," a "capital" key with "Paris," and a "population_of_paris" key with the number, perhaps in parentheses.\n\nAlso, thinking about formatting, maybe line breaks to make it readable. So each key-value pair would be on its own line. I should make sure there\'s proper comma placement between the key-value pairs.\n\nWait, but if the population is an estimate, should it be a rounded number or precise? I think including an approximate figure is fine, so 2 million with parentheses indicating it\'s an estimate would be clear.\n\nSo putting it all together, I\'ll structure the JSON like this:\n\n{\n  "country": "France",\n  "capital": "Paris",\n  "population_of_paris": "(Approximately 2023 population)"\n}\n\nWait, that last part seems a bit awkward because number is 2 million. Maybe I should format it as (2,000,000) to make it clearer. That way, it\'s both explicit and more readable.\n\nSo revising:\n\n{\n  "country": "France",\n  "capital": "Paris",\n  "population_of_paris": "(2,000,000)"\n}\n\nBut I should remember that population figures can change over time, so it\'s essential to mention that it\'s an estimate, perhaps including a range if more accuracy is desired. However, since the user only asked for the information and population, providing a single number with a note that it\'s an estimate seems sufficient.\n\nAlternatively, I could check if Paris\'s population is often cited as around 2.19 million as of recent years, so maybe (2,190,000). That would be more specific.\n\nBut I want to confirm—when was the population last updated? If it\'s from 2022, maybe the population is around 2.19 million. However, without checking current data, I shouldn\'t rely on that number. Given that I don\'t have access to real-time data, I\'ll stick with the commonly known figure, which is around 2 million.\n\nSo, the final JSON would be:\n\n{\n  "country": "France",\n  "capital": "Paris",\n  "population_of_paris": "(Approximately 2 million)"\n}\n\nBut to make it more precise, I\'ll use numbers without the word \'million\' for simplicity, but it\'s better to express it as a number with a note. Alternatively, I could write it as a numerical value with commas for readability.\n\nSo, to sum up, the JSON would have the country, the capital, and the population of the capital, with the population being an approximate number, likely around 2 million, presented in a readable format.\n</think>\n\n```json\n{\n  "country": "France",\n  "capital": "Paris",\n  "population_of_paris": "(Approximately 2,000,000)"\n}\n```', 'output_ids': [32313, 11, 773, 358, 1184, 311, 1477, 279, 7042, 315, 279, 6722, 315, 9625, 323, 3042, 432, 304, 4718, 3561, 13, 88190, 11, 1077, 752, 1191, 553, 88646, 892, 3283, 374, 279, 6722, 315, 9625, 13, 358, 1744, 432, 594, 12095, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 2704, 1576, 7025, 92999, 646, 2297, 11, 7892, 358, 1513, 944, 6099, 894, 3213, 4344, 13, 2055, 358, 3278, 728, 448, 12095, 438, 279, 6722, 382, 5847, 11, 358, 1184, 279, 7042, 7071, 13, 358, 3003, 6617, 429, 12095, 374, 264, 1602, 94451, 3283, 11, 4658, 279, 1429, 94451, 304, 9625, 13, 6771, 752, 1744, 11, 358, 6099, 5290, 429, 279, 7042, 572, 2163, 220, 17, 3526, 11, 714, 358, 2776, 537, 3654, 13, 38478, 11, 7025, 8173, 2968, 432, 438, 220, 17, 3526, 5519, 11, 7196, 1496, 220, 17, 13, 16, 476, 220, 17, 13, 17, 3526, 13, 358, 1265, 1990, 15934, 311, 387, 13382, 13, 1988, 2474, 358, 2776, 9966, 311, 12793, 419, 438, 4325, 7274, 1526, 11, 358, 3278, 5185, 429, 3590, 3687, 323, 3213, 17530, 2578, 1977, 220, 17, 3526, 13, 1084, 594, 1083, 3204, 429, 279, 4734, 7071, 646, 13289, 817, 1042, 4152, 311, 65232, 11, 16375, 11, 323, 17063, 382, 7039, 11, 2036, 1677, 279, 1995, 304, 4718, 3561, 13, 4718, 11136, 5711, 1376, 19083, 13530, 11, 773, 279, 6894, 1035, 387, 2513, 1075, 330, 11141, 1335, 330, 65063, 1335, 323, 330, 44441, 1189, 576, 7042, 3880, 311, 387, 264, 34776, 897, 11, 4658, 458, 7546, 11, 2474, 582, 2299, 7404, 911, 4361, 5109, 315, 1251, 13, 358, 2578, 2924, 73975, 2163, 279, 7042, 1372, 311, 13216, 432, 594, 458, 16045, 11, 892, 1410, 387, 10950, 369, 31273, 382, 14190, 11, 1265, 358, 1281, 279, 3146, 829, 42047, 30, 17688, 304, 4718, 11, 432, 594, 20692, 311, 614, 6894, 304, 42047, 11, 773, 330, 11141, 1, 1035, 387, 42047, 13, 576, 6722, 594, 829, 1265, 4658, 387, 2265, 1142, 476, 49152, 1142, 11, 773, 330, 59604, 1, 3643, 5530, 13, 576, 7042, 1376, 1265, 387, 2494, 52844, 11, 7196, 330, 44441, 3575, 22654, 285, 1189, 2055, 10687, 432, 678, 3786, 11, 279, 4718, 5944, 1035, 614, 264, 330, 11141, 1, 1376, 448, 330, 49000, 1335, 264, 330, 65063, 1, 1376, 448, 330, 59604, 1335, 323, 264, 330, 44441, 3575, 22654, 285, 1, 1376, 448, 279, 1372, 11, 8365, 304, 73975, 382, 13394, 11, 7274, 911, 36566, 11, 7196, 1555, 18303, 311, 1281, 432, 33798, 13, 2055, 1817, 1376, 19083, 6716, 1035, 387, 389, 1181, 1828, 1555, 13, 358, 1265, 1281, 2704, 1052, 594, 6169, 31683, 21448, 1948, 279, 1376, 19083, 13530, 382, 14190, 11, 714, 421, 279, 7042, 374, 458, 16045, 11, 1265, 432, 387, 264, 17976, 1372, 476, 23560, 30, 358, 1744, 2670, 458, 44868, 7071, 374, 6915, 11, 773, 220, 17, 3526, 448, 73975, 18860, 432, 594, 458, 16045, 1035, 387, 2797, 382, 4416, 10687, 432, 678, 3786, 11, 358, 3278, 5944, 279, 4718, 1075, 419, 1447, 515, 220, 330, 11141, 788, 330, 49000, 756, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 3575, 22654, 285, 788, 11993, 69520, 7108, 220, 17, 15, 17, 18, 7042, 12954, 630, 14190, 11, 429, 1537, 949, 4977, 264, 2699, 28759, 1576, 1372, 374, 220, 17, 3526, 13, 10696, 358, 1265, 3561, 432, 438, 320, 17, 11, 15, 15, 15, 11, 15, 15, 15, 8, 311, 1281, 432, 48379, 13, 2938, 1616, 11, 432, 594, 2176, 11464, 323, 803, 33798, 382, 4416, 17493, 287, 1447, 515, 220, 330, 11141, 788, 330, 49000, 756, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 3575, 22654, 285, 788, 11993, 17, 11, 15, 15, 15, 11, 15, 15, 15, 12954, 630, 3983, 358, 1265, 6099, 429, 7042, 12396, 646, 2297, 916, 882, 11, 773, 432, 594, 7565, 311, 6286, 429, 432, 594, 458, 16045, 11, 8365, 2670, 264, 2088, 421, 803, 13403, 374, 12685, 13, 4354, 11, 2474, 279, 1196, 1172, 4588, 369, 279, 1995, 323, 7042, 11, 8241, 264, 3175, 1372, 448, 264, 5185, 429, 432, 594, 458, 16045, 4977, 14016, 382, 92014, 11, 358, 1410, 1779, 421, 12095, 594, 7042, 374, 3545, 21870, 438, 2163, 220, 17, 13, 16, 24, 3526, 438, 315, 3213, 1635, 11, 773, 7196, 320, 17, 11, 16, 24, 15, 11, 15, 15, 15, 568, 2938, 1035, 387, 803, 3151, 382, 3983, 358, 1366, 311, 7683, 2293, 9309, 572, 279, 7042, 1537, 6049, 30, 1416, 432, 594, 504, 220, 17, 15, 17, 17, 11, 7196, 279, 7042, 374, 2163, 220, 17, 13, 16, 24, 3526, 13, 4354, 11, 2041, 13295, 1482, 821, 11, 358, 13133, 944, 17188, 389, 429, 1372, 13, 16246, 429, 358, 1513, 944, 614, 2615, 311, 1931, 7246, 821, 11, 358, 3278, 9214, 448, 279, 16626, 3881, 7071, 11, 892, 374, 2163, 220, 17, 3526, 382, 4416, 11, 279, 1590, 4718, 1035, 387, 1447, 515, 220, 330, 11141, 788, 330, 49000, 756, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 3575, 22654, 285, 788, 11993, 69520, 7108, 220, 17, 3526, 12954, 630, 3983, 311, 1281, 432, 803, 23560, 11, 358, 3278, 990, 5109, 2041, 279, 3409, 364, 58313, 6, 369, 38975, 11, 714, 432, 594, 2664, 311, 3158, 432, 438, 264, 1372, 448, 264, 5185, 13, 38478, 11, 358, 1410, 3270, 432, 438, 264, 34776, 897, 448, 76602, 369, 91494, 382, 4416, 11, 311, 2629, 705, 11, 279, 4718, 1035, 614, 279, 3146, 11, 279, 6722, 11, 323, 279, 7042, 315, 279, 6722, 11, 448, 279, 7042, 1660, 458, 44868, 1372, 11, 4363, 2163, 220, 17, 3526, 11, 10449, 304, 264, 33798, 3561, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 11141, 788, 330, 49000, 756, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 3575, 22654, 285, 788, 11993, 69520, 7108, 220, 17, 11, 15, 15, 15, 11, 15, 15, 15, 12954, 532, 73594, 151643], 'meta_info': {'id': 'becfce147e0841548a976dbd7a2fb261', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 974, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 8.854095950257033, 'response_sent_to_client_ts': 1773068589.9963448}}</strong>



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


    [2026-03-09 15:03:12] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.


    [2026-03-09 15:03:12] INFO server_args.py:3223: Set soft_watchdog_timeout since in CI


    [2026-03-09 15:03:12] INFO engine.py:177: server_args=ServerArgs(model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=424642877, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser='deepseek-r1', tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.19s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.51s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.46s/it]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=46.31 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=46.31 GB):   5%|▌         | 1/20 [00:00<00:02,  6.44it/s]Capturing batches (bs=120 avail_mem=46.17 GB):   5%|▌         | 1/20 [00:00<00:02,  6.44it/s]Capturing batches (bs=112 avail_mem=46.17 GB):   5%|▌         | 1/20 [00:00<00:02,  6.44it/s]

    Capturing batches (bs=104 avail_mem=46.17 GB):   5%|▌         | 1/20 [00:00<00:02,  6.44it/s]Capturing batches (bs=96 avail_mem=46.17 GB):   5%|▌         | 1/20 [00:00<00:02,  6.44it/s] Capturing batches (bs=96 avail_mem=46.17 GB):  25%|██▌       | 5/20 [00:00<00:00, 20.44it/s]Capturing batches (bs=88 avail_mem=46.17 GB):  25%|██▌       | 5/20 [00:00<00:00, 20.44it/s]Capturing batches (bs=80 avail_mem=46.17 GB):  25%|██▌       | 5/20 [00:00<00:00, 20.44it/s]Capturing batches (bs=72 avail_mem=46.16 GB):  25%|██▌       | 5/20 [00:00<00:00, 20.44it/s]Capturing batches (bs=64 avail_mem=46.16 GB):  25%|██▌       | 5/20 [00:00<00:00, 20.44it/s]Capturing batches (bs=64 avail_mem=46.16 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.46it/s]Capturing batches (bs=56 avail_mem=46.16 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.46it/s]

    Capturing batches (bs=48 avail_mem=46.16 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.46it/s]Capturing batches (bs=40 avail_mem=46.16 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.46it/s]Capturing batches (bs=32 avail_mem=46.16 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.46it/s]Capturing batches (bs=32 avail_mem=46.16 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.83it/s]Capturing batches (bs=24 avail_mem=46.16 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.83it/s]Capturing batches (bs=16 avail_mem=46.15 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.83it/s]Capturing batches (bs=12 avail_mem=46.15 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.83it/s]

    Capturing batches (bs=12 avail_mem=46.15 GB):  80%|████████  | 16/20 [00:00<00:00, 27.05it/s]Capturing batches (bs=8 avail_mem=46.15 GB):  80%|████████  | 16/20 [00:00<00:00, 27.05it/s] Capturing batches (bs=4 avail_mem=46.15 GB):  80%|████████  | 16/20 [00:00<00:00, 27.05it/s]Capturing batches (bs=2 avail_mem=46.15 GB):  80%|████████  | 16/20 [00:00<00:00, 27.05it/s]Capturing batches (bs=1 avail_mem=46.15 GB):  80%|████████  | 16/20 [00:00<00:00, 27.05it/s]Capturing batches (bs=1 avail_mem=46.15 GB): 100%|██████████| 20/20 [00:00<00:00, 27.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:30,  1.61s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:30,  1.61s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:59,  1.08s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:59,  1.08s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:43,  1.24it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:43,  1.24it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.56it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.56it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:27,  1.86it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:27,  1.86it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:23,  2.20it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:23,  2.20it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:19,  2.55it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:19,  2.55it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:16,  2.91it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:16,  2.91it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:14,  3.37it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:14,  3.37it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:12,  3.74it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:12,  3.74it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:10,  4.21it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:10,  4.21it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:09,  4.83it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:09,  4.83it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:08,  5.32it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:08,  5.32it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:07,  5.96it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:07,  5.96it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:06,  6.75it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:06,  6.75it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:06<00:06,  6.75it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:07<00:04,  8.34it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:07<00:04,  8.34it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:07<00:04,  8.34it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:03, 10.03it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:03, 10.03it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:03, 10.03it/s] 

    Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:03, 10.03it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:02, 13.40it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:02, 13.40it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:02, 13.40it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:07<00:02, 13.40it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:07<00:02, 13.40it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:07<00:01, 18.14it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:07<00:01, 18.14it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:07<00:01, 18.14it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:07<00:01, 18.14it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:07<00:01, 18.14it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:07<00:01, 22.55it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:07<00:01, 22.55it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:07<00:01, 22.55it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:07<00:01, 22.55it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:07<00:01, 22.55it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 25.58it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 25.58it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 25.58it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 25.58it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 25.58it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 25.58it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:07<00:00, 31.33it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:07<00:00, 31.33it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:07<00:00, 31.33it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:07<00:00, 31.33it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:07<00:00, 31.33it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:07<00:00, 31.33it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:07<00:00, 31.33it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:07<00:00, 31.33it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:07<00:00, 40.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=9.48 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=9.48 GB):   2%|▏         | 1/58 [00:00<00:32,  1.73it/s]Capturing num tokens (num_tokens=7680 avail_mem=9.17 GB):   2%|▏         | 1/58 [00:00<00:32,  1.73it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=9.17 GB):   3%|▎         | 2/58 [00:01<00:34,  1.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=9.06 GB):   3%|▎         | 2/58 [00:01<00:34,  1.64it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=9.06 GB):   5%|▌         | 3/58 [00:01<00:33,  1.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=9.23 GB):   5%|▌         | 3/58 [00:01<00:33,  1.64it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=9.23 GB):   7%|▋         | 4/58 [00:02<00:31,  1.71it/s]Capturing num tokens (num_tokens=6144 avail_mem=9.30 GB):   7%|▋         | 4/58 [00:02<00:31,  1.71it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=9.30 GB):   9%|▊         | 5/58 [00:02<00:29,  1.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=10.07 GB):   9%|▊         | 5/58 [00:02<00:29,  1.81it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=10.07 GB):  10%|█         | 6/58 [00:03<00:26,  1.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=10.08 GB):  10%|█         | 6/58 [00:03<00:26,  1.96it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=10.08 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.22it/s]Capturing num tokens (num_tokens=4608 avail_mem=10.08 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.22it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=10.08 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=10.09 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.53it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=10.09 GB):  16%|█▌        | 9/58 [00:04<00:17,  2.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=10.09 GB):  16%|█▌        | 9/58 [00:04<00:17,  2.83it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=10.09 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=10.09 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.12it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=10.09 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=10.08 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.46it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=10.08 GB):  21%|██        | 12/58 [00:04<00:12,  3.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=10.08 GB):  21%|██        | 12/58 [00:04<00:12,  3.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=10.08 GB):  22%|██▏       | 13/58 [00:05<00:10,  4.18it/s]Capturing num tokens (num_tokens=2816 avail_mem=10.08 GB):  22%|██▏       | 13/58 [00:05<00:10,  4.18it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=10.08 GB):  24%|██▍       | 14/58 [00:05<00:09,  4.47it/s]Capturing num tokens (num_tokens=2560 avail_mem=10.08 GB):  24%|██▍       | 14/58 [00:05<00:09,  4.47it/s]Capturing num tokens (num_tokens=2560 avail_mem=10.08 GB):  26%|██▌       | 15/58 [00:05<00:08,  4.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=10.07 GB):  26%|██▌       | 15/58 [00:05<00:08,  4.99it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=10.07 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=9.88 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.26it/s] Capturing num tokens (num_tokens=2048 avail_mem=9.88 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=9.57 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.67it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=9.57 GB):  31%|███       | 18/58 [00:05<00:06,  6.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=9.44 GB):  31%|███       | 18/58 [00:05<00:06,  6.43it/s]Capturing num tokens (num_tokens=1280 avail_mem=9.42 GB):  31%|███       | 18/58 [00:05<00:06,  6.43it/s]Capturing num tokens (num_tokens=1280 avail_mem=9.42 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=9.41 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]

    Capturing num tokens (num_tokens=960 avail_mem=9.41 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s] Capturing num tokens (num_tokens=960 avail_mem=9.41 GB):  38%|███▊      | 22/58 [00:06<00:03,  9.89it/s]Capturing num tokens (num_tokens=896 avail_mem=9.37 GB):  38%|███▊      | 22/58 [00:06<00:03,  9.89it/s]Capturing num tokens (num_tokens=832 avail_mem=9.36 GB):  38%|███▊      | 22/58 [00:06<00:03,  9.89it/s]

    Capturing num tokens (num_tokens=832 avail_mem=9.36 GB):  41%|████▏     | 24/58 [00:06<00:03, 11.31it/s]Capturing num tokens (num_tokens=768 avail_mem=9.35 GB):  41%|████▏     | 24/58 [00:06<00:03, 11.31it/s]Capturing num tokens (num_tokens=704 avail_mem=9.35 GB):  41%|████▏     | 24/58 [00:06<00:03, 11.31it/s]Capturing num tokens (num_tokens=704 avail_mem=9.35 GB):  45%|████▍     | 26/58 [00:06<00:02, 12.69it/s]Capturing num tokens (num_tokens=640 avail_mem=9.34 GB):  45%|████▍     | 26/58 [00:06<00:02, 12.69it/s]Capturing num tokens (num_tokens=576 avail_mem=9.28 GB):  45%|████▍     | 26/58 [00:06<00:02, 12.69it/s]

    Capturing num tokens (num_tokens=576 avail_mem=9.28 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.02it/s]Capturing num tokens (num_tokens=512 avail_mem=9.27 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.02it/s]Capturing num tokens (num_tokens=480 avail_mem=9.26 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.02it/s]Capturing num tokens (num_tokens=448 avail_mem=9.26 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.02it/s]Capturing num tokens (num_tokens=448 avail_mem=9.26 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.96it/s]Capturing num tokens (num_tokens=416 avail_mem=9.25 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.96it/s]Capturing num tokens (num_tokens=384 avail_mem=9.24 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.96it/s]

    Capturing num tokens (num_tokens=352 avail_mem=9.23 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.96it/s]Capturing num tokens (num_tokens=352 avail_mem=9.23 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.95it/s]Capturing num tokens (num_tokens=320 avail_mem=9.23 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.95it/s]Capturing num tokens (num_tokens=288 avail_mem=9.22 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.95it/s]Capturing num tokens (num_tokens=256 avail_mem=9.21 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.95it/s]Capturing num tokens (num_tokens=256 avail_mem=9.21 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.90it/s]Capturing num tokens (num_tokens=240 avail_mem=9.21 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.90it/s]

    Capturing num tokens (num_tokens=224 avail_mem=9.20 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.90it/s]Capturing num tokens (num_tokens=208 avail_mem=9.19 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.90it/s]Capturing num tokens (num_tokens=208 avail_mem=9.19 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.56it/s]Capturing num tokens (num_tokens=192 avail_mem=9.18 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.56it/s]Capturing num tokens (num_tokens=176 avail_mem=9.17 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.56it/s]Capturing num tokens (num_tokens=160 avail_mem=9.16 GB):  69%|██████▉   | 40/58 [00:07<00:00, 21.56it/s]Capturing num tokens (num_tokens=160 avail_mem=9.16 GB):  74%|███████▍  | 43/58 [00:07<00:00, 22.94it/s]Capturing num tokens (num_tokens=144 avail_mem=9.17 GB):  74%|███████▍  | 43/58 [00:07<00:00, 22.94it/s]

    Capturing num tokens (num_tokens=128 avail_mem=9.18 GB):  74%|███████▍  | 43/58 [00:07<00:00, 22.94it/s]Capturing num tokens (num_tokens=112 avail_mem=9.17 GB):  74%|███████▍  | 43/58 [00:07<00:00, 22.94it/s]Capturing num tokens (num_tokens=112 avail_mem=9.17 GB):  79%|███████▉  | 46/58 [00:07<00:00, 24.06it/s]Capturing num tokens (num_tokens=96 avail_mem=9.16 GB):  79%|███████▉  | 46/58 [00:07<00:00, 24.06it/s] Capturing num tokens (num_tokens=80 avail_mem=9.16 GB):  79%|███████▉  | 46/58 [00:07<00:00, 24.06it/s]Capturing num tokens (num_tokens=64 avail_mem=9.15 GB):  79%|███████▉  | 46/58 [00:07<00:00, 24.06it/s]

    Capturing num tokens (num_tokens=64 avail_mem=9.15 GB):  84%|████████▍ | 49/58 [00:07<00:00, 22.17it/s]Capturing num tokens (num_tokens=48 avail_mem=9.15 GB):  84%|████████▍ | 49/58 [00:07<00:00, 22.17it/s]Capturing num tokens (num_tokens=32 avail_mem=9.09 GB):  84%|████████▍ | 49/58 [00:07<00:00, 22.17it/s]Capturing num tokens (num_tokens=28 avail_mem=9.09 GB):  84%|████████▍ | 49/58 [00:07<00:00, 22.17it/s]Capturing num tokens (num_tokens=28 avail_mem=9.09 GB):  90%|████████▉ | 52/58 [00:07<00:00, 22.39it/s]Capturing num tokens (num_tokens=24 avail_mem=9.08 GB):  90%|████████▉ | 52/58 [00:07<00:00, 22.39it/s]Capturing num tokens (num_tokens=20 avail_mem=9.07 GB):  90%|████████▉ | 52/58 [00:07<00:00, 22.39it/s]

    Capturing num tokens (num_tokens=16 avail_mem=9.04 GB):  90%|████████▉ | 52/58 [00:07<00:00, 22.39it/s]Capturing num tokens (num_tokens=16 avail_mem=9.04 GB):  95%|█████████▍| 55/58 [00:07<00:00, 24.00it/s]Capturing num tokens (num_tokens=12 avail_mem=9.05 GB):  95%|█████████▍| 55/58 [00:07<00:00, 24.00it/s]Capturing num tokens (num_tokens=8 avail_mem=9.04 GB):  95%|█████████▍| 55/58 [00:07<00:00, 24.00it/s] Capturing num tokens (num_tokens=4 avail_mem=9.03 GB):  95%|█████████▍| 55/58 [00:07<00:00, 24.00it/s]Capturing num tokens (num_tokens=4 avail_mem=9.03 GB): 100%|██████████| 58/58 [00:07<00:00, 24.06it/s]Capturing num tokens (num_tokens=4 avail_mem=9.03 GB): 100%|██████████| 58/58 [00:07<00:00,  7.54it/s]


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
    
    Generated text: Okay, so I need to figure out the capital of France and its population. Let me start by recalling what I know about France. I know that Paris is the capital, but I'm not exactly sure about the current population. I think it's over 30 million, but I'm not 100% certain. Maybe I should check some sources or think about recent demographic trends.
    
    Wait, when was the population last updated? I remember reading that France has been going through some population changes because of immigration and natural growth. Maybe the population is around 30 million, but perhaps a bit more or less. I'm not certain about the exact number, though.
    
    I also wonder if there's any recent news about population growth in France. I think there was talk about how the country is dealing with an aging population, which might affect the growth rate. But immigration seems to be a balancing factor. So, the population might be slightly higher than it was a few years ago.
    
    Let me try to piece together some numbers. I think the population in the 2010s was around 30-32 million. Maybe it's about 30.7 million now? I'm not sure if that's accurate, but I've heard numbers around that range. I should consider that the figure might vary slightly depending on the source and the year.
    
    So, putting it together, the capital is Paris, and the population is approximately 30.7 million. I should present this in a JSON format as the user requested. I need to make sure the JSON is correctly formatted with the key-value pairs for "capital" and "population". I should also check for any typos or formatting issues to ensure it's valid JSON.
    
    Wait, should I include the units? Probably not necessary since population is typically in whole numbers without units in JSON like that. So, just the number as an integer or a float. Given that population can be a precise number, maybe I should present it as an integer, but I'm not sure if 30.7 million is an exact figure or an estimate. If it's an estimate, maybe I should leave it as a float. 
    
    Alternatively, I could check if the exact current population is available. But since I can't access real-time data, I have to go with the best estimate I have, which is around 30.7 million. So, I'll format it as 30700000, which is 30.7 million in integer form. That should be acceptable.
    
    So, the final JSON should have "capital": "Paris" and "population": 30700000. Let me write that out correctly, making sure the keys are in quotes and the values are properly formatted.
    </think>
    
    Here is the information about the capital of France and its population in JSON format:
    
    ```json
    {
      "capital": "Paris",
      "population": 30700000
    }
    ```



```python
llm.shutdown()
```
