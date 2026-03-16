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

    [2026-03-16 08:16:39] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-16 08:16:39] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-16 08:16:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-16 08:16:43] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-16 08:16:43] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-16 08:16:43] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-16 08:16:45] INFO server_args.py:2154: Attention backend not specified. Use fa3 backend by default.
    [2026-03-16 08:16:45] INFO server_args.py:3311: Set soft_watchdog_timeout since in CI


    [2026-03-16 08:16:50] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-16 08:16:50] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-16 08:16:50] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-16 08:16:50] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-16 08:16:50] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-16 08:16:50] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-16 08:16:55] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-16 08:16:55] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-16 08:16:55] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.40s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.34s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:53,  3.05s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:53,  3.05s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:25,  1.53s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:25,  1.53s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:50,  1.09it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:50,  1.09it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.59it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.59it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.19it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.19it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.83it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.83it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.56it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.56it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.21it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.21it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:11,  4.18it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:11,  4.18it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:11,  4.09it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:12,  3.76it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:12,  3.76it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.74it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.74it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:10,  4.19it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:10,  4.19it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:09,  4.62it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:09,  4.62it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:08,  5.20it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:08,  5.20it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:07,  5.74it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:07,  5.74it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:06,  6.33it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:06,  6.33it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:06,  6.33it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:06<00:05,  7.63it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:06<00:05,  7.63it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:06<00:05,  7.63it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:06<00:04,  8.84it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:06<00:04,  8.84it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:06<00:04,  8.84it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:06<00:03, 10.51it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:06<00:03, 10.51it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:06<00:03, 10.51it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:06<00:02, 11.97it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:06<00:02, 11.97it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:06<00:02, 11.97it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:06<00:02, 11.97it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:07<00:02, 14.39it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:07<00:02, 14.39it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:07<00:02, 14.39it/s]

    Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:07<00:02, 14.39it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:07<00:01, 16.85it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:07<00:01, 16.85it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:07<00:01, 16.85it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:07<00:01, 16.84it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:07<00:01, 16.84it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:07<00:01, 16.84it/s]

    Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:07<00:01, 16.84it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:01, 19.14it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:01, 19.14it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:01, 19.14it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:01, 19.14it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:07<00:00, 21.52it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:07<00:00, 21.52it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:07<00:00, 21.52it/s]

    Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:07<00:00, 21.52it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:07<00:00, 22.26it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:07<00:00, 22.26it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:07<00:00, 22.26it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:07<00:00, 22.26it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:07<00:00, 24.05it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:07<00:00, 24.05it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:07<00:00, 24.05it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:07<00:00, 24.05it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:07<00:00, 25.48it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:07<00:00, 25.48it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:07<00:00, 25.48it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:07<00:00, 25.48it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:07<00:00, 25.58it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:07<00:00, 25.58it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:08<00:00, 25.58it/s]

    Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:08<00:00, 25.58it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:08<00:00, 25.58it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:08<00:00, 27.84it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:08<00:00, 27.84it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:08<00:00, 27.84it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:08<00:00, 27.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=33.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=33.30 GB):   2%|▏         | 1/58 [00:00<00:32,  1.74it/s]Capturing num tokens (num_tokens=7680 avail_mem=31.76 GB):   2%|▏         | 1/58 [00:00<00:32,  1.74it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=31.76 GB):   3%|▎         | 2/58 [00:01<00:27,  2.01it/s]Capturing num tokens (num_tokens=7168 avail_mem=29.45 GB):   3%|▎         | 2/58 [00:01<00:27,  2.01it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=29.45 GB):   5%|▌         | 3/58 [00:01<00:24,  2.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=27.03 GB):   5%|▌         | 3/58 [00:01<00:24,  2.27it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=27.03 GB):   7%|▋         | 4/58 [00:01<00:20,  2.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=27.03 GB):   7%|▋         | 4/58 [00:01<00:20,  2.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=27.03 GB):   9%|▊         | 5/58 [00:01<00:17,  3.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=27.03 GB):   9%|▊         | 5/58 [00:01<00:17,  3.09it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=27.03 GB):  10%|█         | 6/58 [00:02<00:17,  2.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.84 GB):  10%|█         | 6/58 [00:02<00:17,  2.97it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.84 GB):  12%|█▏        | 7/58 [00:02<00:16,  3.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=26.02 GB):  12%|█▏        | 7/58 [00:02<00:16,  3.03it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=26.02 GB):  14%|█▍        | 8/58 [00:02<00:16,  3.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.03 GB):  14%|█▍        | 8/58 [00:02<00:16,  3.00it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.03 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.10 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.12it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=26.10 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=27.02 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.20it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=27.02 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=26.16 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.31it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=26.16 GB):  21%|██        | 12/58 [00:04<00:13,  3.53it/s]Capturing num tokens (num_tokens=3072 avail_mem=26.24 GB):  21%|██        | 12/58 [00:04<00:13,  3.53it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=26.24 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=27.03 GB):  22%|██▏       | 13/58 [00:04<00:12,  3.72it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=27.03 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.30 GB):  24%|██▍       | 14/58 [00:04<00:11,  3.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.30 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.37 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.38it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=26.37 GB):  28%|██▊       | 16/58 [00:04<00:10,  4.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=27.03 GB):  28%|██▊       | 16/58 [00:04<00:10,  4.03it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=27.03 GB):  29%|██▉       | 17/58 [00:05<00:10,  3.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.44 GB):  29%|██▉       | 17/58 [00:05<00:10,  3.95it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=26.44 GB):  31%|███       | 18/58 [00:05<00:09,  4.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.51 GB):  31%|███       | 18/58 [00:05<00:09,  4.16it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=26.51 GB):  33%|███▎      | 19/58 [00:05<00:09,  4.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.54 GB):  33%|███▎      | 19/58 [00:05<00:09,  4.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.54 GB):  34%|███▍      | 20/58 [00:05<00:07,  5.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=27.00 GB):  34%|███▍      | 20/58 [00:05<00:07,  5.05it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=27.00 GB):  36%|███▌      | 21/58 [00:05<00:06,  5.46it/s]Capturing num tokens (num_tokens=960 avail_mem=27.03 GB):  36%|███▌      | 21/58 [00:05<00:06,  5.46it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=27.03 GB):  38%|███▊      | 22/58 [00:06<00:07,  5.02it/s]Capturing num tokens (num_tokens=896 avail_mem=36.10 GB):  38%|███▊      | 22/58 [00:06<00:07,  5.02it/s]Capturing num tokens (num_tokens=832 avail_mem=35.67 GB):  38%|███▊      | 22/58 [00:06<00:07,  5.02it/s]Capturing num tokens (num_tokens=832 avail_mem=35.67 GB):  41%|████▏     | 24/58 [00:06<00:05,  6.67it/s]Capturing num tokens (num_tokens=768 avail_mem=36.09 GB):  41%|████▏     | 24/58 [00:06<00:05,  6.67it/s]

    Capturing num tokens (num_tokens=704 avail_mem=35.69 GB):  41%|████▏     | 24/58 [00:06<00:05,  6.67it/s]Capturing num tokens (num_tokens=704 avail_mem=35.69 GB):  45%|████▍     | 26/58 [00:06<00:04,  8.00it/s]Capturing num tokens (num_tokens=640 avail_mem=36.09 GB):  45%|████▍     | 26/58 [00:06<00:04,  8.00it/s]Capturing num tokens (num_tokens=576 avail_mem=36.08 GB):  45%|████▍     | 26/58 [00:06<00:04,  8.00it/s]

    Capturing num tokens (num_tokens=576 avail_mem=36.08 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.99it/s]Capturing num tokens (num_tokens=512 avail_mem=35.74 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.99it/s]Capturing num tokens (num_tokens=480 avail_mem=36.07 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.99it/s]Capturing num tokens (num_tokens=480 avail_mem=36.07 GB):  52%|█████▏    | 30/58 [00:06<00:02, 10.10it/s]Capturing num tokens (num_tokens=448 avail_mem=36.08 GB):  52%|█████▏    | 30/58 [00:06<00:02, 10.10it/s]

    Capturing num tokens (num_tokens=416 avail_mem=35.80 GB):  52%|█████▏    | 30/58 [00:06<00:02, 10.10it/s]Capturing num tokens (num_tokens=416 avail_mem=35.80 GB):  55%|█████▌    | 32/58 [00:06<00:02, 11.39it/s]Capturing num tokens (num_tokens=384 avail_mem=36.07 GB):  55%|█████▌    | 32/58 [00:06<00:02, 11.39it/s]Capturing num tokens (num_tokens=352 avail_mem=36.06 GB):  55%|█████▌    | 32/58 [00:07<00:02, 11.39it/s]Capturing num tokens (num_tokens=352 avail_mem=36.06 GB):  59%|█████▊    | 34/58 [00:07<00:01, 12.29it/s]Capturing num tokens (num_tokens=320 avail_mem=35.85 GB):  59%|█████▊    | 34/58 [00:07<00:01, 12.29it/s]

    Capturing num tokens (num_tokens=288 avail_mem=36.05 GB):  59%|█████▊    | 34/58 [00:07<00:01, 12.29it/s]Capturing num tokens (num_tokens=288 avail_mem=36.05 GB):  62%|██████▏   | 36/58 [00:07<00:01, 13.97it/s]Capturing num tokens (num_tokens=256 avail_mem=36.04 GB):  62%|██████▏   | 36/58 [00:07<00:01, 13.97it/s]Capturing num tokens (num_tokens=240 avail_mem=36.04 GB):  62%|██████▏   | 36/58 [00:07<00:01, 13.97it/s]Capturing num tokens (num_tokens=240 avail_mem=36.04 GB):  66%|██████▌   | 38/58 [00:07<00:01, 15.29it/s]Capturing num tokens (num_tokens=224 avail_mem=35.91 GB):  66%|██████▌   | 38/58 [00:07<00:01, 15.29it/s]

    Capturing num tokens (num_tokens=208 avail_mem=35.91 GB):  66%|██████▌   | 38/58 [00:07<00:01, 15.29it/s]Capturing num tokens (num_tokens=192 avail_mem=36.02 GB):  66%|██████▌   | 38/58 [00:07<00:01, 15.29it/s]Capturing num tokens (num_tokens=192 avail_mem=36.02 GB):  71%|███████   | 41/58 [00:07<00:00, 17.25it/s]Capturing num tokens (num_tokens=176 avail_mem=36.01 GB):  71%|███████   | 41/58 [00:07<00:00, 17.25it/s]Capturing num tokens (num_tokens=160 avail_mem=36.01 GB):  71%|███████   | 41/58 [00:07<00:00, 17.25it/s]Capturing num tokens (num_tokens=144 avail_mem=36.00 GB):  71%|███████   | 41/58 [00:07<00:00, 17.25it/s]

    Capturing num tokens (num_tokens=144 avail_mem=36.00 GB):  76%|███████▌  | 44/58 [00:07<00:00, 18.63it/s]Capturing num tokens (num_tokens=128 avail_mem=36.01 GB):  76%|███████▌  | 44/58 [00:07<00:00, 18.63it/s]Capturing num tokens (num_tokens=112 avail_mem=35.98 GB):  76%|███████▌  | 44/58 [00:07<00:00, 18.63it/s]Capturing num tokens (num_tokens=96 avail_mem=35.99 GB):  76%|███████▌  | 44/58 [00:07<00:00, 18.63it/s] Capturing num tokens (num_tokens=96 avail_mem=35.99 GB):  81%|████████  | 47/58 [00:07<00:00, 20.30it/s]Capturing num tokens (num_tokens=80 avail_mem=35.93 GB):  81%|████████  | 47/58 [00:07<00:00, 20.30it/s]Capturing num tokens (num_tokens=64 avail_mem=35.93 GB):  81%|████████  | 47/58 [00:07<00:00, 20.30it/s]

    Capturing num tokens (num_tokens=48 avail_mem=35.97 GB):  81%|████████  | 47/58 [00:07<00:00, 20.30it/s]Capturing num tokens (num_tokens=48 avail_mem=35.97 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.23it/s]Capturing num tokens (num_tokens=32 avail_mem=35.96 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.23it/s]Capturing num tokens (num_tokens=28 avail_mem=35.95 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.23it/s]Capturing num tokens (num_tokens=24 avail_mem=35.95 GB):  86%|████████▌ | 50/58 [00:07<00:00, 21.23it/s]Capturing num tokens (num_tokens=24 avail_mem=35.95 GB):  91%|█████████▏| 53/58 [00:07<00:00, 22.70it/s]Capturing num tokens (num_tokens=20 avail_mem=35.94 GB):  91%|█████████▏| 53/58 [00:07<00:00, 22.70it/s]Capturing num tokens (num_tokens=16 avail_mem=35.93 GB):  91%|█████████▏| 53/58 [00:07<00:00, 22.70it/s]

    Capturing num tokens (num_tokens=12 avail_mem=35.93 GB):  91%|█████████▏| 53/58 [00:08<00:00, 22.70it/s]Capturing num tokens (num_tokens=12 avail_mem=35.93 GB):  97%|█████████▋| 56/58 [00:08<00:00, 23.84it/s]Capturing num tokens (num_tokens=8 avail_mem=35.92 GB):  97%|█████████▋| 56/58 [00:08<00:00, 23.84it/s] Capturing num tokens (num_tokens=4 avail_mem=35.91 GB):  97%|█████████▋| 56/58 [00:08<00:00, 23.84it/s]Capturing num tokens (num_tokens=4 avail_mem=35.91 GB): 100%|██████████| 58/58 [00:08<00:00,  7.15it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably double-check that. <br><br>Wait, I think I heard somewhere that Paris has a population over 21 million. Maybe 21.6 million? I'm not sure if that's accurate. I should look up the latest data to confirm. Also, I wonder if the population includes just the city proper or the entire metropolitan area. I think sometimes population counts include the broader area, so that might be a consideration. <br><br>I should make sure to present the information clearly in JSON format, as the user requested. So, the key would be "capital" with the value "Paris," and another key for "population." I'll need to include the number, maybe with a note about whether it's an approximate figure or the most recent data. <br><br>I'm a bit confused about whether the population figure I have is up to date. I think the population can change over time due to births, deaths, and migration. So, it's important to mention that the figure is approximate or based on the latest available data. <br><br>Also, I should consider the units. Is it in thousands, millions, or just a plain number? I think it's best to present it as a number without units, just the raw count. <br><br>Putting it all together, I'll structure the JSON with the keys "capital" and "population," and include the population as 21.6 million. I'll add a comment or note that the population figure is approximate. That should cover everything the user asked for.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21600000}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably double-check that. <br><br>Wait, I think I heard somewhere that Paris has a population over 21 million. Maybe 21.6 million? I'm not sure if that's accurate. I should look up the latest data to confirm. Also, I wonder if the population includes just the city proper or the entire metropolitan area. I think sometimes population counts include the broader area, so that might be a consideration. <br><br>I should make sure to present the information clearly in JSON format, as the user requested. So, the key would be "capital" with the value "Paris," and another key for "population." I'll need to include the number, maybe with a note about whether it's an approximate figure or the most recent data. <br><br>I'm a bit confused about whether the population figure I have is up to date. I think the population can change over time due to births, deaths, and migration. So, it's important to mention that the figure is approximate or based on the latest available data. <br><br>Also, I should consider the units. Is it in thousands, millions, or just a plain number? I think it's best to present it as a number without units, just the raw count. <br><br>Putting it all together, I'll structure the JSON with the keys "capital" and "population," and include the population as 21.6 million. I'll add a comment or note that the population figure is approximate. That should cover everything the user asked for.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21600000}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population. I think it's a big city, maybe around 3 million? I remember hearing that Paris is one of the most populous cities in Europe, but I'm not certain about the exact number. Maybe I should check some sources or think about recent growth. I think the population has been increasing over the years, so perhaps it's now over 3.5 million? I'm a bit confused because sometimes I hear different numbers, so I should make sure. Maybe I can recall that Paris has a metropolitan area that's much larger, but the city proper is around 3.5 million. I think I'll go with that for now, but I'm not 100% sure. I should probably double-check this information to be accurate.<br><br><br>content: Paris is the capital of France</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light." People go there for museums, landmarks like the Eiffel Tower, and it's a cultural hub. But is it the capital?<br><br>Wait, I think the capital is the official seat of government, right? So maybe Paris is both the capital and the most famous city. But I'm not entirely certain. I recall that some countries have their capital in a different city than their main tourist attraction. For example, I think Brazil's capital is not Rio de Janeiro, which is more famous. So maybe France is like that too.<br><br>I should try to remember any specific information. I think the French government declares Paris as the capital. Yeah, that sounds right. I also remember that the Eiffel Tower is in Paris, and it's a symbol of the country. So if Paris is the capital, then that makes sense. But I'm a bit confused because sometimes people say "the capital of France is Paris," but I want to make sure I'm not mixing it up with other countries.<br><br>Let me think about other capitals I know. For example, Germany's capital is Berlin, Italy's is Rome, Spain's is Madrid. So France's capital should be a major city in the north, maybe. Paris is in the north, so that fits. I think I've heard it said that Paris is the capital, so I'm pretty confident now. Yeah, I think I'm right.<br><br><br>content: Paris</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the current date and time in New York and the weather there. Let me break this down step by step.<br><br>First, I need to figure out which functions to use. The user mentioned two functions: get_current_weather and get_current_date. I should use both.<br><br>Starting with the date, I'll use get_current_date. The required parameter is timezone, which is 'America/New_York'. I'll structure the function call as <function=date>{{"timezone": "America/New_York"}}</function>.<br><br>Next, for the weather, I'll use get_current_weather. The parameters needed are city, state, and unit. The city is New York, the state is NY, and the unit should be Fahrenheit since the user didn't specify. So the function call will be <function=weather>{{"city": "New York", "state": "NY", "unit": "fahrenheit"}}</function>.<br><br>I should make sure each function call is on its own line and follows the specified format. Also, I need to include the sources in the response. Using brave_search for both functions makes sense since they're not available, but I should note that as a fallback if needed.<br><br>Putting it all together, I'll send the two function calls with their respective parameters and add the sources at the end to explain how the information was obtained.<br><br><br>content: <function[date]{{<br>  timezone": "America/New_York"<br>}}}</function>  <br><function=get_current_weather>{  <br>  "city": "New York",  <br>  "state": "NY",  <br>  "unit": "fahrenheit"  <br>}</function>  <br><br>Sources:  <br>- Function calls made using brave_search  <br>- get_current_weather and get_current_date functions</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s straightforward. \n\nNext, I need to find the population. I remember that Paris is a major city, so its population is quite large. I think it\'s over 3 million, but I\'m not exactly sure of the exact number. Maybe I should double-check that. \n\nWait, I recall that the population figure can vary depending on the source and the year. The user didn\'t specify a particular year, so I should probably go with the most recent estimate. I believe the population is around 3,500,000 as of 2023. \n\nNow, I need to structure this information into a JSON format. JSON typically uses key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" since the user mentioned France. \n\nI should make sure the keys are in English to keep it clear. The city is Paris, the population is 3,500,000, and the country is France. I\'ll format this into a JSON object, ensuring proper syntax with commas and quotation marks. \n\nI also need to present this in a way that\'s easy to read, so I\'ll put each key on a new line. That way, the user can quickly see the information without confusion. \n\nI wonder if the user needs more details, like the exact current population or additional statistics. But since they only asked for the capital and population, I\'ll stick to that. \n\nLastly, I\'ll make sure the JSON is valid by checking the syntax. No trailing commas, proper use of braces, and correct quotation marks. That should cover everything the user needs.\n</think>{\n  "name": "Paris",\n  "population": 3500000\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4710, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 11, 773, 1181, 7042, 374, 5008, 3460, 13, 358, 1744, 432, 594, 916, 220, 18, 3526, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 10696, 358, 1265, 1990, 15934, 429, 13, 4710, 14190, 11, 358, 19091, 429, 279, 7042, 7071, 646, 13289, 11649, 389, 279, 2530, 323, 279, 1042, 13, 576, 1196, 3207, 944, 13837, 264, 3953, 1042, 11, 773, 358, 1265, 4658, 728, 448, 279, 1429, 3213, 16045, 13, 358, 4411, 279, 7042, 374, 2163, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 4710, 7039, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 11136, 5711, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 2474, 279, 1196, 9733, 9625, 13, 4710, 40, 1265, 1281, 2704, 279, 6894, 525, 304, 6364, 311, 2506, 432, 2797, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 13, 358, 3278, 3561, 419, 1119, 264, 4718, 1633, 11, 22573, 6169, 19482, 448, 76602, 323, 54231, 15423, 13, 4710, 40, 1083, 1184, 311, 3042, 419, 304, 264, 1616, 429, 594, 4135, 311, 1349, 11, 773, 358, 3278, 2182, 1817, 1376, 389, 264, 501, 1555, 13, 2938, 1616, 11, 279, 1196, 646, 6157, 1490, 279, 1995, 2041, 21340, 13, 4710, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 4734, 1482, 7042, 476, 5107, 13142, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 6722, 323, 7042, 11, 358, 3278, 9214, 311, 429, 13, 4710, 80486, 11, 358, 3278, 1281, 2704, 279, 4718, 374, 2697, 553, 13295, 279, 19482, 13, 2308, 27748, 76602, 11, 6169, 990, 315, 59191, 11, 323, 4396, 54231, 15423, 13, 2938, 1265, 3421, 4297, 279, 1196, 3880, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 18, 20, 15, 15, 15, 15, 15, 198, 92, 151643], 'meta_info': {'id': 'da741060864745a99a5a460655745f7d', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 412, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.841605253983289, 'response_sent_to_client_ts': 1773649061.0906665}}



<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that's straightforward. <br><br>Next, I need to find the population. I remember that Paris is a major city, so its population is quite large. I think it's over 3 million, but I'm not exactly sure of the exact number. Maybe I should double-check that. <br><br>Wait, I recall that the population figure can vary depending on the source and the year. The user didn't specify a particular year, so I should probably go with the most recent estimate. I believe the population is around 3,500,000 as of 2023. <br><br>Now, I need to structure this information into a JSON format. JSON typically uses key-value pairs, so I'll create an object with keys like "city", "population", and maybe "country" since the user mentioned France. <br><br>I should make sure the keys are in English to keep it clear. The city is Paris, the population is 3,500,000, and the country is France. I'll format this into a JSON object, ensuring proper syntax with commas and quotation marks. <br><br>I also need to present this in a way that's easy to read, so I'll put each key on a new line. That way, the user can quickly see the information without confusion. <br><br>I wonder if the user needs more details, like the exact current population or additional statistics. But since they only asked for the capital and population, I'll stick to that. <br><br>Lastly, I'll make sure the JSON is valid by checking the syntax. No trailing commas, proper use of braces, and correct quotation marks. That should cover everything the user needs.<br><br><br>content: {<br>  "name": "Paris",<br>  "population": 3500000<br>}</strong>


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see... Yes, according to the latest data, the population is approximately 2,174,300 as of 2023. That seems accurate.\n\nNext, I need to structure this information into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and "country". The city is Paris, the population is the number I found, and the country is France.\n\nI should make sure the JSON syntax is correct. Each key should be in quotes, and the values as well. The entire structure should be enclosed in curly braces. I\'ll format it properly to avoid any syntax errors.\n\nPutting it all together, the JSON object will have the city, population, and country. I\'ll present this to the user, making sure it\'s clear and easy to understand. I don\'t think the user needs anything more detailed, so this should suffice.\n\nI should also consider if the user might need additional information, like the area or age distribution, but since they only asked for population, I\'ll stick to that. Maybe they\'ll ask for more details later, but for now, this response should be helpful.\n</think>{\n  "name": "Paris",\n  "population": 2174300\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 1112, 7414, 11, 4092, 311, 279, 5535, 821, 11, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 13382, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 3263, 576, 3283, 374, 12095, 11, 279, 7042, 374, 279, 1372, 358, 1730, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 8886, 1376, 1265, 387, 304, 17194, 11, 323, 279, 2750, 438, 1632, 13, 576, 4453, 5944, 1265, 387, 43810, 304, 68103, 59191, 13, 358, 3278, 3561, 432, 10277, 311, 5648, 894, 19482, 5975, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 686, 614, 279, 3283, 11, 7042, 11, 323, 3146, 13, 358, 3278, 3042, 419, 311, 279, 1196, 11, 3259, 2704, 432, 594, 2797, 323, 4135, 311, 3535, 13, 358, 1513, 944, 1744, 279, 1196, 3880, 4113, 803, 11682, 11, 773, 419, 1265, 76156, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 5107, 1995, 11, 1075, 279, 3082, 476, 4231, 7982, 11, 714, 2474, 807, 1172, 4588, 369, 7042, 11, 358, 3278, 9214, 311, 429, 13, 10696, 807, 3278, 2548, 369, 803, 3565, 2937, 11, 714, 369, 1431, 11, 419, 2033, 1265, 387, 10950, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 198, 92, 151643], 'meta_info': {'id': 'cec552a72a114e9d82bce39bf6756101', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 384, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.661581705790013, 'response_sent_to_client_ts': 1773649063.7622936}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '2b9d94115fe84861856f04d936c5915a', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.4915705192834139, 'response_sent_to_client_ts': 1773649064.284443}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '371844d731c142188a67d04350e3e62a', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.49150886898860335, 'response_sent_to_client_ts': 1773649064.2844555}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '2dac39d0c6014961872f13620e636681', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.4914693757891655, 'response_sent_to_client_ts': 1773649064.2844596}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '58598bde5b264aec9e92b410fb8b776c', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 20.44439019775018, 'response_sent_to_client_ts': 1773649084.7362182}}


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


<strong style='color: #00008B;'>{'text': 'Okay, I need to figure out how to respond to this user\'s request. They asked for the information and population of the capital of France in JSON format. \n\nAlright, so first, I know that the capital of France is Paris. That\'s a given. Now, they want the population. I\'m not exactly up to date on the latest numbers, but I think I can find the approximate figure. From what I remember, the population of Paris is around 2 million people. But I should double-check that to make sure it\'s accurate.\n\nWait, let me think. I recall that the metro area of Paris is much larger, maybe over 12 million? But the user specifically asked for the capital, so I think it\'s just the city proper, not the metro. That would make sense because sometimes people refer to just the city when talking about the capital.\n\nOkay, so structuring this in JSON. The user wants it in JSON format, so I should create a key-value pair where the key is "capital" and the value is another object containing the city name and population. That way, it\'s organized and clear.\n\nPutting it all together, I\'ll write something like: \n\n{\n  "capital": {\n    "city": "Paris",\n    "population": 2000000\n  }\n}\n\nHmm, but I should make sure that the population number is correct. Maybe it\'s 2,092,307 as of 2022, but I\'m not certain. If the user needs the most current data, they might want to check a reliable source later. But for now, providing a recent estimate should be sufficient.\n\nAlright, I think that\'s all I need to do for this request. Just present it in JSON format with the necessary details about the capital and its population.\n</think>\n\n```json\n{\n  "capital": {\n    "city": "Paris",\n    "population": 2092307\n  }\n}\n```', 'output_ids': [32313, 11, 358, 1184, 311, 7071, 700, 1246, 311, 5889, 311, 419, 1196, 594, 1681, 13, 2379, 4588, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 4710, 71486, 11, 773, 1156, 11, 358, 1414, 429, 279, 6722, 315, 9625, 374, 12095, 13, 2938, 594, 264, 2661, 13, 4695, 11, 807, 1366, 279, 7042, 13, 358, 2776, 537, 6896, 705, 311, 2400, 389, 279, 5535, 5109, 11, 714, 358, 1744, 358, 646, 1477, 279, 44868, 7071, 13, 5542, 1128, 358, 6099, 11, 279, 7042, 315, 12095, 374, 2163, 220, 17, 3526, 1251, 13, 1988, 358, 1265, 1990, 15934, 429, 311, 1281, 2704, 432, 594, 13382, 382, 14190, 11, 1077, 752, 1744, 13, 358, 19091, 429, 279, 33482, 3082, 315, 12095, 374, 1753, 8131, 11, 7196, 916, 220, 16, 17, 3526, 30, 1988, 279, 1196, 11689, 4588, 369, 279, 6722, 11, 773, 358, 1744, 432, 594, 1101, 279, 3283, 6169, 11, 537, 279, 33482, 13, 2938, 1035, 1281, 5530, 1576, 7025, 1251, 8300, 311, 1101, 279, 3283, 979, 7404, 911, 279, 6722, 382, 32313, 11, 773, 2036, 1677, 419, 304, 4718, 13, 576, 1196, 6801, 432, 304, 4718, 3561, 11, 773, 358, 1265, 1855, 264, 1376, 19083, 6716, 1380, 279, 1376, 374, 330, 65063, 1, 323, 279, 897, 374, 2441, 1633, 8482, 279, 3283, 829, 323, 7042, 13, 2938, 1616, 11, 432, 594, 16645, 323, 2797, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 2494, 1075, 25, 4710, 515, 220, 330, 65063, 788, 341, 262, 330, 8926, 788, 330, 59604, 756, 262, 330, 44441, 788, 220, 17, 15, 15, 15, 15, 15, 15, 198, 220, 456, 630, 80022, 11, 714, 358, 1265, 1281, 2704, 429, 279, 7042, 1372, 374, 4396, 13, 10696, 432, 594, 220, 17, 11, 15, 24, 17, 11, 18, 15, 22, 438, 315, 220, 17, 15, 17, 17, 11, 714, 358, 2776, 537, 3654, 13, 1416, 279, 1196, 3880, 279, 1429, 1482, 821, 11, 807, 2578, 1366, 311, 1779, 264, 14720, 2530, 2937, 13, 1988, 369, 1431, 11, 8241, 264, 3213, 16045, 1265, 387, 14016, 382, 71486, 11, 358, 1744, 429, 594, 678, 358, 1184, 311, 653, 369, 419, 1681, 13, 4599, 3042, 432, 304, 4718, 3561, 448, 279, 5871, 3565, 911, 279, 6722, 323, 1181, 7042, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 65063, 788, 341, 262, 330, 8926, 788, 330, 59604, 756, 262, 330, 44441, 788, 220, 17, 15, 24, 17, 18, 15, 22, 198, 220, 456, 532, 73594, 151643], 'meta_info': {'id': 'efb4151f0d1c43d4802827cbef1a4ad0', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 412, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 3.117778095882386, 'response_sent_to_client_ts': 1773649087.8678281}}</strong>



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

    [2026-03-16 08:18:10] INFO server_args.py:2154: Attention backend not specified. Use fa3 backend by default.


    [2026-03-16 08:18:10] INFO server_args.py:3311: Set soft_watchdog_timeout since in CI


    [2026-03-16 08:18:10] INFO engine.py:177: server_args=ServerArgs(model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, incremental_streaming_output=False, enable_streaming_session=False, random_seed=774838913, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser='deepseek-r1', tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.44s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.37s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.38s/it]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:01,  3.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:01,  3.18s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:34,  1.58it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:34,  1.58it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.16it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.16it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.51it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.51it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.29it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.29it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.18it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.18it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.18it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.85it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.85it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.85it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.36it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.36it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.36it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.79it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.79it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.79it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 14.74it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 14.74it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 14.74it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 14.74it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 14.74it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.18it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.18it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.18it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.18it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.18it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.18it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 25.57it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 25.57it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 25.57it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 25.57it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 24.31it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 24.31it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 24.31it/s]

    Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 24.31it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 24.24it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 24.24it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 24.24it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 24.24it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 24.87it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 24.87it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:06<00:00, 24.87it/s]

    Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:06<00:00, 24.87it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:06<00:00, 24.83it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:06<00:00, 24.83it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:06<00:00, 24.83it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 24.83it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 25.20it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 25.20it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 25.20it/s]

    Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 25.20it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 25.86it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 25.86it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 25.86it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 25.86it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:06<00:00, 25.93it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:06<00:00, 25.93it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:06<00:00, 25.93it/s]

    Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:06<00:00, 25.93it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:06<00:00, 26.99it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:06<00:00, 26.99it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:06<00:00, 26.99it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 26.99it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 26.99it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 29.79it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 29.79it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=28.95 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=28.95 GB):   2%|▏         | 1/58 [00:00<00:38,  1.49it/s]Capturing num tokens (num_tokens=7680 avail_mem=28.92 GB):   2%|▏         | 1/58 [00:00<00:38,  1.49it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=28.92 GB):   3%|▎         | 2/58 [00:01<00:33,  1.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=28.92 GB):   3%|▎         | 2/58 [00:01<00:33,  1.65it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=28.92 GB):   5%|▌         | 3/58 [00:01<00:29,  1.90it/s]Capturing num tokens (num_tokens=6656 avail_mem=28.92 GB):   5%|▌         | 3/58 [00:01<00:29,  1.90it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=28.92 GB):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=28.93 GB):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=28.93 GB):   9%|▊         | 5/58 [00:02<00:28,  1.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=28.93 GB):   9%|▊         | 5/58 [00:02<00:28,  1.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=28.93 GB):  10%|█         | 6/58 [00:02<00:21,  2.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=28.93 GB):  10%|█         | 6/58 [00:02<00:21,  2.41it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=28.93 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=27.57 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.55it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=27.57 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=27.57 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.66it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=27.57 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=27.58 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.81it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=27.58 GB):  17%|█▋        | 10/58 [00:04<00:16,  2.96it/s]Capturing num tokens (num_tokens=3584 avail_mem=27.57 GB):  17%|█▋        | 10/58 [00:04<00:16,  2.96it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=27.57 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=27.57 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.14it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=27.57 GB):  21%|██        | 12/58 [00:04<00:13,  3.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=27.57 GB):  21%|██        | 12/58 [00:04<00:13,  3.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=27.57 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=27.57 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.14it/s]Capturing num tokens (num_tokens=2560 avail_mem=27.57 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.14it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=27.57 GB):  26%|██▌       | 15/58 [00:05<00:09,  4.51it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.22 GB):  26%|██▌       | 15/58 [00:05<00:09,  4.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=45.22 GB):  26%|██▌       | 15/58 [00:05<00:09,  4.51it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=45.22 GB):  29%|██▉       | 17/58 [00:05<00:10,  4.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.22 GB):  29%|██▉       | 17/58 [00:05<00:10,  4.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.22 GB):  29%|██▉       | 17/58 [00:05<00:10,  4.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.22 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.66it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.22 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=45.22 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.66it/s]

    Capturing num tokens (num_tokens=960 avail_mem=45.22 GB):  33%|███▎      | 19/58 [00:06<00:06,  5.66it/s] Capturing num tokens (num_tokens=960 avail_mem=45.22 GB):  38%|███▊      | 22/58 [00:06<00:04,  8.48it/s]Capturing num tokens (num_tokens=896 avail_mem=45.21 GB):  38%|███▊      | 22/58 [00:06<00:04,  8.48it/s]Capturing num tokens (num_tokens=832 avail_mem=45.21 GB):  38%|███▊      | 22/58 [00:06<00:04,  8.48it/s]Capturing num tokens (num_tokens=768 avail_mem=45.20 GB):  38%|███▊      | 22/58 [00:06<00:04,  8.48it/s]Capturing num tokens (num_tokens=768 avail_mem=45.20 GB):  43%|████▎     | 25/58 [00:06<00:02, 11.49it/s]Capturing num tokens (num_tokens=704 avail_mem=45.20 GB):  43%|████▎     | 25/58 [00:06<00:02, 11.49it/s]Capturing num tokens (num_tokens=640 avail_mem=45.20 GB):  43%|████▎     | 25/58 [00:06<00:02, 11.49it/s]

    Capturing num tokens (num_tokens=576 avail_mem=45.19 GB):  43%|████▎     | 25/58 [00:06<00:02, 11.49it/s]Capturing num tokens (num_tokens=576 avail_mem=45.19 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.66it/s]Capturing num tokens (num_tokens=512 avail_mem=45.19 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.66it/s]Capturing num tokens (num_tokens=480 avail_mem=45.19 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.66it/s]Capturing num tokens (num_tokens=448 avail_mem=45.18 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.66it/s]Capturing num tokens (num_tokens=416 avail_mem=45.18 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.66it/s]Capturing num tokens (num_tokens=416 avail_mem=45.18 GB):  55%|█████▌    | 32/58 [00:06<00:01, 18.74it/s]Capturing num tokens (num_tokens=384 avail_mem=45.18 GB):  55%|█████▌    | 32/58 [00:06<00:01, 18.74it/s]Capturing num tokens (num_tokens=352 avail_mem=45.17 GB):  55%|█████▌    | 32/58 [00:06<00:01, 18.74it/s]

    Capturing num tokens (num_tokens=320 avail_mem=45.17 GB):  55%|█████▌    | 32/58 [00:06<00:01, 18.74it/s]Capturing num tokens (num_tokens=288 avail_mem=45.16 GB):  55%|█████▌    | 32/58 [00:06<00:01, 18.74it/s]Capturing num tokens (num_tokens=288 avail_mem=45.16 GB):  62%|██████▏   | 36/58 [00:06<00:00, 22.53it/s]Capturing num tokens (num_tokens=256 avail_mem=45.16 GB):  62%|██████▏   | 36/58 [00:06<00:00, 22.53it/s]Capturing num tokens (num_tokens=240 avail_mem=45.16 GB):  62%|██████▏   | 36/58 [00:06<00:00, 22.53it/s]Capturing num tokens (num_tokens=224 avail_mem=45.15 GB):  62%|██████▏   | 36/58 [00:06<00:00, 22.53it/s]Capturing num tokens (num_tokens=208 avail_mem=45.15 GB):  62%|██████▏   | 36/58 [00:06<00:00, 22.53it/s]Capturing num tokens (num_tokens=208 avail_mem=45.15 GB):  69%|██████▉   | 40/58 [00:06<00:00, 26.08it/s]Capturing num tokens (num_tokens=192 avail_mem=45.15 GB):  69%|██████▉   | 40/58 [00:06<00:00, 26.08it/s]Capturing num tokens (num_tokens=176 avail_mem=45.14 GB):  69%|██████▉   | 40/58 [00:06<00:00, 26.08it/s]

    Capturing num tokens (num_tokens=160 avail_mem=43.98 GB):  69%|██████▉   | 40/58 [00:06<00:00, 26.08it/s]Capturing num tokens (num_tokens=144 avail_mem=43.98 GB):  69%|██████▉   | 40/58 [00:06<00:00, 26.08it/s]

    Capturing num tokens (num_tokens=144 avail_mem=43.98 GB):  76%|███████▌  | 44/58 [00:06<00:00, 19.06it/s]Capturing num tokens (num_tokens=128 avail_mem=43.99 GB):  76%|███████▌  | 44/58 [00:06<00:00, 19.06it/s]Capturing num tokens (num_tokens=112 avail_mem=43.99 GB):  76%|███████▌  | 44/58 [00:07<00:00, 19.06it/s]

    Capturing num tokens (num_tokens=96 avail_mem=43.98 GB):  76%|███████▌  | 44/58 [00:07<00:00, 19.06it/s] Capturing num tokens (num_tokens=96 avail_mem=43.98 GB):  81%|████████  | 47/58 [00:07<00:00, 15.10it/s]Capturing num tokens (num_tokens=80 avail_mem=43.97 GB):  81%|████████  | 47/58 [00:07<00:00, 15.10it/s]Capturing num tokens (num_tokens=64 avail_mem=43.97 GB):  81%|████████  | 47/58 [00:07<00:00, 15.10it/s]

    Capturing num tokens (num_tokens=48 avail_mem=43.97 GB):  81%|████████  | 47/58 [00:07<00:00, 15.10it/s]Capturing num tokens (num_tokens=48 avail_mem=43.97 GB):  86%|████████▌ | 50/58 [00:07<00:00, 13.37it/s]Capturing num tokens (num_tokens=32 avail_mem=43.97 GB):  86%|████████▌ | 50/58 [00:07<00:00, 13.37it/s]

    Capturing num tokens (num_tokens=28 avail_mem=43.96 GB):  86%|████████▌ | 50/58 [00:07<00:00, 13.37it/s]Capturing num tokens (num_tokens=28 avail_mem=43.96 GB):  90%|████████▉ | 52/58 [00:07<00:00, 12.67it/s]Capturing num tokens (num_tokens=24 avail_mem=43.96 GB):  90%|████████▉ | 52/58 [00:07<00:00, 12.67it/s]

    Capturing num tokens (num_tokens=20 avail_mem=43.96 GB):  90%|████████▉ | 52/58 [00:07<00:00, 12.67it/s]Capturing num tokens (num_tokens=20 avail_mem=43.96 GB):  93%|█████████▎| 54/58 [00:07<00:00, 11.40it/s]Capturing num tokens (num_tokens=16 avail_mem=43.95 GB):  93%|█████████▎| 54/58 [00:07<00:00, 11.40it/s]

    Capturing num tokens (num_tokens=12 avail_mem=43.95 GB):  93%|█████████▎| 54/58 [00:08<00:00, 11.40it/s]Capturing num tokens (num_tokens=12 avail_mem=43.95 GB):  97%|█████████▋| 56/58 [00:08<00:00, 10.40it/s]Capturing num tokens (num_tokens=8 avail_mem=43.94 GB):  97%|█████████▋| 56/58 [00:08<00:00, 10.40it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=43.94 GB):  97%|█████████▋| 56/58 [00:08<00:00, 10.40it/s]Capturing num tokens (num_tokens=4 avail_mem=43.94 GB): 100%|██████████| 58/58 [00:08<00:00,  9.98it/s]Capturing num tokens (num_tokens=4 avail_mem=43.94 GB): 100%|██████████| 58/58 [00:08<00:00,  6.88it/s]


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
    Generated text: London is the capital of England


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
    
    Generated text: Alright, I need to figure out how to respond to the user's request. They asked for the information and population of the capital of France in JSON format. 
    
    First, I should identify what the capital of France is. I know it's Paris. Next, I need to find the most recent population data for Paris. The population can vary each year, so I should look for the latest estimate. 
    
    I recall that as of the 2020 census, Paris had a population around 2.1 million. It's important to note that this number might have changed a bit since then, but it's a reliable figure. 
    
    Now, I need to structure this information into a JSON format. The user specified that they want the information and population in JSON, so I'll create an object with appropriate keys. 
    
    I'll include "capital" as the key for the name and "population" for the number. It's good practice to include a "description" key to provide context, explaining that the population figure is as of the 2020 census. 
    
    I should ensure that the JSON is properly formatted, with commas in the right places and the data enclosed in quotation marks. Also, I'll make sure there are no syntax errors, like missing quotes or incorrect commas, to avoid any issues when the user uses the JSON.
    
    Finally, I'll present the JSON in a clear and concise manner, as per the user's request. I don't need to add any extra information beyond what's asked unless it's necessary for completeness, which it isn't here. 
    
    So, putting it all together, the JSON should have the key-value pairs for the capital and population, along with a brief description to clarify the population figure's source and timeframe.
    </think>
    
    ```json
    {
      "capital": "Paris",
      "population": 2161000,
      "description": "The population figure is as of the 2020 census."
    }
    ```



```python
llm.shutdown()
```
