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

    [2026-03-03 08:12:09] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-03 08:12:09] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-03 08:12:09] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-03 08:12:14] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 08:12:14] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 08:12:14] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /root/actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:34: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-03 08:12:16] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.
    [2026-03-03 08:12:16] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 08:12:22] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 08:12:22] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 08:12:22] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-03 08:12:22] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 08:12:22] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 08:12:22] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-03 08:12:27] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-03 08:12:27] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-03 08:12:27] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.49s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.42s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.43s/it]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=106.97 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=106.97 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.83it/s]Capturing batches (bs=2 avail_mem=106.91 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.83it/s]Capturing batches (bs=1 avail_mem=106.90 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.83it/s]Capturing batches (bs=1 avail_mem=106.90 GB): 100%|██████████| 3/3 [00:00<00:00, 11.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:02,  3.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:02,  3.21s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:31,  1.63s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:31,  1.63s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:54,  1.02it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:54,  1.02it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:40,  1.33it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:40,  1.33it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:26,  1.99it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:26,  1.99it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:21,  2.41it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:21,  2.41it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:18,  2.71it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:18,  2.71it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:15,  3.16it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:15,  3.16it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:13,  3.50it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:11,  3.92it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:11,  3.92it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:10,  4.39it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:10,  4.39it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:09,  4.71it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:09,  4.71it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:08,  5.33it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:08,  5.33it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:07,  5.85it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:07,  5.85it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:06,  6.28it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:06,  6.28it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:06<00:06,  6.28it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:06<00:05,  7.91it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:06<00:05,  7.91it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:07<00:05,  7.91it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:04,  9.00it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:04,  9.00it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:04,  9.00it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:04,  9.00it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:02, 11.79it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:02, 11.79it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:02, 11.79it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:07<00:02, 13.21it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:07<00:02, 13.21it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:07<00:02, 13.21it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:07<00:02, 13.21it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:07<00:01, 16.83it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:07<00:01, 16.83it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:07<00:01, 16.83it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:07<00:01, 16.83it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:07<00:01, 18.92it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:07<00:01, 18.92it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:07<00:01, 18.92it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:07<00:01, 18.92it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:07<00:01, 18.92it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 23.27it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 23.27it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 23.27it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 23.27it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:07<00:00, 24.80it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:07<00:00, 24.80it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:07<00:00, 24.80it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:07<00:00, 24.80it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:07<00:00, 24.80it/s]

    Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:07<00:00, 24.80it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:08<00:00, 28.45it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:08<00:00, 28.45it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:08<00:00, 28.45it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:08<00:00, 28.45it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:08<00:00, 28.45it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:08<00:00, 28.45it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:08<00:00, 32.13it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:08<00:00, 32.13it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:08<00:00, 32.13it/s]

    Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:08<00:00, 32.13it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:08<00:00, 32.13it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:08<00:00, 32.13it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:08<00:00, 35.28it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:08<00:00, 35.28it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:08<00:00, 35.28it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:08<00:00, 35.28it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:08<00:00, 35.28it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:08<00:00, 35.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00, 38.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=88.61 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=88.61 GB):   2%|▏         | 1/58 [00:00<00:25,  2.22it/s]Capturing num tokens (num_tokens=7680 avail_mem=88.80 GB):   2%|▏         | 1/58 [00:00<00:25,  2.22it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=88.80 GB):   3%|▎         | 2/58 [00:00<00:23,  2.43it/s]Capturing num tokens (num_tokens=7168 avail_mem=88.78 GB):   3%|▎         | 2/58 [00:00<00:23,  2.43it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=88.78 GB):   5%|▌         | 3/58 [00:01<00:20,  2.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=88.76 GB):   5%|▌         | 3/58 [00:01<00:20,  2.65it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=88.76 GB):   7%|▋         | 4/58 [00:01<00:21,  2.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=88.70 GB):   7%|▋         | 4/58 [00:01<00:21,  2.51it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=88.70 GB):   9%|▊         | 5/58 [00:01<00:18,  2.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=88.74 GB):   9%|▊         | 5/58 [00:01<00:18,  2.85it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=88.74 GB):  10%|█         | 6/58 [00:02<00:16,  3.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=88.73 GB):  10%|█         | 6/58 [00:02<00:16,  3.21it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=88.73 GB):  12%|█▏        | 7/58 [00:02<00:14,  3.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=88.71 GB):  12%|█▏        | 7/58 [00:02<00:14,  3.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=88.71 GB):  14%|█▍        | 8/58 [00:02<00:12,  4.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=88.68 GB):  14%|█▍        | 8/58 [00:02<00:12,  4.03it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=88.68 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=88.70 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=88.70 GB):  17%|█▋        | 10/58 [00:02<00:09,  4.89it/s]Capturing num tokens (num_tokens=3584 avail_mem=88.69 GB):  17%|█▋        | 10/58 [00:02<00:09,  4.89it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=88.69 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=88.68 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=88.68 GB):  21%|██        | 12/58 [00:03<00:07,  5.85it/s]Capturing num tokens (num_tokens=3072 avail_mem=88.67 GB):  21%|██        | 12/58 [00:03<00:07,  5.85it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=88.67 GB):  22%|██▏       | 13/58 [00:03<00:07,  6.33it/s]Capturing num tokens (num_tokens=2816 avail_mem=88.66 GB):  22%|██▏       | 13/58 [00:03<00:07,  6.33it/s]Capturing num tokens (num_tokens=2816 avail_mem=88.66 GB):  24%|██▍       | 14/58 [00:03<00:06,  6.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=88.64 GB):  24%|██▍       | 14/58 [00:03<00:06,  6.99it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=88.64 GB):  26%|██▌       | 15/58 [00:03<00:05,  7.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=88.63 GB):  26%|██▌       | 15/58 [00:03<00:05,  7.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=88.62 GB):  26%|██▌       | 15/58 [00:03<00:05,  7.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=88.62 GB):  29%|██▉       | 17/58 [00:03<00:04,  8.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=88.61 GB):  29%|██▉       | 17/58 [00:03<00:04,  8.88it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=88.60 GB):  29%|██▉       | 17/58 [00:03<00:04,  8.88it/s]Capturing num tokens (num_tokens=1536 avail_mem=88.60 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.43it/s]Capturing num tokens (num_tokens=1280 avail_mem=88.59 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.43it/s]Capturing num tokens (num_tokens=1024 avail_mem=88.59 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.43it/s]Capturing num tokens (num_tokens=960 avail_mem=88.59 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.43it/s] Capturing num tokens (num_tokens=960 avail_mem=88.59 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.00it/s]Capturing num tokens (num_tokens=896 avail_mem=88.58 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.00it/s]

    Capturing num tokens (num_tokens=832 avail_mem=88.57 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.00it/s]Capturing num tokens (num_tokens=768 avail_mem=88.57 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.00it/s]Capturing num tokens (num_tokens=768 avail_mem=88.57 GB):  43%|████▎     | 25/58 [00:04<00:01, 17.12it/s]Capturing num tokens (num_tokens=704 avail_mem=88.56 GB):  43%|████▎     | 25/58 [00:04<00:01, 17.12it/s]Capturing num tokens (num_tokens=640 avail_mem=88.56 GB):  43%|████▎     | 25/58 [00:04<00:01, 17.12it/s]Capturing num tokens (num_tokens=576 avail_mem=88.55 GB):  43%|████▎     | 25/58 [00:04<00:01, 17.12it/s]Capturing num tokens (num_tokens=576 avail_mem=88.55 GB):  48%|████▊     | 28/58 [00:04<00:01, 19.60it/s]Capturing num tokens (num_tokens=512 avail_mem=88.55 GB):  48%|████▊     | 28/58 [00:04<00:01, 19.60it/s]

    Capturing num tokens (num_tokens=480 avail_mem=88.54 GB):  48%|████▊     | 28/58 [00:04<00:01, 19.60it/s]Capturing num tokens (num_tokens=448 avail_mem=88.54 GB):  48%|████▊     | 28/58 [00:04<00:01, 19.60it/s]Capturing num tokens (num_tokens=416 avail_mem=88.53 GB):  48%|████▊     | 28/58 [00:04<00:01, 19.60it/s]Capturing num tokens (num_tokens=416 avail_mem=88.53 GB):  55%|█████▌    | 32/58 [00:04<00:01, 23.17it/s]Capturing num tokens (num_tokens=384 avail_mem=88.53 GB):  55%|█████▌    | 32/58 [00:04<00:01, 23.17it/s]Capturing num tokens (num_tokens=352 avail_mem=88.52 GB):  55%|█████▌    | 32/58 [00:04<00:01, 23.17it/s]Capturing num tokens (num_tokens=320 avail_mem=88.51 GB):  55%|█████▌    | 32/58 [00:04<00:01, 23.17it/s]Capturing num tokens (num_tokens=288 avail_mem=88.51 GB):  55%|█████▌    | 32/58 [00:04<00:01, 23.17it/s]

    Capturing num tokens (num_tokens=288 avail_mem=88.51 GB):  62%|██████▏   | 36/58 [00:04<00:00, 26.06it/s]Capturing num tokens (num_tokens=256 avail_mem=88.50 GB):  62%|██████▏   | 36/58 [00:04<00:00, 26.06it/s]Capturing num tokens (num_tokens=240 avail_mem=88.50 GB):  62%|██████▏   | 36/58 [00:04<00:00, 26.06it/s]Capturing num tokens (num_tokens=224 avail_mem=88.49 GB):  62%|██████▏   | 36/58 [00:04<00:00, 26.06it/s]Capturing num tokens (num_tokens=208 avail_mem=88.48 GB):  62%|██████▏   | 36/58 [00:04<00:00, 26.06it/s]Capturing num tokens (num_tokens=208 avail_mem=88.48 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.57it/s]Capturing num tokens (num_tokens=192 avail_mem=88.48 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.57it/s]Capturing num tokens (num_tokens=176 avail_mem=88.48 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.57it/s]Capturing num tokens (num_tokens=160 avail_mem=88.47 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.57it/s]Capturing num tokens (num_tokens=144 avail_mem=88.46 GB):  69%|██████▉   | 40/58 [00:04<00:00, 28.57it/s]

    Capturing num tokens (num_tokens=144 avail_mem=88.46 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.38it/s]Capturing num tokens (num_tokens=128 avail_mem=88.47 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.38it/s]Capturing num tokens (num_tokens=112 avail_mem=88.47 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.38it/s]Capturing num tokens (num_tokens=96 avail_mem=88.46 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.38it/s] Capturing num tokens (num_tokens=80 avail_mem=88.45 GB):  76%|███████▌  | 44/58 [00:04<00:00, 30.38it/s]Capturing num tokens (num_tokens=80 avail_mem=88.45 GB):  83%|████████▎ | 48/58 [00:04<00:00, 31.71it/s]Capturing num tokens (num_tokens=64 avail_mem=88.45 GB):  83%|████████▎ | 48/58 [00:04<00:00, 31.71it/s]Capturing num tokens (num_tokens=48 avail_mem=88.44 GB):  83%|████████▎ | 48/58 [00:04<00:00, 31.71it/s]Capturing num tokens (num_tokens=32 avail_mem=88.37 GB):  83%|████████▎ | 48/58 [00:04<00:00, 31.71it/s]

    Capturing num tokens (num_tokens=28 avail_mem=88.37 GB):  83%|████████▎ | 48/58 [00:04<00:00, 31.71it/s]Capturing num tokens (num_tokens=28 avail_mem=88.37 GB):  90%|████████▉ | 52/58 [00:04<00:00, 32.66it/s]Capturing num tokens (num_tokens=24 avail_mem=88.36 GB):  90%|████████▉ | 52/58 [00:04<00:00, 32.66it/s]Capturing num tokens (num_tokens=20 avail_mem=88.36 GB):  90%|████████▉ | 52/58 [00:04<00:00, 32.66it/s]Capturing num tokens (num_tokens=16 avail_mem=88.35 GB):  90%|████████▉ | 52/58 [00:04<00:00, 32.66it/s]Capturing num tokens (num_tokens=12 avail_mem=88.34 GB):  90%|████████▉ | 52/58 [00:04<00:00, 32.66it/s]Capturing num tokens (num_tokens=12 avail_mem=88.34 GB):  97%|█████████▋| 56/58 [00:04<00:00, 33.26it/s]Capturing num tokens (num_tokens=8 avail_mem=88.34 GB):  97%|█████████▋| 56/58 [00:04<00:00, 33.26it/s] Capturing num tokens (num_tokens=4 avail_mem=88.33 GB):  97%|█████████▋| 56/58 [00:04<00:00, 33.26it/s]

    Capturing num tokens (num_tokens=4 avail_mem=88.33 GB): 100%|██████████| 58/58 [00:05<00:00, 11.58it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should check if that's correct.<br><br>Wait, I think the population might have changed a bit over the years. I recall reading somewhere that Paris has grown a lot, especially with the influx of people moving there for work. But I'm not sure if it's exactly 21 million or maybe a bit more. I should look up the latest data to confirm.<br><br>I also wonder if the population figure includes just the city proper or the entire metropolitan area. Sometimes, people talk about the metro area, which can be much larger. But I think the question is specifically about the capital, so probably just the city limits. <br><br>Another thing to consider is that population figures can vary depending on the source. Some might cite estimates from government agencies, while others might use more recent surveys. I should make sure to use a reliable source to get the most accurate number.<br><br>I'm pretty confident that Paris is the capital, so I don't need to worry about that part. But for the population, I should double-check. Maybe I can recall that in recent years, Paris has been growing steadily. I think it's somewhere between 20 and 22 million. Let me try to remember any specific numbers or events that might have affected the population, like the COVID-19 pandemic. I think that had a temporary impact, but the city has been recovering since then.<br><br>So, putting it all together, I'm going to say that the capital of France is Paris, and its population is approximately 21 million people. I'll present this information in a JSON format as requested.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21000000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should check if that's correct.<br><br>Wait, I think the population might have changed a bit over the years. I recall reading somewhere that Paris has grown a lot, especially with the influx of people moving there for work. But I'm not sure if it's exactly 21 million or maybe a bit more. I should look up the latest data to confirm.<br><br>I also wonder if the population figure includes just the city proper or the entire metropolitan area. Sometimes, people talk about the metro area, which can be much larger. But I think the question is specifically about the capital, so probably just the city limits. <br><br>Another thing to consider is that population figures can vary depending on the source. Some might cite estimates from government agencies, while others might use more recent surveys. I should make sure to use a reliable source to get the most accurate number.<br><br>I'm pretty confident that Paris is the capital, so I don't need to worry about that part. But for the population, I should double-check. Maybe I can recall that in recent years, Paris has been growing steadily. I think it's somewhere between 20 and 22 million. Let me try to remember any specific numbers or events that might have affected the population, like the COVID-19 pandemic. I think that had a temporary impact, but the city has been recovering since then.<br><br>So, putting it all together, I'm going to say that the capital of France is Paris, and its population is approximately 21 million people. I'll present this information in a JSON format as requested.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21000000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not entirely sure about the population number. I think it's a big city, so maybe around 3 million? But I'm not certain. I should probably double-check that. Maybe I can recall that Paris is one of the largest cities in Europe, so 3.5 million sounds about right. I don't think it's more than that because I've heard it's a major tourist attraction but not the largest in the world. So, I'll go with Paris having a population of approximately 3.5 million.<br><br><br>content: London is the capital of France</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning a bit about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light" because of the famous Eiffel Tower. But is it the capital? I think so, but I'm not entirely certain. <br><br>Wait, I also recall that there's another city called Lyon. Isn't that the capital? No, I think I'm mixing things up. Lyon is definitely a significant city in France, known for its gastronomy and being the second-largest city. But I'm pretty sure Paris is the capital. <br><br>Let me try to remember any other capitals I know. London is the capital of the UK, Rome is Italy, Beijing is China, and Tokyo is Japan. So, for France, it's probably Paris. I think I've heard it a lot in news and media. Also, the Eiffel Tower is in Paris, which is a symbol of the country, so that reinforces the idea that Paris is the capital. <br><br>I don't remember any major political figures from Lyon; they're more from France's historical past. Maybe some people confuse Lyon with the capital, but I'm pretty confident it's Paris. I'll go with Paris as the capital of France.<br><br><br>content: Paris</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out how to get the current date and time in New York and the weather there. Let me think about the functions available. There's get_current_weather and get_current_date. <br><br>First, I should use get_current_date because that's for the date and time. The parameters require a timezone, so I'll put 'America/New_York' there. That should give me the current date and time in New York.<br><br>Next, for the weather, I'll use get_current_weather. I need the city, state, and unit. The city is New York, the state is NY, and I want the temperature in Fahrenheit. So I'll structure that function call accordingly.<br><br>I should make sure each function is called separately, as per the instructions. So I'll send two separate messages: one for the date and time, and another for the weather. Each with the correct parameters in the specified format.<br><br>I also need to remember to include the sources in the response. So when I call get_current_date, I'll mention the function and parameters, and similarly for get_current_weather. That way, anyone reading the response knows where the information came from.<br><br>Wait, but the user asked for both in one message. Hmm, but the instructions say to call one function at a time. So maybe I should split them into two separate messages. Or perhaps combine them in a single message, but each function call on its own line. Let me check the example given. Oh right, in the example, each function call is on its own line with the parameters. So I'll do the same here, making two separate function calls.<br><br>So first, I'll call get_current_date with the timezone parameter set to 'America/New_York'. Then, I'll call get_current_weather with city 'New York', state 'NY', and unit 'fahrenheit'. Each with the correct JSON parameters enclosed in the start and end tags.<br><br>I think that covers everything. I just need to structure each function call properly, ensuring the parameters are correctly formatted as JSON objects within the function syntax.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, the JSON will have the city name, population, and a concise description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid any errors.\n\nFinally, I\'ll present the JSON to the user, ensuring it\'s clear and well-structured. I\'ll double-check the population number to confirm it\'s up to date and accurate. That should fulfill the user\'s request effectively.\n</think>{\n\n"name": "Paris",\n"population": 2174300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 279, 4718, 686, 614, 279, 3283, 829, 11, 7042, 11, 323, 264, 63594, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 894, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 22573, 432, 594, 2797, 323, 1632, 12, 51143, 13, 358, 3278, 1990, 15934, 279, 7042, 1372, 311, 7683, 432, 594, 705, 311, 2400, 323, 13382, 13, 2938, 1265, 20423, 279, 1196, 594, 1681, 13444, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '1233376eab5a435c8fa32e247c4d8f23', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 10.489866913994774, 'response_sent_to_client_ts': 1772525593.9140563}}



<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.<br><br>First, I need to identify the capital of France. I know that Paris is the capital, so that's straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I'm not sure of the exact number. I think it's around 2 million, but I should verify that.<br><br>I'll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.<br><br>Next, I need to structure this information into a JSON format. The user wants a JSON, so I'll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.<br><br>I should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it's a count of people.<br><br>Putting it all together, the JSON will have the city name, population, and a concise description. I'll make sure the syntax is correct, with commas and brackets in the right places to avoid any errors.<br><br>Finally, I'll present the JSON to the user, ensuring it's clear and well-structured. I'll double-check the population number to confirm it's up to date and accurate. That should fulfill the user's request effectively.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 2174300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000</strong>


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user is a student working on a project or maybe a developer integrating this data into an app. Either way, providing the information in a structured JSON format is likely helpful for their needs. I should ensure the data is accurate to avoid any confusion.\n\nAlso, considering the population figure, I should note that it\'s an approximate number because population figures can change over time due to various factors like births, deaths, and migration. It\'s good to be transparent about that so the user knows the data is current as of a specific point in time.\n\nIn summary, I need to provide a clear, concise JSON response with the correct capital and population, making sure the syntax is correct and the information is accurate.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 374, 264, 5458, 3238, 389, 264, 2390, 476, 7196, 264, 15754, 53852, 419, 821, 1119, 458, 906, 13, 20988, 1616, 11, 8241, 279, 1995, 304, 264, 32930, 4718, 3561, 374, 4363, 10950, 369, 862, 3880, 13, 358, 1265, 5978, 279, 821, 374, 13382, 311, 5648, 894, 21340, 382, 13394, 11, 12831, 279, 7042, 7071, 11, 358, 1265, 5185, 429, 432, 594, 458, 44868, 1372, 1576, 7042, 12396, 646, 2297, 916, 882, 4152, 311, 5257, 9363, 1075, 65232, 11, 16375, 11, 323, 11906, 13, 1084, 594, 1661, 311, 387, 17821, 911, 429, 773, 279, 1196, 8788, 279, 821, 374, 1482, 438, 315, 264, 3151, 1459, 304, 882, 382, 641, 12126, 11, 358, 1184, 311, 3410, 264, 2797, 11, 63594, 4718, 2033, 448, 279, 4396, 6722, 323, 7042, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 1995, 374, 13382, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '5a65fd6baf114711ab35ea3644b24528', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 456, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.2314969550352544, 'response_sent_to_client_ts': 1772525596.164387}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '1c92494fed3541beb3b2e1b369c8e683', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.12247660104185343, 'response_sent_to_client_ts': 1772525596.311943}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '876324c2e0974ae2970539c31fefdec3', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.12241305503994226, 'response_sent_to_client_ts': 1772525596.3119504}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '20f460a3f4c845b1a93ff0a263f4ffad', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.12236480507999659, 'response_sent_to_client_ts': 1772525596.3119545}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\)', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8], 'meta_info': {'id': '5caa329ff0f2446fa71ea025aa66a507', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 17.475335899041966, 'response_sent_to_client_ts': 1772525613.795732}}


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


<strong style='color: #00008B;'>{'text': 'Alright, so the user is asking for the information and population of Paris, the capital of France. They want this in JSON format. Let me start by recalling that Paris is indeed the capital, so I don\'t need to double-check that.\n\nI should gather the necessary details. Population is a key point, so I need an up-to-date figure. The population has been changing over the years, so I should find the latest data. I remember reading that the population was around 2.17 million in 2020, but it\'s been growing since then. Maybe it\'s approximately 2.2 million now. I\'ll need to verify if that\'s accurate.\n\nNext, I should list some key facts about Paris. I know it\'s a major city with rich history, being the political and cultural hub. Its location at the confluence of the Seine and Eure rivers gives it a Regards for its landmarks like the Eiffel Tower, Notre-Dame, and the Louvre. Including these makes the information more informative.\n\nFactoids are helpful too. Maybe something about its historical significance, like being the first permanent seat of the French Republic, which makes sense because it was declared there in 18世纪. Also, mentioning the average number of children per family can add a demographic angle, showing it\'s a family-friendly place.\n\nI should structure the JSON properly. It\'s standard to have an "Info" key followed by "Population" with two keys: "Value" and "Description". For the population, I can split it into current and historical data, each with their values and descriptions.\n\nI need to ensure the JSON syntax is correct. Proper commas and colons are essential. Also, the text within quotes should be accurate and concise. \n\nWait, in the population section, the user wants both the general figure and a historical one. So I\'ll add an array under "Population" with each item having "Value" and "Description". That way, it\'s clear and organized.\n\nLet me double-check the population numbers from a reliable source. Maybe the latest census or a recent UN database. I think the population is around 2.2 million as of 2023. For the past decade, it\'s been steadily increasing, so mentioning that is important.\n\nAlso, the factoids should be accurate. Paris is named after the gestational period of Mary, so that\'s correct. The number of children per family reflects a stable, family-oriented environment, which is true.\n\nPutting it all together, the JSON structure should be clean and easy to parse. I can format it neatly, using line breaks for readability without affecting functionality.\n\nFinally, I\'ll offer to provide more information or adjust the data if needed, showing willingness to help further.\n</think>\n\nCertainly! Here\'s the information about the capital of France (Paris) in JSON format:\n\n```json\n{\n  "Info": {\n    "Capital": "Paris",\n    " литератур Das capital of France is Paris",\n    "Hauptstadt": "Paris",\n    "Landschaft": "Paris is the most visited city in the world and a global cultural, economic, and political center.",\n    "Museums": "Some of the world\'s most famous museums, such as the Louvre Museum and the Paris Observatory, are located in the city.",\n    "Kulture": "Paris is known for its rich history, famous landmarks like the Eiffel Tower and Notre-Dame, and its vibrant cultural scene."\n  },\n  "Bevölkerung": {\n    "Jahresbevölkerung": {\n      "Wert": 2173000,\n      "Beschreibung": "Population in 2020 (approximate)"\n    },\n    "Geschichtliche Bevölkerung": {\n      "Wert": 1791000,\n      "Beschreibung": "Historical population (approximate)"\n    }\n  },\n  "Fakten": {\n    "Geschichtliche Bedeutung": "Paris is the birthplace of modern democracy and was the first permanent seat of the French Republic.",\n    "Landschaft": "Paris is known for its riverine setting, with the Seine and Eure rivers flowing through the city.",\n    "Kultur": "Paris is renowned for its festivals, cuisine, and artistic movements such as Impressionism."\n  },\n  "Fakten": {\n    "Durchschnittliche Anzahl Kinder pro Famille": {\n      "Wert": 2.1,\n      "Beschreibung": "The average number of children per family in Paris"\n    }\n  }\n}\n```\n\nLet me know if you\'d like more details or adjustments!', 'output_ids': [71486, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 12095, 11, 279, 6722, 315, 9625, 13, 2379, 1366, 419, 304, 4718, 3561, 13, 6771, 752, 1191, 553, 88646, 429, 12095, 374, 12824, 279, 6722, 11, 773, 358, 1513, 944, 1184, 311, 1990, 15934, 429, 382, 40, 1265, 9567, 279, 5871, 3565, 13, 39529, 374, 264, 1376, 1459, 11, 773, 358, 1184, 458, 705, 4686, 18413, 7071, 13, 576, 7042, 702, 1012, 10018, 916, 279, 1635, 11, 773, 358, 1265, 1477, 279, 5535, 821, 13, 358, 6099, 5290, 429, 279, 7042, 572, 2163, 220, 17, 13, 16, 22, 3526, 304, 220, 17, 15, 17, 15, 11, 714, 432, 594, 1012, 7826, 2474, 1221, 13, 10696, 432, 594, 13187, 220, 17, 13, 17, 3526, 1431, 13, 358, 3278, 1184, 311, 10146, 421, 429, 594, 13382, 382, 5847, 11, 358, 1265, 1140, 1045, 1376, 13064, 911, 12095, 13, 358, 1414, 432, 594, 264, 3598, 3283, 448, 9080, 3840, 11, 1660, 279, 4948, 323, 12752, 18719, 13, 11445, 3728, 518, 279, 390, 40016, 315, 279, 1345, 482, 323, 468, 552, 35517, 6696, 432, 264, 91079, 369, 1181, 59924, 1075, 279, 468, 3092, 301, 21938, 11, 43464, 9420, 373, 11, 323, 279, 9729, 48506, 13, 55121, 1493, 3643, 279, 1995, 803, 38219, 382, 17417, 16960, 525, 10950, 2238, 13, 10696, 2494, 911, 1181, 13656, 25361, 11, 1075, 1660, 279, 1156, 15330, 10723, 315, 279, 8585, 5429, 11, 892, 3643, 5530, 1576, 432, 572, 14275, 1052, 304, 220, 16, 23, 101186, 13, 7281, 11, 44291, 279, 5461, 1372, 315, 2841, 817, 2997, 646, 912, 264, 37362, 9210, 11, 9027, 432, 594, 264, 2997, 21896, 1992, 382, 40, 1265, 5944, 279, 4718, 10277, 13, 1084, 594, 5297, 311, 614, 458, 330, 1731, 1, 1376, 8110, 553, 330, 53371, 1, 448, 1378, 6894, 25, 330, 1130, 1, 323, 330, 5009, 3263, 1752, 279, 7042, 11, 358, 646, 6718, 432, 1119, 1482, 323, 13656, 821, 11, 1817, 448, 862, 2750, 323, 27787, 382, 40, 1184, 311, 5978, 279, 4718, 19482, 374, 4396, 13, 64558, 76602, 323, 1375, 2382, 525, 7565, 13, 7281, 11, 279, 1467, 2878, 17194, 1265, 387, 13382, 323, 63594, 13, 4710, 14190, 11, 304, 279, 7042, 3772, 11, 279, 1196, 6801, 2176, 279, 4586, 7071, 323, 264, 13656, 825, 13, 2055, 358, 3278, 912, 458, 1334, 1212, 330, 53371, 1, 448, 1817, 1509, 3432, 330, 1130, 1, 323, 330, 5009, 3263, 2938, 1616, 11, 432, 594, 2797, 323, 16645, 382, 10061, 752, 1990, 15934, 279, 7042, 5109, 504, 264, 14720, 2530, 13, 10696, 279, 5535, 43602, 476, 264, 3213, 6643, 4625, 13, 358, 1744, 279, 7042, 374, 2163, 220, 17, 13, 17, 3526, 438, 315, 220, 17, 15, 17, 18, 13, 1752, 279, 3267, 13212, 11, 432, 594, 1012, 41971, 7703, 11, 773, 44291, 429, 374, 2989, 382, 13394, 11, 279, 2097, 16960, 1265, 387, 13382, 13, 12095, 374, 6941, 1283, 279, 12743, 1663, 4168, 315, 10244, 11, 773, 429, 594, 4396, 13, 576, 1372, 315, 2841, 817, 2997, 25963, 264, 15175, 11, 2997, 35085, 4573, 11, 892, 374, 830, 382, 97904, 432, 678, 3786, 11, 279, 4718, 5944, 1265, 387, 4240, 323, 4135, 311, 4715, 13, 358, 646, 3561, 432, 62166, 11, 1667, 1555, 18303, 369, 91494, 2041, 27887, 14944, 382, 23949, 11, 358, 3278, 3010, 311, 3410, 803, 1995, 476, 7500, 279, 821, 421, 4362, 11, 9027, 38275, 311, 1492, 4623, 624, 151649, 271, 95456, 0, 5692, 594, 279, 1995, 911, 279, 6722, 315, 9625, 320, 59604, 8, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 1731, 788, 341, 262, 330, 63593, 788, 330, 59604, 756, 262, 330, 140566, 19000, 6722, 315, 9625, 374, 12095, 756, 262, 330, 39, 51898, 44278, 788, 330, 59604, 756, 262, 330, 43, 2844, 20542, 788, 330, 59604, 374, 279, 1429, 11994, 3283, 304, 279, 1879, 323, 264, 3644, 12752, 11, 6955, 11, 323, 4948, 4126, 10346, 262, 330, 44, 810, 6237, 788, 330, 8373, 315, 279, 1879, 594, 1429, 11245, 50577, 11, 1741, 438, 279, 9729, 48506, 16328, 323, 279, 12095, 57874, 11, 525, 7407, 304, 279, 3283, 10346, 262, 330, 42, 10523, 788, 330, 59604, 374, 3881, 369, 1181, 9080, 3840, 11, 11245, 59924, 1075, 279, 468, 3092, 301, 21938, 323, 43464, 9420, 373, 11, 323, 1181, 32976, 12752, 6109, 10040, 220, 1153, 220, 330, 3430, 85, 49802, 7052, 2185, 788, 341, 262, 330, 41, 1466, 416, 1371, 85, 49802, 7052, 2185, 788, 341, 414, 330, 54, 529, 788, 220, 17, 16, 22, 18, 15, 15, 15, 345, 414, 330, 39726, 331, 75399, 788, 330, 53371, 304, 220, 17, 15, 17, 15, 320, 48053, 3426, 12954, 262, 1153, 262, 330, 38, 59378, 4865, 20005, 2823, 85, 49802, 7052, 2185, 788, 341, 414, 330, 54, 529, 788, 220, 16, 22, 24, 16, 15, 15, 15, 345, 414, 330, 39726, 331, 75399, 788, 330, 48983, 938, 7042, 320, 48053, 3426, 12954, 262, 456, 220, 1153, 220, 330, 37, 585, 1960, 788, 341, 262, 330, 38, 59378, 4865, 20005, 13093, 60053, 2185, 788, 330, 59604, 374, 279, 7194, 2007, 315, 6481, 19525, 323, 572, 279, 1156, 15330, 10723, 315, 279, 8585, 5429, 10346, 262, 330, 43, 2844, 20542, 788, 330, 59604, 374, 3881, 369, 1181, 14796, 482, 6243, 11, 448, 279, 1345, 482, 323, 468, 552, 35517, 35512, 1526, 279, 3283, 10346, 262, 330, 42, 76431, 788, 330, 59604, 374, 35948, 369, 1181, 44417, 11, 35005, 11, 323, 31592, 19029, 1741, 438, 14390, 11185, 2142, 10040, 220, 1153, 220, 330, 37, 585, 1960, 788, 341, 262, 330, 35, 2574, 82, 88718, 20005, 1527, 46728, 43999, 462, 33601, 4517, 788, 341, 414, 330, 54, 529, 788, 220, 17, 13, 16, 345, 414, 330, 39726, 331, 75399, 788, 330, 785, 5461, 1372, 315, 2841, 817, 2997, 304, 12095, 698, 262, 456, 220, 456, 532, 13874, 19324, 10061, 752, 1414, 421, 498, 4172, 1075, 803, 3565, 476, 32974, 0, 151643], 'meta_info': {'id': '6cb19043b7154d78aa5fa2a8bccb426b', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 973, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 5.502983503974974, 'response_sent_to_client_ts': 1772525619.3110147}}</strong>



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


    [2026-03-03 08:13:41] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.


    [2026-03-03 08:13:41] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 08:13:41] INFO engine.py:157: server_args=ServerArgs(model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.903, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=130639830, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser='deepseek-r1', tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.43s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.37s/it]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=102.19 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=102.19 GB):   5%|▌         | 1/20 [00:00<00:04,  4.66it/s]Capturing batches (bs=120 avail_mem=102.08 GB):   5%|▌         | 1/20 [00:00<00:04,  4.66it/s]Capturing batches (bs=112 avail_mem=102.07 GB):   5%|▌         | 1/20 [00:00<00:04,  4.66it/s]Capturing batches (bs=104 avail_mem=102.07 GB):   5%|▌         | 1/20 [00:00<00:04,  4.66it/s]Capturing batches (bs=104 avail_mem=102.07 GB):  20%|██        | 4/20 [00:00<00:01, 11.18it/s]Capturing batches (bs=96 avail_mem=102.06 GB):  20%|██        | 4/20 [00:00<00:01, 11.18it/s] 

    Capturing batches (bs=88 avail_mem=102.06 GB):  20%|██        | 4/20 [00:00<00:01, 11.18it/s]Capturing batches (bs=80 avail_mem=102.05 GB):  20%|██        | 4/20 [00:00<00:01, 11.18it/s]Capturing batches (bs=80 avail_mem=102.05 GB):  35%|███▌      | 7/20 [00:00<00:00, 15.44it/s]Capturing batches (bs=72 avail_mem=102.05 GB):  35%|███▌      | 7/20 [00:00<00:00, 15.44it/s]Capturing batches (bs=64 avail_mem=102.04 GB):  35%|███▌      | 7/20 [00:00<00:00, 15.44it/s]Capturing batches (bs=56 avail_mem=102.04 GB):  35%|███▌      | 7/20 [00:00<00:00, 15.44it/s]Capturing batches (bs=56 avail_mem=102.04 GB):  50%|█████     | 10/20 [00:00<00:00, 19.08it/s]Capturing batches (bs=48 avail_mem=102.03 GB):  50%|█████     | 10/20 [00:00<00:00, 19.08it/s]

    Capturing batches (bs=40 avail_mem=102.03 GB):  50%|█████     | 10/20 [00:00<00:00, 19.08it/s]Capturing batches (bs=32 avail_mem=102.02 GB):  50%|█████     | 10/20 [00:00<00:00, 19.08it/s]Capturing batches (bs=32 avail_mem=102.02 GB):  65%|██████▌   | 13/20 [00:00<00:00, 21.64it/s]Capturing batches (bs=24 avail_mem=102.01 GB):  65%|██████▌   | 13/20 [00:00<00:00, 21.64it/s]Capturing batches (bs=16 avail_mem=102.01 GB):  65%|██████▌   | 13/20 [00:00<00:00, 21.64it/s]Capturing batches (bs=12 avail_mem=102.00 GB):  65%|██████▌   | 13/20 [00:00<00:00, 21.64it/s]

    Capturing batches (bs=12 avail_mem=102.00 GB):  80%|████████  | 16/20 [00:00<00:00, 21.30it/s]Capturing batches (bs=8 avail_mem=102.00 GB):  80%|████████  | 16/20 [00:00<00:00, 21.30it/s] Capturing batches (bs=4 avail_mem=101.99 GB):  80%|████████  | 16/20 [00:00<00:00, 21.30it/s]Capturing batches (bs=2 avail_mem=101.99 GB):  80%|████████  | 16/20 [00:00<00:00, 21.30it/s]Capturing batches (bs=1 avail_mem=101.99 GB):  80%|████████  | 16/20 [00:00<00:00, 21.30it/s]Capturing batches (bs=1 avail_mem=101.99 GB): 100%|██████████| 20/20 [00:01<00:00, 25.09it/s]Capturing batches (bs=1 avail_mem=101.99 GB): 100%|██████████| 20/20 [00:01<00:00, 19.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:03,  3.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:03,  3.22s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:27,  1.57s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:27,  1.57s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.16it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.16it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.54it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.54it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.24it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.24it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.02it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.02it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:08,  5.88it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:08,  5.88it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:07,  6.70it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:07,  6.70it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:07,  6.70it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:05,  8.23it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:05,  8.23it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.23it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.14it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.14it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:04,  9.00it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:04,  9.00it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04,  9.15it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04,  9.15it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04,  9.15it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:03,  9.84it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:03,  9.84it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:03,  9.84it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:03, 11.19it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:03, 11.19it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:03, 11.19it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 21.86it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 21.86it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:06<00:01, 21.86it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:06<00:01, 21.86it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:06<00:01, 21.86it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:06<00:01, 21.86it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:06<00:01, 21.86it/s]

    Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:06<00:00, 31.39it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:06<00:00, 31.39it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:06<00:00, 31.39it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:06<00:00, 31.39it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:06<00:00, 31.39it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:06<00:00, 31.39it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:06<00:00, 31.39it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:06<00:00, 31.39it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:06<00:00, 31.39it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:06<00:00, 42.69it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:06<00:00, 42.69it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:06<00:00, 42.69it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:06<00:00, 42.69it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:06<00:00, 42.69it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:06<00:00, 42.69it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:06<00:00, 40.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.00it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=121.24 GB):   2%|▏         | 1/58 [00:00<00:16,  3.52it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.21 GB):   2%|▏         | 1/58 [00:00<00:16,  3.52it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=121.21 GB):   3%|▎         | 2/58 [00:00<00:15,  3.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.20 GB):   3%|▎         | 2/58 [00:00<00:15,  3.65it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=121.20 GB):   5%|▌         | 3/58 [00:00<00:14,  3.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.21 GB):   5%|▌         | 3/58 [00:00<00:14,  3.87it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=121.21 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.21 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=121.21 GB):   9%|▊         | 5/58 [00:01<00:12,  4.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.21 GB):   9%|▊         | 5/58 [00:01<00:12,  4.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.21 GB):  10%|█         | 6/58 [00:01<00:10,  4.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.21 GB):  10%|█         | 6/58 [00:01<00:10,  4.76it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=121.21 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.21 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.21 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.21 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.66it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=121.21 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.22 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.22 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.65it/s]Capturing num tokens (num_tokens=3584 avail_mem=121.21 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.65it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=121.21 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.00 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.00 GB):  21%|██        | 12/58 [00:02<00:06,  7.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=106.19 GB):  21%|██        | 12/58 [00:02<00:06,  7.65it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=106.19 GB):  21%|██        | 12/58 [00:02<00:06,  7.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=106.19 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.83it/s]Capturing num tokens (num_tokens=2560 avail_mem=106.19 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=106.19 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.83it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=106.19 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=106.18 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=106.18 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=106.18 GB):  31%|███       | 18/58 [00:02<00:03, 10.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=106.18 GB):  31%|███       | 18/58 [00:02<00:03, 10.48it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=106.18 GB):  31%|███       | 18/58 [00:02<00:03, 10.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=106.18 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=106.17 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.48it/s]Capturing num tokens (num_tokens=960 avail_mem=106.17 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.48it/s] Capturing num tokens (num_tokens=896 avail_mem=106.16 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.48it/s]Capturing num tokens (num_tokens=896 avail_mem=106.16 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.79it/s]Capturing num tokens (num_tokens=832 avail_mem=106.16 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.79it/s]

    Capturing num tokens (num_tokens=768 avail_mem=106.15 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.79it/s]Capturing num tokens (num_tokens=704 avail_mem=106.15 GB):  40%|███▉      | 23/58 [00:03<00:02, 14.79it/s]Capturing num tokens (num_tokens=704 avail_mem=106.15 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]Capturing num tokens (num_tokens=640 avail_mem=106.14 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]Capturing num tokens (num_tokens=576 avail_mem=106.14 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]

    Capturing num tokens (num_tokens=576 avail_mem=106.14 GB):  48%|████▊     | 28/58 [00:03<00:01, 16.88it/s]Capturing num tokens (num_tokens=512 avail_mem=106.13 GB):  48%|████▊     | 28/58 [00:03<00:01, 16.88it/s]Capturing num tokens (num_tokens=480 avail_mem=106.13 GB):  48%|████▊     | 28/58 [00:03<00:01, 16.88it/s]Capturing num tokens (num_tokens=480 avail_mem=106.13 GB):  52%|█████▏    | 30/58 [00:03<00:01, 17.33it/s]Capturing num tokens (num_tokens=448 avail_mem=106.12 GB):  52%|█████▏    | 30/58 [00:03<00:01, 17.33it/s]Capturing num tokens (num_tokens=416 avail_mem=106.12 GB):  52%|█████▏    | 30/58 [00:03<00:01, 17.33it/s]Capturing num tokens (num_tokens=384 avail_mem=106.11 GB):  52%|█████▏    | 30/58 [00:03<00:01, 17.33it/s]

    Capturing num tokens (num_tokens=352 avail_mem=106.10 GB):  52%|█████▏    | 30/58 [00:03<00:01, 17.33it/s]Capturing num tokens (num_tokens=352 avail_mem=106.10 GB):  59%|█████▊    | 34/58 [00:03<00:01, 21.45it/s]Capturing num tokens (num_tokens=320 avail_mem=106.10 GB):  59%|█████▊    | 34/58 [00:03<00:01, 21.45it/s]Capturing num tokens (num_tokens=288 avail_mem=106.09 GB):  59%|█████▊    | 34/58 [00:03<00:01, 21.45it/s]Capturing num tokens (num_tokens=256 avail_mem=106.09 GB):  59%|█████▊    | 34/58 [00:03<00:01, 21.45it/s]Capturing num tokens (num_tokens=240 avail_mem=106.08 GB):  59%|█████▊    | 34/58 [00:03<00:01, 21.45it/s]Capturing num tokens (num_tokens=240 avail_mem=106.08 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.10it/s]Capturing num tokens (num_tokens=224 avail_mem=106.08 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.10it/s]Capturing num tokens (num_tokens=208 avail_mem=106.07 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.10it/s]

    Capturing num tokens (num_tokens=192 avail_mem=106.07 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.10it/s]Capturing num tokens (num_tokens=176 avail_mem=106.06 GB):  66%|██████▌   | 38/58 [00:03<00:00, 25.10it/s]Capturing num tokens (num_tokens=176 avail_mem=106.06 GB):  72%|███████▏  | 42/58 [00:03<00:00, 28.01it/s]Capturing num tokens (num_tokens=160 avail_mem=106.05 GB):  72%|███████▏  | 42/58 [00:03<00:00, 28.01it/s]Capturing num tokens (num_tokens=144 avail_mem=106.05 GB):  72%|███████▏  | 42/58 [00:03<00:00, 28.01it/s]Capturing num tokens (num_tokens=128 avail_mem=106.06 GB):  72%|███████▏  | 42/58 [00:03<00:00, 28.01it/s]Capturing num tokens (num_tokens=112 avail_mem=106.05 GB):  72%|███████▏  | 42/58 [00:03<00:00, 28.01it/s]Capturing num tokens (num_tokens=112 avail_mem=106.05 GB):  79%|███████▉  | 46/58 [00:03<00:00, 30.01it/s]Capturing num tokens (num_tokens=96 avail_mem=106.04 GB):  79%|███████▉  | 46/58 [00:03<00:00, 30.01it/s] Capturing num tokens (num_tokens=80 avail_mem=106.04 GB):  79%|███████▉  | 46/58 [00:03<00:00, 30.01it/s]

    Capturing num tokens (num_tokens=64 avail_mem=106.03 GB):  79%|███████▉  | 46/58 [00:03<00:00, 30.01it/s]Capturing num tokens (num_tokens=48 avail_mem=106.03 GB):  79%|███████▉  | 46/58 [00:03<00:00, 30.01it/s]Capturing num tokens (num_tokens=48 avail_mem=106.03 GB):  86%|████████▌ | 50/58 [00:03<00:00, 31.46it/s]Capturing num tokens (num_tokens=32 avail_mem=106.02 GB):  86%|████████▌ | 50/58 [00:03<00:00, 31.46it/s]Capturing num tokens (num_tokens=28 avail_mem=106.02 GB):  86%|████████▌ | 50/58 [00:03<00:00, 31.46it/s]Capturing num tokens (num_tokens=24 avail_mem=106.01 GB):  86%|████████▌ | 50/58 [00:03<00:00, 31.46it/s]Capturing num tokens (num_tokens=20 avail_mem=106.01 GB):  86%|████████▌ | 50/58 [00:03<00:00, 31.46it/s]Capturing num tokens (num_tokens=20 avail_mem=106.01 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.57it/s]Capturing num tokens (num_tokens=16 avail_mem=106.00 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.57it/s]Capturing num tokens (num_tokens=12 avail_mem=105.99 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.57it/s]

    Capturing num tokens (num_tokens=8 avail_mem=105.99 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.57it/s] Capturing num tokens (num_tokens=4 avail_mem=105.98 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.57it/s]Capturing num tokens (num_tokens=4 avail_mem=105.98 GB): 100%|██████████| 58/58 [00:04<00:00, 33.43it/s]Capturing num tokens (num_tokens=4 avail_mem=105.98 GB): 100%|██████████| 58/58 [00:04<00:00, 14.07it/s]


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
    Generated text: London is the capital of England
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
    
    Generated text: Alright, so the user just asked me to provide the information and population of the capital of France in JSON format. Let me break this down.
    
    First, I need to figure out what exactly they're asking for. The capital of France is definitely Paris, right? So the main points are the name of the city, its population, and maybe some other relevant details. 
    
    They mentioned both the capital and Paris, so I should clarify that Paris is the capital, but sometimes people might confuse it with other cities in France. It's good to specify that Paris is the capital to avoid any confusion.
    
    Next, population data. I know that population numbers can change over time, so I should check the most recent estimate. I recall that as of 2023, Paris had around 2.2 million residents, but it's always best to verify this. Maybe I should note that the figure can vary based on sources and the exact time it's measured.
    
    Now, structuring this information into JSON. JSON requires key-value pairs, so I need to decide on the keys. "Name" and "Population" seem straightforward, but perhaps adding a "Location" field would make it clearer, specifying that it's a city in France.
    
    I should also format the number properly, maybe using commas for readability. So, 2,206,545 instead of 2206545.
    
    Putting it all together, the JSON should be simple and clear. I'll make sure to include comments if necessary, but the user didn't specify that, so keeping it straightforward is probably best.
    
    I should also consider if there's a need for more details, but since the user only asked for the capital and the population, sticking to those is best. I'll double-check the population number to ensure accuracy.
    
    Finally, I'll present the JSON, making sure it's correctly formatted and easy to read. That should fulfill the user's request effectively.
    </think>
    
    Here is the information about the capital of France in JSON format:
    
    ```json
    {
      "Name": "Paris",
      "Population": 2206545,
      "Location": "City of France"
    }
    ```
    
    This JSON object contains the name of the capital (Paris), its population, and its location as the "City of France."



```python
llm.shutdown()
```
