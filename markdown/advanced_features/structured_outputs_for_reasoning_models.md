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

    [2026-03-07 22:01:15] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-07 22:01:15] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-07 22:01:15] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-07 22:01:19] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-07 22:01:19] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-07 22:01:19] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-07 22:01:20] INFO server_args.py:2048: Attention backend not specified. Use fa3 backend by default.
    [2026-03-07 22:01:20] INFO server_args.py:3155: Set soft_watchdog_timeout since in CI


    [2026-03-07 22:01:25] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-07 22:01:25] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-07 22:01:25] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-07 22:01:25] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-07 22:01:25] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-07 22:01:25] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-07 22:01:32] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-07 22:01:32] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-07 22:01:32] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.15s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.41s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.37s/it]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=31.34 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=31.34 GB):  33%|███▎      | 1/3 [00:00<00:00,  3.49it/s]Capturing batches (bs=2 avail_mem=28.03 GB):  33%|███▎      | 1/3 [00:00<00:00,  3.49it/s]Capturing batches (bs=1 avail_mem=28.03 GB):  33%|███▎      | 1/3 [00:00<00:00,  3.49it/s]Capturing batches (bs=1 avail_mem=28.03 GB): 100%|██████████| 3/3 [00:00<00:00,  9.01it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:14,  1.33s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:14,  1.33s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:44,  1.22it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:44,  1.22it/s]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:30,  1.77it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:30,  1.77it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:22,  2.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:22,  2.40it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:20,  2.60it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:20,  2.60it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.66it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.66it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.90it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.90it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:13,  3.58it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:12,  3.88it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:12,  3.88it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:10,  4.25it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:10,  4.25it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:09,  4.60it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:09,  4.60it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:08,  5.03it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:08,  5.03it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:07,  5.42it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:07,  5.42it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:07,  5.94it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:07,  5.94it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:06,  6.64it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:06,  6.64it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:06<00:05,  7.20it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:06<00:05,  7.20it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:06<00:04,  7.84it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:06<00:04,  7.84it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:06<00:04,  7.84it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:06<00:03,  9.34it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:06<00:03,  9.34it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:06<00:03,  9.34it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:06<00:03, 11.04it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:06<00:03, 11.04it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:06<00:03, 11.04it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:06<00:02, 12.77it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:06<00:02, 12.77it/s]

    Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:06<00:02, 12.77it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:06<00:02, 14.22it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:06<00:02, 14.22it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:06<00:02, 14.22it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:06<00:02, 14.22it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:06<00:01, 16.76it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:06<00:01, 16.76it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:01, 16.76it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:07<00:01, 17.17it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:07<00:01, 17.17it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:07<00:01, 17.17it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:07<00:01, 17.17it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:01, 18.54it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:01, 18.54it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:01, 18.54it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:01, 18.54it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:07<00:01, 19.97it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:07<00:01, 19.97it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:07<00:01, 19.97it/s]

    Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:07<00:01, 19.97it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:07<00:00, 21.77it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:07<00:00, 21.77it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:07<00:00, 21.77it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:07<00:00, 21.77it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 23.20it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 23.20it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 23.20it/s]

    Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 23.20it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:07<00:00, 23.33it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:07<00:00, 23.33it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:07<00:00, 23.33it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:07<00:00, 23.33it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:07<00:00, 23.78it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:07<00:00, 23.78it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 23.78it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 23.78it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:07<00:00, 24.84it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:07<00:00, 24.84it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:07<00:00, 24.84it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:08<00:00, 24.84it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:08<00:00, 24.84it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:08<00:00, 28.36it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:08<00:00, 28.36it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=22.91 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=22.91 GB):   2%|▏         | 1/58 [00:00<00:37,  1.54it/s]Capturing num tokens (num_tokens=7680 avail_mem=41.99 GB):   2%|▏         | 1/58 [00:00<00:37,  1.54it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=41.99 GB):   3%|▎         | 2/58 [00:01<00:27,  2.02it/s]Capturing num tokens (num_tokens=7168 avail_mem=41.99 GB):   3%|▎         | 2/58 [00:01<00:27,  2.02it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=41.99 GB):   5%|▌         | 3/58 [00:01<00:23,  2.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=41.99 GB):   5%|▌         | 3/58 [00:01<00:23,  2.39it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=41.99 GB):   7%|▋         | 4/58 [00:01<00:18,  2.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.00 GB):   7%|▋         | 4/58 [00:01<00:18,  2.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.00 GB):   9%|▊         | 5/58 [00:01<00:16,  3.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=40.97 GB):   9%|▊         | 5/58 [00:01<00:16,  3.20it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=40.97 GB):  10%|█         | 6/58 [00:02<00:17,  2.95it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.97 GB):  10%|█         | 6/58 [00:02<00:17,  2.95it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=41.97 GB):  12%|█▏        | 7/58 [00:02<00:17,  2.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.16 GB):  12%|█▏        | 7/58 [00:02<00:17,  2.98it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=41.16 GB):  14%|█▍        | 8/58 [00:02<00:16,  3.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.99 GB):  14%|█▍        | 8/58 [00:02<00:16,  3.00it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=41.99 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.22 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.25it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=41.22 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.27 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.25it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=41.27 GB):  19%|█▉        | 11/58 [00:03<00:13,  3.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.28 GB):  19%|█▉        | 11/58 [00:03<00:13,  3.46it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=41.28 GB):  21%|██        | 12/58 [00:03<00:12,  3.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.99 GB):  21%|██        | 12/58 [00:03<00:12,  3.57it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=41.99 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.33 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.79it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=41.33 GB):  24%|██▍       | 14/58 [00:04<00:11,  4.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.00 GB):  24%|██▍       | 14/58 [00:04<00:11,  4.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.00 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.39 GB):  26%|██▌       | 15/58 [00:04<00:09,  4.32it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=41.39 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.63it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.00 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.63it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.00 GB):  29%|██▉       | 17/58 [00:04<00:08,  5.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.45 GB):  29%|██▉       | 17/58 [00:04<00:08,  5.05it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=41.45 GB):  31%|███       | 18/58 [00:05<00:07,  5.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.00 GB):  31%|███       | 18/58 [00:05<00:07,  5.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.00 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.50 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.81it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=41.50 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.01 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.35it/s]Capturing num tokens (num_tokens=960 avail_mem=41.56 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.35it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=41.56 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.43it/s]Capturing num tokens (num_tokens=896 avail_mem=42.00 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.43it/s]Capturing num tokens (num_tokens=832 avail_mem=41.58 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.43it/s]

    Capturing num tokens (num_tokens=832 avail_mem=41.58 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.22it/s]Capturing num tokens (num_tokens=768 avail_mem=41.99 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.22it/s]Capturing num tokens (num_tokens=704 avail_mem=41.60 GB):  41%|████▏     | 24/58 [00:05<00:04,  8.22it/s]Capturing num tokens (num_tokens=704 avail_mem=41.60 GB):  45%|████▍     | 26/58 [00:05<00:03,  8.87it/s]Capturing num tokens (num_tokens=640 avail_mem=41.99 GB):  45%|████▍     | 26/58 [00:05<00:03,  8.87it/s]

    Capturing num tokens (num_tokens=576 avail_mem=41.62 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.87it/s]Capturing num tokens (num_tokens=576 avail_mem=41.62 GB):  48%|████▊     | 28/58 [00:06<00:03,  9.53it/s]Capturing num tokens (num_tokens=512 avail_mem=42.03 GB):  48%|████▊     | 28/58 [00:06<00:03,  9.53it/s]Capturing num tokens (num_tokens=480 avail_mem=41.93 GB):  48%|████▊     | 28/58 [00:06<00:03,  9.53it/s]

    Capturing num tokens (num_tokens=480 avail_mem=41.93 GB):  52%|█████▏    | 30/58 [00:06<00:02, 10.36it/s]Capturing num tokens (num_tokens=448 avail_mem=41.85 GB):  52%|█████▏    | 30/58 [00:06<00:02, 10.36it/s]Capturing num tokens (num_tokens=416 avail_mem=41.66 GB):  52%|█████▏    | 30/58 [00:06<00:02, 10.36it/s]Capturing num tokens (num_tokens=416 avail_mem=41.66 GB):  55%|█████▌    | 32/58 [00:06<00:02, 11.07it/s]Capturing num tokens (num_tokens=384 avail_mem=41.96 GB):  55%|█████▌    | 32/58 [00:06<00:02, 11.07it/s]

    Capturing num tokens (num_tokens=352 avail_mem=41.68 GB):  55%|█████▌    | 32/58 [00:06<00:02, 11.07it/s]

    Capturing num tokens (num_tokens=352 avail_mem=41.68 GB):  59%|█████▊    | 34/58 [00:06<00:03,  7.05it/s]Capturing num tokens (num_tokens=320 avail_mem=41.95 GB):  59%|█████▊    | 34/58 [00:06<00:03,  7.05it/s]Capturing num tokens (num_tokens=288 avail_mem=41.70 GB):  59%|█████▊    | 34/58 [00:07<00:03,  7.05it/s]Capturing num tokens (num_tokens=288 avail_mem=41.70 GB):  62%|██████▏   | 36/58 [00:07<00:02,  8.49it/s]Capturing num tokens (num_tokens=256 avail_mem=41.94 GB):  62%|██████▏   | 36/58 [00:07<00:02,  8.49it/s]

    Capturing num tokens (num_tokens=240 avail_mem=41.94 GB):  62%|██████▏   | 36/58 [00:07<00:02,  8.49it/s]Capturing num tokens (num_tokens=240 avail_mem=41.94 GB):  66%|██████▌   | 38/58 [00:07<00:01, 10.09it/s]Capturing num tokens (num_tokens=224 avail_mem=41.74 GB):  66%|██████▌   | 38/58 [00:07<00:01, 10.09it/s]Capturing num tokens (num_tokens=208 avail_mem=41.93 GB):  66%|██████▌   | 38/58 [00:07<00:01, 10.09it/s]Capturing num tokens (num_tokens=208 avail_mem=41.93 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.15it/s]Capturing num tokens (num_tokens=192 avail_mem=41.95 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.15it/s]

    Capturing num tokens (num_tokens=176 avail_mem=41.79 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.15it/s]Capturing num tokens (num_tokens=160 avail_mem=41.92 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.15it/s]Capturing num tokens (num_tokens=160 avail_mem=41.92 GB):  74%|███████▍  | 43/58 [00:07<00:01, 13.49it/s]Capturing num tokens (num_tokens=144 avail_mem=41.91 GB):  74%|███████▍  | 43/58 [00:07<00:01, 13.49it/s]Capturing num tokens (num_tokens=128 avail_mem=41.81 GB):  74%|███████▍  | 43/58 [00:07<00:01, 13.49it/s]

    Capturing num tokens (num_tokens=128 avail_mem=41.81 GB):  78%|███████▊  | 45/58 [00:07<00:00, 14.03it/s]Capturing num tokens (num_tokens=112 avail_mem=41.84 GB):  78%|███████▊  | 45/58 [00:07<00:00, 14.03it/s]Capturing num tokens (num_tokens=96 avail_mem=41.90 GB):  78%|███████▊  | 45/58 [00:07<00:00, 14.03it/s] Capturing num tokens (num_tokens=96 avail_mem=41.90 GB):  81%|████████  | 47/58 [00:07<00:00, 14.93it/s]Capturing num tokens (num_tokens=80 avail_mem=41.89 GB):  81%|████████  | 47/58 [00:07<00:00, 14.93it/s]Capturing num tokens (num_tokens=64 avail_mem=41.89 GB):  81%|████████  | 47/58 [00:07<00:00, 14.93it/s]

    Capturing num tokens (num_tokens=48 avail_mem=41.88 GB):  81%|████████  | 47/58 [00:07<00:00, 14.93it/s]Capturing num tokens (num_tokens=48 avail_mem=41.88 GB):  86%|████████▌ | 50/58 [00:07<00:00, 17.05it/s]Capturing num tokens (num_tokens=32 avail_mem=41.82 GB):  86%|████████▌ | 50/58 [00:07<00:00, 17.05it/s]Capturing num tokens (num_tokens=28 avail_mem=41.87 GB):  86%|████████▌ | 50/58 [00:07<00:00, 17.05it/s]Capturing num tokens (num_tokens=24 avail_mem=41.86 GB):  86%|████████▌ | 50/58 [00:07<00:00, 17.05it/s]Capturing num tokens (num_tokens=24 avail_mem=41.86 GB):  91%|█████████▏| 53/58 [00:07<00:00, 19.63it/s]Capturing num tokens (num_tokens=20 avail_mem=41.85 GB):  91%|█████████▏| 53/58 [00:07<00:00, 19.63it/s]

    Capturing num tokens (num_tokens=16 avail_mem=41.84 GB):  91%|█████████▏| 53/58 [00:08<00:00, 19.63it/s]Capturing num tokens (num_tokens=12 avail_mem=41.84 GB):  91%|█████████▏| 53/58 [00:08<00:00, 19.63it/s]Capturing num tokens (num_tokens=12 avail_mem=41.84 GB):  97%|█████████▋| 56/58 [00:08<00:00, 20.64it/s]Capturing num tokens (num_tokens=8 avail_mem=41.83 GB):  97%|█████████▋| 56/58 [00:08<00:00, 20.64it/s] Capturing num tokens (num_tokens=4 avail_mem=41.79 GB):  97%|█████████▋| 56/58 [00:08<00:00, 20.64it/s]Capturing num tokens (num_tokens=4 avail_mem=41.79 GB): 100%|██████████| 58/58 [00:08<00:00,  7.09it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably double-check that. <br><br>Wait, I think I heard somewhere that Paris has a population over 21 million. Maybe 21.6 million? I'm not sure if that's accurate. I should look up the latest data to confirm. Also, I wonder if the population includes just the city proper or the entire metropolitan area. I think sometimes population counts include the broader area, so maybe that's why the number is higher. <br><br>I should make sure to present this information in JSON format as the user requested. So, the key would be "capital" with the value "Paris" and another key "population" with the number. I need to decide whether to include the metropolitan area or just the city limits. Since the user didn't specify, I'll go with the metropolitan area population, which I think is around 21.6 million. <br><br>I should also consider the source of this information to ensure accuracy. Maybe the World Bank or recent census data. I recall that the population has been growing steadily, so 21.6 million seems reasonable. I don't think it's too high or too low. <br><br>So, putting it all together, the JSON should have two keys: "capital" and "population". The value for "capital" is "Paris", and "population" is 21600000. I should format it correctly with proper syntax, using quotes and commas where necessary. <br><br>I think that's all. I just need to make sure the information is accurate and present it in the required format.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21600000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants the current date and time, along with the weather. I need to figure out how to respond using the allowed functions.<br><br>First, I should determine which functions to use. The user mentioned two things: date and time, and weather. For the date and time, I can use the get_current_date function, which requires a timezone parameter. Since the user is in New York, I'll use 'America/New_York' as the timezone.<br><br>Next, for the weather, I can use get_current_weather. I need the city and state. The city is New York, and the state abbreviation is NY. The unit isn't specified, so I'll default to Fahrenheit, but I should check if the user prefers Celsius. However, since it's not specified, Fahrenheit is acceptable.<br><br>I should structure the response correctly, using the required format. I'll call get_current_date first with the timezone parameter, then get_current_weather with city and state, and finally mention the unit in parentheses.<br><br>I also need to ensure that each function call is on its own line and follows the specified syntax. No markdown, just plain text with the function names and parameters in JSON format.<br><br>Let me put it all together. The date part will be a single function call, and the weather part will include the unit. I'll make sure the parameters are correctly formatted and that the entire response is clear and concise.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function> (in Fahrenheit)</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s straightforward. \n\nNext, I need to find the population. I remember that Paris is a major city, so its population is quite large. I think it\'s over 3 million, but I\'m not exactly sure of the exact number. Maybe I should double-check that. \n\nWait, I recall that the population figure can vary depending on the source and the year. The user didn\'t specify a particular year, so I should probably go with the most recent estimate. I believe the population is around 3,500,000 as of 2023. \n\nNow, I need to structure this information into a JSON format. JSON typically uses key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" since the user mentioned France. \n\nI should make sure the keys are in English to keep it clear. The city is Paris, the population is 3,500,000, and the country is France. I\'ll format this into a JSON object. \n\nI also need to present this in a way that\'s easy to read, so I\'ll use proper syntax with quotation marks and commas in the right places. No trailing commas to avoid errors. \n\nPutting it all together, the JSON should look something like this: a dictionary with the keys and the corresponding values. I\'ll make sure to test it to ensure it\'s valid, but since I\'m just writing it out, I\'ll assume it\'s correct based on my knowledge. \n\nI think that\'s all. The user just needs the information in JSON, so this should satisfy their request.\n</think>{\n\n"name": "Paris",\n"population": 3500000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4710, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 11, 773, 1181, 7042, 374, 5008, 3460, 13, 358, 1744, 432, 594, 916, 220, 18, 3526, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 10696, 358, 1265, 1990, 15934, 429, 13, 4710, 14190, 11, 358, 19091, 429, 279, 7042, 7071, 646, 13289, 11649, 389, 279, 2530, 323, 279, 1042, 13, 576, 1196, 3207, 944, 13837, 264, 3953, 1042, 11, 773, 358, 1265, 4658, 728, 448, 279, 1429, 3213, 16045, 13, 358, 4411, 279, 7042, 374, 2163, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 4710, 7039, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 11136, 5711, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 2474, 279, 1196, 9733, 9625, 13, 4710, 40, 1265, 1281, 2704, 279, 6894, 525, 304, 6364, 311, 2506, 432, 2797, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 13, 358, 3278, 3561, 419, 1119, 264, 4718, 1633, 13, 4710, 40, 1083, 1184, 311, 3042, 419, 304, 264, 1616, 429, 594, 4135, 311, 1349, 11, 773, 358, 3278, 990, 6169, 19482, 448, 54231, 15423, 323, 76602, 304, 279, 1290, 7482, 13, 2308, 27748, 76602, 311, 5648, 5975, 13, 4710, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 264, 10997, 448, 279, 6894, 323, 279, 12159, 2750, 13, 358, 3278, 1281, 2704, 311, 1273, 432, 311, 5978, 432, 594, 2697, 11, 714, 2474, 358, 2776, 1101, 4378, 432, 700, 11, 358, 3278, 9658, 432, 594, 4396, 3118, 389, 847, 6540, 13, 4710, 40, 1744, 429, 594, 678, 13, 576, 1196, 1101, 3880, 279, 1995, 304, 4718, 11, 773, 419, 1265, 26553, 862, 1681, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 18, 20, 15, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '834b1efba9cf4fb6bf2ebe28f5f9a984', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 405, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.8199927881360054, 'response_sent_to_client_ts': 1772920936.9643497}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants both the general information and the population. So, I\'ll create an object with a "name" field for the capital, a "general_information" section that includes the administrative center, area, and government department, and a "population" section that includes the current population and a note about the data source.\n\nI should also add a "source" field to indicate where the population data comes from, which is the 2020 census. This makes the information more transparent and trustworthy.\n\nPutting it all together, I\'ll format the JSON with proper syntax, using double quotes for strings and ensuring that the keys are clear and descriptive. I\'ll make sure there are no typos and that the JSON is valid.\n\nFinally, I\'ll present the JSON in a code block so the user can easily copy and use it. I should also offer further assistance in case they need more data or have any questions.\n</think>{\n  "name": "Paris",\n  "population": 2174300\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 2176, 279, 4586, 1995, 323, 279, 7042, 13, 2055, 11, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 6722, 11, 264, 330, 24595, 35212, 1, 3772, 429, 5646, 279, 22707, 4126, 11, 3082, 11, 323, 3033, 9292, 11, 323, 264, 330, 44441, 1, 3772, 429, 5646, 279, 1482, 7042, 323, 264, 5185, 911, 279, 821, 2530, 382, 40, 1265, 1083, 912, 264, 330, 2427, 1, 2070, 311, 13216, 1380, 279, 7042, 821, 4041, 504, 11, 892, 374, 279, 220, 17, 15, 17, 15, 43602, 13, 1096, 3643, 279, 1995, 803, 17821, 323, 55942, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3561, 279, 4718, 448, 6169, 19482, 11, 1667, 1990, 17194, 369, 9069, 323, 22573, 429, 279, 6894, 525, 2797, 323, 52844, 13, 358, 3278, 1281, 2704, 1052, 525, 902, 13580, 966, 323, 429, 279, 4718, 374, 2697, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 304, 264, 2038, 2504, 773, 279, 1196, 646, 6707, 2975, 323, 990, 432, 13, 358, 1265, 1083, 3010, 4623, 12994, 304, 1142, 807, 1184, 803, 821, 476, 614, 894, 4755, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 198, 92, 151643], 'meta_info': {'id': '29da5332714b4379975abbaa05f5e4c7', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 386, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.317584832198918, 'response_sent_to_client_ts': 1772920939.291267}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'ecdd5dd0b65e417991bbe1c66c915f64', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.12665076088160276, 'response_sent_to_client_ts': 1772920939.4427705}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '71890d63bb93424caf9c4b140f467634', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.1265924321487546, 'response_sent_to_client_ts': 1772920939.4427843}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'd3d4c051644a4554b6fb115e41684a50', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.12655883701518178, 'response_sent_to_client_ts': 1772920939.4427888}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '02f88c998289430aa55be52b5843aa51', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 18.028872039169073, 'response_sent_to_client_ts': 1772920957.4783442}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so I need to figure out what the user is asking for here. They said, "Give me the information and population of the capital of France in the JSON format." Hmm, let me break this down.\n\nFirst, the capital of France is Paris. That\'s straightforward. They want information about the capital, specifically its population. Wait, but they also mentioned "information" in general. Should I include more than just the population, or just the population?\n\nI should consider what they really need. If they just want the population, that\'s simple. But maybe they want a bit more context, like the administrative region, area, or maybe some key facts about it. Let me think, the previous example response included the population under "Population", but perhaps the user expects more details.\n\nI should also make sure about the population numbers. I know Paris is a major city, so the population is in the millions. But maybe the exact number has changed recently? I should probably look up the most recent data to provide an accurate figure.\n\nWait, the user also specified JSON format. So I need to structure this data correctly in JSON. Including the city name, region, population, and maybe other fields like the area or some key features. But the user didn\'t specify all of that, just information and population. So perhaps keeping it simple with population is sufficient.\n\nWait again, in the response I gave earlier, I provided the population as 2,165,645. I should double-check that. Let me think about the most recent census. I believe it was around 2,165,000 in 2020. So maybe that\'s accurate enough for now.\n\nI should also ensure that the format is correctly JSON. So, something like a dictionary with the keys "capital_name", "population", etc. I think the initial response was good, but perhaps I can add just the population for clarity, as the user mentioned.\n\nWait, no, the original response included more than population, so maybe it\'s better to include those other fields as well to give a comprehensive answer. That way, the user gets more information about the capital besides just the population. That might be helpful.\n\nSo, to summarize, I should structure the data in JSON format. The keys can include city name, population, administrative region, and area. I\'ll make sure to include all that. Also, check the population number for accuracy.\n\nLet me also think about possible edge cases. The user might be calling this function for a project or a form, so accuracy is important. If there\'s any risk of the data being outdated, maybe I should suggest checking a reliable source for the latest population number, but since it\'s a static response, I\'ll just provide the data as is.\n\nAlright, I think I\'ve covered the necessary points. Now, just to format everything correctly in JSON. Make sure commas are in the right places to avoid syntax errors. Okay, so finalizing the response with the city info and population.\n</think>\n\n```json\n{\n  "capital": {\n    "name": "Paris",\n    "region": " Île-de-France",\n    "population": 2165645,\n    "area": 105.59,\n    "founded": 348AD,\n    ".topAnchor": 1789\n  },\n  "year_capital": 1815,\n  "parse": true,\n  "data_type": "country info"\n}\n```', 'output_ids': [32313, 11, 773, 358, 1184, 311, 7071, 700, 1128, 279, 1196, 374, 10161, 369, 1588, 13, 2379, 1053, 11, 330, 35127, 752, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 279, 4718, 3561, 1189, 88190, 11, 1077, 752, 1438, 419, 1495, 382, 5338, 11, 279, 6722, 315, 9625, 374, 12095, 13, 2938, 594, 30339, 13, 2379, 1366, 1995, 911, 279, 6722, 11, 11689, 1181, 7042, 13, 13824, 11, 714, 807, 1083, 9733, 330, 25069, 1, 304, 4586, 13, 12260, 358, 2924, 803, 1091, 1101, 279, 7042, 11, 476, 1101, 279, 7042, 1939, 40, 1265, 2908, 1128, 807, 2167, 1184, 13, 1416, 807, 1101, 1366, 279, 7042, 11, 429, 594, 4285, 13, 1988, 7196, 807, 1366, 264, 2699, 803, 2266, 11, 1075, 279, 22707, 5537, 11, 3082, 11, 476, 7196, 1045, 1376, 13064, 911, 432, 13, 6771, 752, 1744, 11, 279, 3681, 3110, 2033, 5230, 279, 7042, 1212, 330, 53371, 497, 714, 8365, 279, 1196, 24297, 803, 3565, 382, 40, 1265, 1083, 1281, 2704, 911, 279, 7042, 5109, 13, 358, 1414, 12095, 374, 264, 3598, 3283, 11, 773, 279, 7042, 374, 304, 279, 11728, 13, 1988, 7196, 279, 4734, 1372, 702, 5497, 5926, 30, 358, 1265, 4658, 1401, 705, 279, 1429, 3213, 821, 311, 3410, 458, 13382, 7071, 382, 14190, 11, 279, 1196, 1083, 5189, 4718, 3561, 13, 2055, 358, 1184, 311, 5944, 419, 821, 12440, 304, 4718, 13, 55121, 279, 3283, 829, 11, 5537, 11, 7042, 11, 323, 7196, 1008, 5043, 1075, 279, 3082, 476, 1045, 1376, 4419, 13, 1988, 279, 1196, 3207, 944, 13837, 678, 315, 429, 11, 1101, 1995, 323, 7042, 13, 2055, 8365, 10282, 432, 4285, 448, 7042, 374, 14016, 382, 14190, 1549, 11, 304, 279, 2033, 358, 6551, 6788, 11, 358, 3897, 279, 7042, 438, 220, 17, 11, 16, 21, 20, 11, 21, 19, 20, 13, 358, 1265, 1990, 15934, 429, 13, 6771, 752, 1744, 911, 279, 1429, 3213, 43602, 13, 358, 4411, 432, 572, 2163, 220, 17, 11, 16, 21, 20, 11, 15, 15, 15, 304, 220, 17, 15, 17, 15, 13, 2055, 7196, 429, 594, 13382, 3322, 369, 1431, 382, 40, 1265, 1083, 5978, 429, 279, 3561, 374, 12440, 4718, 13, 2055, 11, 2494, 1075, 264, 10997, 448, 279, 6894, 330, 65063, 1269, 497, 330, 44441, 497, 4992, 13, 358, 1744, 279, 2856, 2033, 572, 1661, 11, 714, 8365, 358, 646, 912, 1101, 279, 7042, 369, 31273, 11, 438, 279, 1196, 9733, 382, 14190, 11, 902, 11, 279, 4024, 2033, 5230, 803, 1091, 7042, 11, 773, 7196, 432, 594, 2664, 311, 2924, 1846, 1008, 5043, 438, 1632, 311, 2968, 264, 15817, 4226, 13, 2938, 1616, 11, 279, 1196, 5221, 803, 1995, 911, 279, 6722, 27758, 1101, 279, 7042, 13, 2938, 2578, 387, 10950, 382, 4416, 11, 311, 62079, 11, 358, 1265, 5944, 279, 821, 304, 4718, 3561, 13, 576, 6894, 646, 2924, 3283, 829, 11, 7042, 11, 22707, 5537, 11, 323, 3082, 13, 358, 3278, 1281, 2704, 311, 2924, 678, 429, 13, 7281, 11, 1779, 279, 7042, 1372, 369, 13403, 382, 10061, 752, 1083, 1744, 911, 3204, 6821, 5048, 13, 576, 1196, 2578, 387, 8098, 419, 729, 369, 264, 2390, 476, 264, 1352, 11, 773, 13403, 374, 2989, 13, 1416, 1052, 594, 894, 5214, 315, 279, 821, 1660, 40526, 11, 7196, 358, 1265, 4190, 13295, 264, 14720, 2530, 369, 279, 5535, 7042, 1372, 11, 714, 2474, 432, 594, 264, 1099, 2033, 11, 358, 3278, 1101, 3410, 279, 821, 438, 374, 382, 71486, 11, 358, 1744, 358, 3003, 9761, 279, 5871, 3501, 13, 4695, 11, 1101, 311, 3561, 4297, 12440, 304, 4718, 13, 7405, 2704, 76602, 525, 304, 279, 1290, 7482, 311, 5648, 19482, 5975, 13, 35439, 11, 773, 1590, 4849, 279, 2033, 448, 279, 3283, 3546, 323, 7042, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 65063, 788, 341, 262, 330, 606, 788, 330, 59604, 756, 262, 330, 3943, 788, 330, 59108, 273, 6810, 7276, 34106, 756, 262, 330, 44441, 788, 220, 17, 16, 21, 20, 21, 19, 20, 345, 262, 330, 4798, 788, 220, 16, 15, 20, 13, 20, 24, 345, 262, 330, 69, 13082, 788, 220, 18, 19, 23, 1808, 345, 262, 330, 68933, 788, 220, 16, 22, 23, 24, 198, 220, 1153, 220, 330, 3157, 16388, 2174, 788, 220, 16, 23, 16, 20, 345, 220, 330, 6400, 788, 830, 345, 220, 330, 691, 1819, 788, 330, 11141, 3546, 698, 532, 73594, 151643], 'meta_info': {'id': '204a3d34a314441f9c2eaa79cd35c246', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 724, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 6.752813167870045, 'response_sent_to_client_ts': 1772920964.2397377}}</strong>



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


    [2026-03-07 22:02:47] INFO server_args.py:2048: Attention backend not specified. Use fa3 backend by default.


    [2026-03-07 22:02:47] INFO server_args.py:3155: Set soft_watchdog_timeout since in CI


    [2026-03-07 22:02:47] INFO engine.py:177: server_args=ServerArgs(model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=870738758, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser='deepseek-r1', tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.02s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.27s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.23s/it]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=62.65 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=62.65 GB):   5%|▌         | 1/20 [00:00<00:02,  6.47it/s]Capturing batches (bs=120 avail_mem=62.54 GB):   5%|▌         | 1/20 [00:00<00:02,  6.47it/s]Capturing batches (bs=112 avail_mem=62.54 GB):   5%|▌         | 1/20 [00:00<00:02,  6.47it/s]

    Capturing batches (bs=104 avail_mem=62.53 GB):   5%|▌         | 1/20 [00:00<00:02,  6.47it/s]Capturing batches (bs=96 avail_mem=62.53 GB):   5%|▌         | 1/20 [00:00<00:02,  6.47it/s] Capturing batches (bs=96 avail_mem=62.53 GB):  25%|██▌       | 5/20 [00:00<00:00, 20.53it/s]Capturing batches (bs=88 avail_mem=62.53 GB):  25%|██▌       | 5/20 [00:00<00:00, 20.53it/s]Capturing batches (bs=80 avail_mem=62.53 GB):  25%|██▌       | 5/20 [00:00<00:00, 20.53it/s]Capturing batches (bs=72 avail_mem=62.53 GB):  25%|██▌       | 5/20 [00:00<00:00, 20.53it/s]Capturing batches (bs=64 avail_mem=62.53 GB):  25%|██▌       | 5/20 [00:00<00:00, 20.53it/s]Capturing batches (bs=64 avail_mem=62.53 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.45it/s]Capturing batches (bs=56 avail_mem=62.53 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.45it/s]

    Capturing batches (bs=48 avail_mem=62.52 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.45it/s]Capturing batches (bs=40 avail_mem=62.52 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.45it/s]Capturing batches (bs=32 avail_mem=62.52 GB):  45%|████▌     | 9/20 [00:00<00:00, 25.45it/s]Capturing batches (bs=32 avail_mem=62.52 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.38it/s]Capturing batches (bs=24 avail_mem=62.52 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.38it/s]Capturing batches (bs=16 avail_mem=62.52 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.38it/s]

    Capturing batches (bs=12 avail_mem=62.52 GB):  65%|██████▌   | 13/20 [00:00<00:00, 27.38it/s]Capturing batches (bs=12 avail_mem=62.52 GB):  80%|████████  | 16/20 [00:00<00:00, 23.02it/s]Capturing batches (bs=8 avail_mem=62.52 GB):  80%|████████  | 16/20 [00:00<00:00, 23.02it/s] Capturing batches (bs=4 avail_mem=62.52 GB):  80%|████████  | 16/20 [00:00<00:00, 23.02it/s]Capturing batches (bs=2 avail_mem=62.52 GB):  80%|████████  | 16/20 [00:00<00:00, 23.02it/s]Capturing batches (bs=2 avail_mem=62.52 GB):  95%|█████████▌| 19/20 [00:00<00:00, 24.25it/s]Capturing batches (bs=1 avail_mem=62.51 GB):  95%|█████████▌| 19/20 [00:00<00:00, 24.25it/s]Capturing batches (bs=1 avail_mem=62.51 GB): 100%|██████████| 20/20 [00:00<00:00, 23.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:47,  2.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:47,  2.94s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:21,  1.46s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:21,  1.46s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:48,  1.13it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:48,  1.13it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:32,  1.65it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:32,  1.65it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:23,  2.25it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:23,  2.25it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.63it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.63it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.42it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.42it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.31it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.31it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.31it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.96it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.96it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.96it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.47it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.47it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.47it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04, 10.04it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04, 10.04it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04, 10.04it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.92it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.92it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.92it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.92it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.14it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.14it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.14it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.14it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.14it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:01, 20.57it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:00, 29.35it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:00, 29.35it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:00, 29.35it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:00, 29.35it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.35it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.35it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:00, 29.35it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 45.18it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 45.18it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 45.18it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 45.18it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 45.18it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:06<00:00, 45.18it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:06<00:00, 19.58it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:06<00:00, 19.58it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:06<00:00, 19.58it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:06<00:00, 19.58it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:06<00:00, 19.58it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:06<00:00, 19.58it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:06<00:00, 19.58it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:06<00:00, 19.58it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:06<00:00, 19.58it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:06<00:00, 19.58it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:06<00:00, 19.58it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=45.66 GB):   2%|▏         | 1/58 [00:00<00:17,  3.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.63 GB):   2%|▏         | 1/58 [00:00<00:17,  3.34it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=45.63 GB):   3%|▎         | 2/58 [00:00<00:15,  3.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.63 GB):   3%|▎         | 2/58 [00:00<00:15,  3.51it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=45.63 GB):   5%|▌         | 3/58 [00:00<00:14,  3.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=45.63 GB):   5%|▌         | 3/58 [00:00<00:14,  3.74it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=45.63 GB):   7%|▋         | 4/58 [00:01<00:13,  3.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.64 GB):   7%|▋         | 4/58 [00:01<00:13,  3.93it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=45.64 GB):   9%|▊         | 5/58 [00:01<00:12,  4.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.64 GB):   9%|▊         | 5/58 [00:01<00:12,  4.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.64 GB):  10%|█         | 6/58 [00:01<00:11,  4.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.64 GB):  10%|█         | 6/58 [00:01<00:11,  4.56it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=45.64 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.65 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.65 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.65 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.46it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=45.65 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.66 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.66 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.65 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.37it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=45.65 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.65 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.65 GB):  21%|██        | 12/58 [00:02<00:06,  7.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=45.65 GB):  21%|██        | 12/58 [00:02<00:06,  7.38it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=45.65 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.65 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.65 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.65 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.65 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.00it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=45.65 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.00it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=45.65 GB):  29%|██▉       | 17/58 [00:02<00:05,  7.89it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.65 GB):  29%|██▉       | 17/58 [00:02<00:05,  7.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.65 GB):  29%|██▉       | 17/58 [00:02<00:05,  7.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.65 GB):  33%|███▎      | 19/58 [00:02<00:04,  9.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.65 GB):  33%|███▎      | 19/58 [00:02<00:04,  9.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=45.65 GB):  33%|███▎      | 19/58 [00:03<00:04,  9.58it/s]

    Capturing num tokens (num_tokens=960 avail_mem=45.65 GB):  33%|███▎      | 19/58 [00:03<00:04,  9.58it/s] Capturing num tokens (num_tokens=960 avail_mem=45.65 GB):  38%|███▊      | 22/58 [00:03<00:02, 12.70it/s]Capturing num tokens (num_tokens=896 avail_mem=45.64 GB):  38%|███▊      | 22/58 [00:03<00:02, 12.70it/s]Capturing num tokens (num_tokens=832 avail_mem=45.64 GB):  38%|███▊      | 22/58 [00:03<00:02, 12.70it/s]Capturing num tokens (num_tokens=768 avail_mem=45.63 GB):  38%|███▊      | 22/58 [00:03<00:02, 12.70it/s]Capturing num tokens (num_tokens=768 avail_mem=45.63 GB):  43%|████▎     | 25/58 [00:03<00:02, 15.78it/s]Capturing num tokens (num_tokens=704 avail_mem=45.57 GB):  43%|████▎     | 25/58 [00:03<00:02, 15.78it/s]Capturing num tokens (num_tokens=640 avail_mem=45.56 GB):  43%|████▎     | 25/58 [00:03<00:02, 15.78it/s]

    Capturing num tokens (num_tokens=576 avail_mem=45.56 GB):  43%|████▎     | 25/58 [00:03<00:02, 15.78it/s]Capturing num tokens (num_tokens=576 avail_mem=45.56 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.42it/s]Capturing num tokens (num_tokens=512 avail_mem=45.55 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.42it/s]Capturing num tokens (num_tokens=480 avail_mem=45.55 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.42it/s]Capturing num tokens (num_tokens=448 avail_mem=45.55 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.42it/s]Capturing num tokens (num_tokens=416 avail_mem=45.54 GB):  48%|████▊     | 28/58 [00:03<00:01, 18.42it/s]Capturing num tokens (num_tokens=416 avail_mem=45.54 GB):  55%|█████▌    | 32/58 [00:03<00:01, 22.00it/s]Capturing num tokens (num_tokens=384 avail_mem=45.54 GB):  55%|█████▌    | 32/58 [00:03<00:01, 22.00it/s]Capturing num tokens (num_tokens=352 avail_mem=45.53 GB):  55%|█████▌    | 32/58 [00:03<00:01, 22.00it/s]

    Capturing num tokens (num_tokens=320 avail_mem=45.53 GB):  55%|█████▌    | 32/58 [00:03<00:01, 22.00it/s]Capturing num tokens (num_tokens=288 avail_mem=45.53 GB):  55%|█████▌    | 32/58 [00:03<00:01, 22.00it/s]Capturing num tokens (num_tokens=288 avail_mem=45.53 GB):  62%|██████▏   | 36/58 [00:03<00:00, 25.01it/s]Capturing num tokens (num_tokens=256 avail_mem=45.52 GB):  62%|██████▏   | 36/58 [00:03<00:00, 25.01it/s]Capturing num tokens (num_tokens=240 avail_mem=45.52 GB):  62%|██████▏   | 36/58 [00:03<00:00, 25.01it/s]Capturing num tokens (num_tokens=224 avail_mem=45.52 GB):  62%|██████▏   | 36/58 [00:03<00:00, 25.01it/s]Capturing num tokens (num_tokens=208 avail_mem=45.51 GB):  62%|██████▏   | 36/58 [00:03<00:00, 25.01it/s]Capturing num tokens (num_tokens=208 avail_mem=45.51 GB):  69%|██████▉   | 40/58 [00:03<00:00, 28.52it/s]Capturing num tokens (num_tokens=192 avail_mem=45.51 GB):  69%|██████▉   | 40/58 [00:03<00:00, 28.52it/s]Capturing num tokens (num_tokens=176 avail_mem=45.50 GB):  69%|██████▉   | 40/58 [00:03<00:00, 28.52it/s]

    Capturing num tokens (num_tokens=160 avail_mem=45.50 GB):  69%|██████▉   | 40/58 [00:03<00:00, 28.52it/s]Capturing num tokens (num_tokens=144 avail_mem=45.50 GB):  69%|██████▉   | 40/58 [00:03<00:00, 28.52it/s]Capturing num tokens (num_tokens=144 avail_mem=45.50 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.41it/s]Capturing num tokens (num_tokens=128 avail_mem=45.51 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.41it/s]Capturing num tokens (num_tokens=112 avail_mem=45.50 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.41it/s]Capturing num tokens (num_tokens=96 avail_mem=45.50 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.41it/s] Capturing num tokens (num_tokens=80 avail_mem=45.50 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.41it/s]Capturing num tokens (num_tokens=80 avail_mem=45.50 GB):  83%|████████▎ | 48/58 [00:03<00:00, 32.84it/s]Capturing num tokens (num_tokens=64 avail_mem=45.49 GB):  83%|████████▎ | 48/58 [00:03<00:00, 32.84it/s]Capturing num tokens (num_tokens=48 avail_mem=45.49 GB):  83%|████████▎ | 48/58 [00:03<00:00, 32.84it/s]

    Capturing num tokens (num_tokens=32 avail_mem=45.49 GB):  83%|████████▎ | 48/58 [00:03<00:00, 32.84it/s]Capturing num tokens (num_tokens=28 avail_mem=45.48 GB):  83%|████████▎ | 48/58 [00:03<00:00, 32.84it/s]Capturing num tokens (num_tokens=24 avail_mem=45.48 GB):  83%|████████▎ | 48/58 [00:04<00:00, 32.84it/s]Capturing num tokens (num_tokens=24 avail_mem=45.48 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.75it/s]Capturing num tokens (num_tokens=20 avail_mem=45.47 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.75it/s]Capturing num tokens (num_tokens=16 avail_mem=45.47 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.75it/s]Capturing num tokens (num_tokens=12 avail_mem=45.47 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.75it/s]Capturing num tokens (num_tokens=8 avail_mem=45.46 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.75it/s] Capturing num tokens (num_tokens=8 avail_mem=45.46 GB):  98%|█████████▊| 57/58 [00:04<00:00, 36.73it/s]Capturing num tokens (num_tokens=4 avail_mem=45.46 GB):  98%|█████████▊| 57/58 [00:04<00:00, 36.73it/s]Capturing num tokens (num_tokens=4 avail_mem=45.46 GB): 100%|██████████| 58/58 [00:04<00:00, 13.94it/s]


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
    Generated text: Berlin is the capital of Germany
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
    
    Generated text: Okay, so the user asked me to provide the information and population of the capital of France in JSON format. Hmm, first, I should figure out who the user is and why they're asking this. They might be a student working on a project, someone doing research, or maybe just curious. Either way, they need a straightforward answer in a structured format.
    
    Alright, the capital of France is Paris. I should confirm that first. Paris is indeed the administrative capital, but sometimes people confuse it with Lyon or Marseille, which are more historically significant. But I think the user just needs the current information, so Paris it is.
    
    Next, I need to find out the population. I remember that Paris is the most populous city in France, but I'm not sure of the exact number. I should check recent data, probably from the 2020 census since that's when the latest figures were available. Let me think, I think the population is around 2.2 million. But wait, I'm not 100% certain. Maybe I should verify that. Alternatively, I could mention that the population is approximately 2,175,000 as of 2022. That seems accurate.
    
    Now, the user wants this in JSON format. JSON is a data interchange format, so I'll structure it with a key-value pair. The keys should be "city" and "population". The value for city is obviously "Paris". For population, I'll include the approximate number along with the year it was counted, just to be clear about the data's source and accuracy.
    
    I should also consider the user's possible deeper needs. Maybe they need this data for a presentation, a report, or just for personal knowledge. Either way, providing a reliable and formatted answer is key. I don't want to include outdated or incorrect information, so double-checking the numbers is important. 
    
    I think that's it. I'll format it neatly in JSON, making sure the keys are in English for clarity. That should satisfy the user's request effectively.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "city": "Paris",
      "population": 2175000
    }
    ```



```python
llm.shutdown()
```
