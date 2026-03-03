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

    [2026-03-03 04:59:31] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-03 04:59:31] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-03 04:59:31] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-03 04:59:35] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 04:59:35] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 04:59:35] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:34: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-03 04:59:36] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.
    [2026-03-03 04:59:36] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 04:59:41] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 04:59:41] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 04:59:41] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-03 04:59:41] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 04:59:41] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 04:59:41] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-03 04:59:46] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-03 04:59:46] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-03 04:59:46] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.09s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.45s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.39s/it]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=27.33 GB):   0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=27.33 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.09it/s]Capturing batches (bs=2 avail_mem=27.27 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.09it/s]

    Capturing batches (bs=1 avail_mem=27.27 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.09it/s]Capturing batches (bs=1 avail_mem=27.27 GB): 100%|██████████| 3/3 [00:00<00:00, 12.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:28,  2.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:28,  2.61s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:30,  1.61s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:30,  1.61s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:59,  1.08s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:59,  1.08s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:28,  1.83it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:28,  1.83it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:23,  2.17it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:23,  2.17it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:20,  2.49it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:20,  2.49it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:16,  2.88it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:16,  2.88it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:14,  3.28it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:14,  3.28it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:12,  3.73it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:12,  3.73it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:11,  4.14it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:11,  4.14it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:09,  4.68it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:09,  4.68it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:08,  5.21it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:08,  5.21it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:07,  6.00it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:07,  6.00it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:06,  6.69it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:06,  6.69it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:07<00:06,  6.69it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:07<00:04,  8.13it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:07<00:04,  8.13it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:07<00:04,  8.13it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:03,  9.81it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:03,  9.81it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:03,  9.81it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:07<00:03, 11.96it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:07<00:03, 11.96it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:07<00:03, 11.96it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:07<00:03, 11.96it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:07<00:02, 15.07it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:07<00:02, 15.07it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:07<00:02, 15.07it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:07<00:02, 15.07it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:07<00:01, 18.55it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:07<00:01, 18.55it/s]

    Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:07<00:01, 18.55it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:07<00:01, 18.55it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:07<00:01, 18.55it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:07<00:01, 22.32it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:07<00:01, 22.32it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:07<00:01, 22.32it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:07<00:01, 22.32it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:07<00:01, 22.32it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 26.67it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 26.67it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 26.67it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 26.67it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 26.67it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 26.67it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:07<00:00, 32.15it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:07<00:00, 32.15it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:08<00:00, 32.15it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:08<00:00, 32.15it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:08<00:00, 32.15it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:08<00:00, 32.15it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:08<00:00, 32.15it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:08<00:00, 38.41it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:08<00:00, 38.41it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:08<00:00, 38.41it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:08<00:00, 38.41it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:08<00:00, 38.41it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:08<00:00, 35.04it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:08<00:00, 35.04it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:08<00:00, 35.04it/s]

    Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:08<00:00, 35.04it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:08<00:00, 35.04it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:08<00:00, 34.58it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:08<00:00, 34.58it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:08<00:00, 34.58it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:08<00:00, 34.58it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.92 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.92 GB):   2%|▏         | 1/58 [00:00<00:37,  1.53it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.04 GB):   2%|▏         | 1/58 [00:00<00:37,  1.53it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.04 GB):   3%|▎         | 2/58 [00:01<00:33,  1.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.12 GB):   3%|▎         | 2/58 [00:01<00:33,  1.68it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.12 GB):   5%|▌         | 3/58 [00:01<00:30,  1.83it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.91 GB):   5%|▌         | 3/58 [00:01<00:30,  1.83it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.91 GB):   7%|▋         | 4/58 [00:02<00:26,  2.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.92 GB):   7%|▋         | 4/58 [00:02<00:26,  2.02it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.92 GB):   9%|▊         | 5/58 [00:02<00:24,  2.18it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.40 GB):   9%|▊         | 5/58 [00:02<00:24,  2.18it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.40 GB):  10%|█         | 6/58 [00:02<00:22,  2.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.43 GB):  10%|█         | 6/58 [00:02<00:22,  2.35it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.43 GB):  12%|█▏        | 7/58 [00:03<00:19,  2.57it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.93 GB):  12%|█▏        | 7/58 [00:03<00:19,  2.57it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.93 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.93 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.87it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.93 GB):  16%|█▌        | 9/58 [00:03<00:16,  3.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.58 GB):  16%|█▌        | 9/58 [00:03<00:16,  3.03it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.58 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.26it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.94 GB):  17%|█▋        | 10/58 [00:03<00:14,  3.26it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.94 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.94 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.69it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=43.94 GB):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.70 GB):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.70 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.93 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.41it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.93 GB):  24%|██▍       | 14/58 [00:04<00:08,  5.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.93 GB):  24%|██▍       | 14/58 [00:04<00:08,  5.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.93 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.55it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.81 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.55it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=43.81 GB):  28%|██▊       | 16/58 [00:04<00:06,  6.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.92 GB):  28%|██▊       | 16/58 [00:04<00:06,  6.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.92 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.92 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.83it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=43.91 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.91 GB):  33%|███▎      | 19/58 [00:05<00:04,  8.43it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.90 GB):  33%|███▎      | 19/58 [00:05<00:04,  8.43it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.90 GB):  33%|███▎      | 19/58 [00:05<00:04,  8.43it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=43.90 GB):  36%|███▌      | 21/58 [00:05<00:03,  9.89it/s]Capturing num tokens (num_tokens=960 avail_mem=43.91 GB):  36%|███▌      | 21/58 [00:05<00:03,  9.89it/s] Capturing num tokens (num_tokens=896 avail_mem=43.89 GB):  36%|███▌      | 21/58 [00:05<00:03,  9.89it/s]Capturing num tokens (num_tokens=896 avail_mem=43.89 GB):  40%|███▉      | 23/58 [00:05<00:03, 11.43it/s]Capturing num tokens (num_tokens=832 avail_mem=43.88 GB):  40%|███▉      | 23/58 [00:05<00:03, 11.43it/s]Capturing num tokens (num_tokens=768 avail_mem=43.87 GB):  40%|███▉      | 23/58 [00:05<00:03, 11.43it/s]

    Capturing num tokens (num_tokens=768 avail_mem=43.87 GB):  43%|████▎     | 25/58 [00:05<00:02, 13.11it/s]Capturing num tokens (num_tokens=704 avail_mem=43.86 GB):  43%|████▎     | 25/58 [00:05<00:02, 13.11it/s]Capturing num tokens (num_tokens=640 avail_mem=43.85 GB):  43%|████▎     | 25/58 [00:05<00:02, 13.11it/s]Capturing num tokens (num_tokens=640 avail_mem=43.85 GB):  47%|████▋     | 27/58 [00:05<00:02, 14.75it/s]Capturing num tokens (num_tokens=576 avail_mem=43.84 GB):  47%|████▋     | 27/58 [00:05<00:02, 14.75it/s]Capturing num tokens (num_tokens=512 avail_mem=43.82 GB):  47%|████▋     | 27/58 [00:05<00:02, 14.75it/s]

    Capturing num tokens (num_tokens=512 avail_mem=43.82 GB):  50%|█████     | 29/58 [00:05<00:01, 15.32it/s]Capturing num tokens (num_tokens=480 avail_mem=43.83 GB):  50%|█████     | 29/58 [00:05<00:01, 15.32it/s]Capturing num tokens (num_tokens=448 avail_mem=43.82 GB):  50%|█████     | 29/58 [00:05<00:01, 15.32it/s]Capturing num tokens (num_tokens=448 avail_mem=43.82 GB):  53%|█████▎    | 31/58 [00:05<00:01, 16.10it/s]Capturing num tokens (num_tokens=416 avail_mem=43.82 GB):  53%|█████▎    | 31/58 [00:05<00:01, 16.10it/s]Capturing num tokens (num_tokens=384 avail_mem=43.81 GB):  53%|█████▎    | 31/58 [00:05<00:01, 16.10it/s]Capturing num tokens (num_tokens=352 avail_mem=43.80 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.10it/s]

    Capturing num tokens (num_tokens=352 avail_mem=43.80 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.65it/s]Capturing num tokens (num_tokens=320 avail_mem=43.79 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.65it/s]Capturing num tokens (num_tokens=288 avail_mem=43.80 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.65it/s]Capturing num tokens (num_tokens=256 avail_mem=43.79 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.65it/s]Capturing num tokens (num_tokens=256 avail_mem=43.79 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.74it/s]Capturing num tokens (num_tokens=240 avail_mem=43.78 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.74it/s]Capturing num tokens (num_tokens=224 avail_mem=43.78 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.74it/s]Capturing num tokens (num_tokens=208 avail_mem=43.77 GB):  64%|██████▍   | 37/58 [00:06<00:01, 20.74it/s]

    Capturing num tokens (num_tokens=208 avail_mem=43.77 GB):  69%|██████▉   | 40/58 [00:06<00:00, 22.58it/s]Capturing num tokens (num_tokens=192 avail_mem=43.76 GB):  69%|██████▉   | 40/58 [00:06<00:00, 22.58it/s]Capturing num tokens (num_tokens=176 avail_mem=43.76 GB):  69%|██████▉   | 40/58 [00:06<00:00, 22.58it/s]Capturing num tokens (num_tokens=160 avail_mem=43.75 GB):  69%|██████▉   | 40/58 [00:06<00:00, 22.58it/s]Capturing num tokens (num_tokens=160 avail_mem=43.75 GB):  74%|███████▍  | 43/58 [00:06<00:00, 24.08it/s]Capturing num tokens (num_tokens=144 avail_mem=43.74 GB):  74%|███████▍  | 43/58 [00:06<00:00, 24.08it/s]Capturing num tokens (num_tokens=128 avail_mem=43.75 GB):  74%|███████▍  | 43/58 [00:06<00:00, 24.08it/s]Capturing num tokens (num_tokens=112 avail_mem=43.74 GB):  74%|███████▍  | 43/58 [00:06<00:00, 24.08it/s]

    Capturing num tokens (num_tokens=112 avail_mem=43.74 GB):  79%|███████▉  | 46/58 [00:06<00:00, 25.06it/s]Capturing num tokens (num_tokens=96 avail_mem=43.73 GB):  79%|███████▉  | 46/58 [00:06<00:00, 25.06it/s] Capturing num tokens (num_tokens=80 avail_mem=43.72 GB):  79%|███████▉  | 46/58 [00:06<00:00, 25.06it/s]Capturing num tokens (num_tokens=64 avail_mem=43.72 GB):  79%|███████▉  | 46/58 [00:06<00:00, 25.06it/s]Capturing num tokens (num_tokens=64 avail_mem=43.72 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.90it/s]Capturing num tokens (num_tokens=48 avail_mem=43.71 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.90it/s]Capturing num tokens (num_tokens=32 avail_mem=43.70 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.90it/s]Capturing num tokens (num_tokens=28 avail_mem=43.70 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.90it/s]Capturing num tokens (num_tokens=24 avail_mem=43.70 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.90it/s]

    Capturing num tokens (num_tokens=24 avail_mem=43.70 GB):  91%|█████████▏| 53/58 [00:06<00:00, 29.75it/s]Capturing num tokens (num_tokens=20 avail_mem=43.70 GB):  91%|█████████▏| 53/58 [00:06<00:00, 29.75it/s]Capturing num tokens (num_tokens=16 avail_mem=43.69 GB):  91%|█████████▏| 53/58 [00:06<00:00, 29.75it/s]Capturing num tokens (num_tokens=12 avail_mem=43.69 GB):  91%|█████████▏| 53/58 [00:06<00:00, 29.75it/s]Capturing num tokens (num_tokens=8 avail_mem=43.68 GB):  91%|█████████▏| 53/58 [00:06<00:00, 29.75it/s] Capturing num tokens (num_tokens=4 avail_mem=43.68 GB):  91%|█████████▏| 53/58 [00:06<00:00, 29.75it/s]Capturing num tokens (num_tokens=4 avail_mem=43.68 GB): 100%|██████████| 58/58 [00:06<00:00, 33.29it/s]Capturing num tokens (num_tokens=4 avail_mem=43.68 GB): 100%|██████████| 58/58 [00:06<00:00,  8.49it/s]



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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants the current date and time, along with the weather. I need to figure out how to get this using the provided functions.<br><br>First, I should use the get_current_date function. The parameters require a timezone. Since the user is in New York, I'll set the timezone parameter to 'America/New_York'. That should give me the correct date and time.<br><br>Next, for the weather, I'll use get_current_weather. The city is New York, the state is NY, and I'll choose Fahrenheit since the user didn't specify otherwise. So I'll include city: 'New York', state: 'NY', unit: 'fahrenheit' in the parameters.<br><br>I need to make sure each function call is separate and follows the required format. So I'll call get_current_date first with the timezone, then get_current_weather with the city, state, and unit. Each call will be on its own line with the correct parameters in JSON format.<br><br>I should also remember to add the sources where I get the information from, but since the functions are provided, I don't need to look up anything else. Just make sure the parameters are correctly formatted.<br><br>Putting it all together, I'll write two function calls: one for the date and time with the timezone, and another for the weather with the specified city, state, and unit. Each will be enclosed in the <function> tags with the parameters inside.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s straightforward. \n\nNext, I need to find the population. I remember that Paris is a major city, so its population is quite large. I think it\'s over 3 million, but I\'m not exactly sure of the exact number. Maybe I should double-check that. \n\nWait, I recall that the population figure can vary depending on the source and the year. The user didn\'t specify a particular year, so I should probably go with the most recent estimate. I believe the population is around 3,500,000 as of 2023. \n\nNow, I need to structure this information into a JSON format. JSON typically uses key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" since the user mentioned France. \n\nI should make sure the keys are in English to keep it clear. The city is Paris, the population is 3,500,000, and the country is France. I\'ll format this into a JSON object. \n\nI also need to present this in a way that\'s easy to read, so I\'ll use proper syntax with quotation marks and commas in the right places. No trailing commas to avoid errors. \n\nPutting it all together, the JSON should look something like this: a dictionary with the keys and the corresponding values. I\'ll make sure to test it to ensure it\'s valid, but since I\'m just writing it out, I\'ll assume it\'s correct based on my knowledge. \n\nI think that\'s all. The user just needs the information in JSON, so this should satisfy their request.\n</think>{\n\n"name": "Paris",\n"population": 3500000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4710, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 11, 773, 1181, 7042, 374, 5008, 3460, 13, 358, 1744, 432, 594, 916, 220, 18, 3526, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 10696, 358, 1265, 1990, 15934, 429, 13, 4710, 14190, 11, 358, 19091, 429, 279, 7042, 7071, 646, 13289, 11649, 389, 279, 2530, 323, 279, 1042, 13, 576, 1196, 3207, 944, 13837, 264, 3953, 1042, 11, 773, 358, 1265, 4658, 728, 448, 279, 1429, 3213, 16045, 13, 358, 4411, 279, 7042, 374, 2163, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 4710, 7039, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 11136, 5711, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 2474, 279, 1196, 9733, 9625, 13, 4710, 40, 1265, 1281, 2704, 279, 6894, 525, 304, 6364, 311, 2506, 432, 2797, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 13, 358, 3278, 3561, 419, 1119, 264, 4718, 1633, 13, 4710, 40, 1083, 1184, 311, 3042, 419, 304, 264, 1616, 429, 594, 4135, 311, 1349, 11, 773, 358, 3278, 990, 6169, 19482, 448, 54231, 15423, 323, 76602, 304, 279, 1290, 7482, 13, 2308, 27748, 76602, 311, 5648, 5975, 13, 4710, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 264, 10997, 448, 279, 6894, 323, 279, 12159, 2750, 13, 358, 3278, 1281, 2704, 311, 1273, 432, 311, 5978, 432, 594, 2697, 11, 714, 2474, 358, 2776, 1101, 4378, 432, 700, 11, 358, 3278, 9658, 432, 594, 4396, 3118, 389, 847, 6540, 13, 4710, 40, 1744, 429, 594, 678, 13, 576, 1196, 1101, 3880, 279, 1995, 304, 4718, 11, 773, 419, 1265, 26553, 862, 1681, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 18, 20, 15, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'f6ada6700cca416d83461c8e366e57ce', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 405, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.7770794820971787, 'response_sent_to_client_ts': 1772514031.7411964}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s straightforward. \n\nNext, I need to find the population. I remember that Paris is a major city, so its population is quite large. I think it\'s over 3 million, but I\'m not exactly sure of the exact number. Maybe I should double-check that. \n\nWait, I recall that the population figure can vary depending on the source and the year. The user didn\'t specify a particular year, so I should probably go with the most recent estimate. I believe the population is around 3,500,000 as of 2023. \n\nNow, I need to structure this information into a JSON format. JSON typically uses key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" since the user mentioned France. \n\nI should make sure the keys are in English to keep it clear. The city is Paris, the population is 3,500,000, and the country is France. I\'ll format this into a JSON object. \n\nI also need to present this in a way that\'s easy to read, so I\'ll use proper syntax with quotation marks and commas in the right places. No trailing commas to avoid errors. \n\nPutting it all together, the JSON should look something like this: a dictionary with the keys and the corresponding values. I\'ll make sure to test it to ensure it\'s valid, but since I\'m just writing it out, I\'ll assume it\'s correct based on my knowledge. \n\nI think that\'s all. The user just needs the information in JSON, so this should satisfy their request.\n</think>{\n\n"name": "Paris",\n"population": 3500000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4710, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 11, 773, 1181, 7042, 374, 5008, 3460, 13, 358, 1744, 432, 594, 916, 220, 18, 3526, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 10696, 358, 1265, 1990, 15934, 429, 13, 4710, 14190, 11, 358, 19091, 429, 279, 7042, 7071, 646, 13289, 11649, 389, 279, 2530, 323, 279, 1042, 13, 576, 1196, 3207, 944, 13837, 264, 3953, 1042, 11, 773, 358, 1265, 4658, 728, 448, 279, 1429, 3213, 16045, 13, 358, 4411, 279, 7042, 374, 2163, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 4710, 7039, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 11136, 5711, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 2474, 279, 1196, 9733, 9625, 13, 4710, 40, 1265, 1281, 2704, 279, 6894, 525, 304, 6364, 311, 2506, 432, 2797, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 13, 358, 3278, 3561, 419, 1119, 264, 4718, 1633, 13, 4710, 40, 1083, 1184, 311, 3042, 419, 304, 264, 1616, 429, 594, 4135, 311, 1349, 11, 773, 358, 3278, 990, 6169, 19482, 448, 54231, 15423, 323, 76602, 304, 279, 1290, 7482, 13, 2308, 27748, 76602, 311, 5648, 5975, 13, 4710, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 264, 10997, 448, 279, 6894, 323, 279, 12159, 2750, 13, 358, 3278, 1281, 2704, 311, 1273, 432, 311, 5978, 432, 594, 2697, 11, 714, 2474, 358, 2776, 1101, 4378, 432, 700, 11, 358, 3278, 9658, 432, 594, 4396, 3118, 389, 847, 6540, 13, 4710, 40, 1744, 429, 594, 678, 13, 576, 1196, 1101, 3880, 279, 1995, 304, 4718, 11, 773, 419, 1265, 26553, 862, 1681, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 18, 20, 15, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'd0920e944ddb433bb7d04093d74cc88e', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 405, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.400143758393824, 'response_sent_to_client_ts': 1772514034.1535442}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '5a70748519e542ff839fc0fb773b6435', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.12548694713041186, 'response_sent_to_client_ts': 1772514034.3071327}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '3e91c209b2044a25a1738e832156786e', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.12538525881245732, 'response_sent_to_client_ts': 1772514034.3071532}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '822c5b0e76bf4189b9d0d53c7f4ea84c', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.12534466478973627, 'response_sent_to_client_ts': 1772514034.30716}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '249d7763511f4042bc084c7d6ef764c7', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 18.241443667095155, 'response_sent_to_client_ts': 1772514052.5563467}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so I need to figure out how to respond to this user\'s question. They asked for the information and population of the capital of France in JSON format. Hmm, the capital of France is Paris, so that\'s my starting point.\n\nFirst, I should gather the necessary information. Paris is indeed the capital, so that\'s straightforward. Now, what\'s its population? I remember hearing it\'s around 2 million, but I\'m not exactly sure of the exact figure. Maybe I should double-check that. Let me think—yes, as of the latest data, Paris has a population of approximately 2,170,000 people. That sounds right.\n\nNext, I need to present this information in JSON format. JSON requires key-value pairs, so I should structure it with keys like "city," "country," and "population." Each key will be a string, and the value will be the respective data. \n\nI should make sure the JSON syntax is correct because any mistake there could cause errors when reading the data. Let me write it out: the city is Paris, France, and the population is 2170000. I\'ll format it neatly inside double quotes and use commas appropriately.\n\nWait, should I include units for population? Probably not, since the user just asked for the number, so I\'ll leave it as is. Also, is there an image included? The user didn\'t mention that, so I should stick to just the JSON response.\n\nI should also consider if the population figure is up to date. Population numbers can change, so maybe I should mention that it\'s an approximate figure based on recent estimates. That makes the response more accurate and transparent.\n\nPutting it all together, I\'ll structure the JSON with the key-value pairs, ensure the commas are in the right places, and include a comment about the population being an estimate. That way, the user knows there might be slight variations depending on the year or source.\n\nAlright, I think I have everything I need. Time to format it properly and present it clearly so the user gets the exact information they asked for in the correct format.\n</think>\n\nCertainly! Here is the information about the capital of France in JSON format:\n\n```json\n{\n  "city": "Paris",\n  "country": "France",\n  "population": 2170000\n}\n```\n\nThe population figure is an approximate number based on recent estimates.', 'output_ids': [32313, 11, 773, 358, 1184, 311, 7071, 700, 1246, 311, 5889, 311, 419, 1196, 594, 3405, 13, 2379, 4588, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 88190, 11, 279, 6722, 315, 9625, 374, 12095, 11, 773, 429, 594, 847, 5916, 1459, 382, 5338, 11, 358, 1265, 9567, 279, 5871, 1995, 13, 12095, 374, 12824, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 1128, 594, 1181, 7042, 30, 358, 6099, 10778, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 7071, 13, 10696, 358, 1265, 1990, 15934, 429, 13, 6771, 752, 1744, 2293, 9693, 11, 438, 315, 279, 5535, 821, 11, 12095, 702, 264, 7042, 315, 13187, 220, 17, 11, 16, 22, 15, 11, 15, 15, 15, 1251, 13, 2938, 10362, 1290, 382, 5847, 11, 358, 1184, 311, 3042, 419, 1995, 304, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 1265, 5944, 432, 448, 6894, 1075, 330, 8926, 1335, 330, 11141, 1335, 323, 330, 44441, 1189, 8886, 1376, 686, 387, 264, 914, 11, 323, 279, 897, 686, 387, 279, 19511, 821, 13, 4710, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 1576, 894, 16523, 1052, 1410, 5240, 5975, 979, 5290, 279, 821, 13, 6771, 752, 3270, 432, 700, 25, 279, 3283, 374, 12095, 11, 9625, 11, 323, 279, 7042, 374, 220, 17, 16, 22, 15, 15, 15, 15, 13, 358, 3278, 3561, 432, 62166, 4766, 1990, 17194, 323, 990, 76602, 34901, 382, 14190, 11, 1265, 358, 2924, 8153, 369, 7042, 30, 37154, 537, 11, 2474, 279, 1196, 1101, 4588, 369, 279, 1372, 11, 773, 358, 3278, 5274, 432, 438, 374, 13, 7281, 11, 374, 1052, 458, 2168, 5230, 30, 576, 1196, 3207, 944, 6286, 429, 11, 773, 358, 1265, 9214, 311, 1101, 279, 4718, 2033, 382, 40, 1265, 1083, 2908, 421, 279, 7042, 7071, 374, 705, 311, 2400, 13, 39529, 5109, 646, 2297, 11, 773, 7196, 358, 1265, 6286, 429, 432, 594, 458, 44868, 7071, 3118, 389, 3213, 17530, 13, 2938, 3643, 279, 2033, 803, 13382, 323, 17821, 382, 97904, 432, 678, 3786, 11, 358, 3278, 5944, 279, 4718, 448, 279, 1376, 19083, 13530, 11, 5978, 279, 76602, 525, 304, 279, 1290, 7482, 11, 323, 2924, 264, 3980, 911, 279, 7042, 1660, 458, 16045, 13, 2938, 1616, 11, 279, 1196, 8788, 1052, 2578, 387, 8112, 26244, 11649, 389, 279, 1042, 476, 2530, 382, 71486, 11, 358, 1744, 358, 614, 4297, 358, 1184, 13, 4120, 311, 3561, 432, 10277, 323, 3042, 432, 9355, 773, 279, 1196, 5221, 279, 4734, 1995, 807, 4588, 369, 304, 279, 4396, 3561, 624, 151649, 271, 95456, 0, 5692, 374, 279, 1995, 911, 279, 6722, 315, 9625, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 8926, 788, 330, 59604, 756, 220, 330, 11141, 788, 330, 49000, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 198, 532, 13874, 19324, 785, 7042, 7071, 374, 458, 44868, 1372, 3118, 389, 3213, 17530, 13, 151643], 'meta_info': {'id': 'f87ef595d61243d8b09125ce6ff3b918', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 500, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 4.0822718646377325, 'response_sent_to_client_ts': 1772514056.6482463}}</strong>



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


    [2026-03-03 05:00:59] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.


    [2026-03-03 05:00:59] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 05:00:59] INFO engine.py:157: server_args=ServerArgs(model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=71381612, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser='deepseek-r1', tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.03s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.31s/it]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=62.65 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=62.65 GB):   5%|▌         | 1/20 [00:00<00:03,  4.86it/s]Capturing batches (bs=120 avail_mem=62.54 GB):   5%|▌         | 1/20 [00:00<00:03,  4.86it/s]Capturing batches (bs=112 avail_mem=62.54 GB):   5%|▌         | 1/20 [00:00<00:03,  4.86it/s]Capturing batches (bs=104 avail_mem=62.53 GB):   5%|▌         | 1/20 [00:00<00:03,  4.86it/s]Capturing batches (bs=104 avail_mem=62.53 GB):  20%|██        | 4/20 [00:00<00:01, 13.73it/s]Capturing batches (bs=96 avail_mem=62.53 GB):  20%|██        | 4/20 [00:00<00:01, 13.73it/s] Capturing batches (bs=88 avail_mem=62.53 GB):  20%|██        | 4/20 [00:00<00:01, 13.73it/s]Capturing batches (bs=80 avail_mem=62.53 GB):  20%|██        | 4/20 [00:00<00:01, 13.73it/s]

    Capturing batches (bs=80 avail_mem=62.53 GB):  35%|███▌      | 7/20 [00:00<00:00, 17.62it/s]Capturing batches (bs=72 avail_mem=62.53 GB):  35%|███▌      | 7/20 [00:00<00:00, 17.62it/s]Capturing batches (bs=64 avail_mem=62.53 GB):  35%|███▌      | 7/20 [00:00<00:00, 17.62it/s]Capturing batches (bs=56 avail_mem=62.53 GB):  35%|███▌      | 7/20 [00:00<00:00, 17.62it/s]Capturing batches (bs=56 avail_mem=62.53 GB):  50%|█████     | 10/20 [00:00<00:00, 20.70it/s]Capturing batches (bs=48 avail_mem=62.52 GB):  50%|█████     | 10/20 [00:00<00:00, 20.70it/s]Capturing batches (bs=40 avail_mem=62.52 GB):  50%|█████     | 10/20 [00:00<00:00, 20.70it/s]Capturing batches (bs=32 avail_mem=62.52 GB):  50%|█████     | 10/20 [00:00<00:00, 20.70it/s]

    Capturing batches (bs=32 avail_mem=62.52 GB):  65%|██████▌   | 13/20 [00:00<00:00, 22.52it/s]Capturing batches (bs=24 avail_mem=62.52 GB):  65%|██████▌   | 13/20 [00:00<00:00, 22.52it/s]Capturing batches (bs=16 avail_mem=62.52 GB):  65%|██████▌   | 13/20 [00:00<00:00, 22.52it/s]Capturing batches (bs=12 avail_mem=62.52 GB):  65%|██████▌   | 13/20 [00:00<00:00, 22.52it/s]Capturing batches (bs=12 avail_mem=62.52 GB):  80%|████████  | 16/20 [00:00<00:00, 21.32it/s]Capturing batches (bs=8 avail_mem=62.52 GB):  80%|████████  | 16/20 [00:00<00:00, 21.32it/s] Capturing batches (bs=4 avail_mem=62.52 GB):  80%|████████  | 16/20 [00:00<00:00, 21.32it/s]

    Capturing batches (bs=2 avail_mem=62.52 GB):  80%|████████  | 16/20 [00:00<00:00, 21.32it/s]Capturing batches (bs=1 avail_mem=62.51 GB):  80%|████████  | 16/20 [00:00<00:00, 21.32it/s]Capturing batches (bs=1 avail_mem=62.51 GB): 100%|██████████| 20/20 [00:00<00:00, 25.04it/s]Capturing batches (bs=1 avail_mem=62.51 GB): 100%|██████████| 20/20 [00:00<00:00, 20.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:16,  1.36s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:16,  1.36s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:45,  1.20it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:45,  1.20it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:22,  2.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:22,  2.36it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:17,  3.01it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:17,  3.01it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:13,  3.74it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:13,  3.74it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.52it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.52it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.38it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.38it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.38it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  7.00it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  7.00it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  7.00it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  9.81it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  9.81it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  9.81it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.72it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.72it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.72it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.72it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:02, 14.94it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:02, 14.94it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:02, 14.94it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:02, 14.94it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 14.94it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.31it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.31it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.31it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.31it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.31it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.31it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 27.43it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 27.43it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 27.43it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 27.43it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 27.43it/s]

    Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 27.43it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 27.43it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 42.67it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 42.67it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 42.67it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 42.67it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 42.67it/s]

    Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 42.67it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 42.67it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 42.67it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 42.67it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 42.67it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 55.14it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 55.14it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 55.14it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 55.14it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 55.14it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 55.14it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 55.14it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 55.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.51it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=60.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=60.68 GB):   2%|▏         | 1/58 [00:00<00:16,  3.36it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.65 GB):   2%|▏         | 1/58 [00:00<00:16,  3.36it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=60.65 GB):   3%|▎         | 2/58 [00:00<00:15,  3.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.65 GB):   3%|▎         | 2/58 [00:00<00:15,  3.53it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=60.65 GB):   5%|▌         | 3/58 [00:00<00:14,  3.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.11 GB):   5%|▌         | 3/58 [00:00<00:14,  3.72it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.11 GB):   7%|▋         | 4/58 [00:01<00:13,  3.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.64 GB):   7%|▋         | 4/58 [00:01<00:13,  3.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=45.64 GB):   9%|▊         | 5/58 [00:01<00:12,  4.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.64 GB):   9%|▊         | 5/58 [00:01<00:12,  4.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.64 GB):  10%|█         | 6/58 [00:01<00:11,  4.62it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.64 GB):  10%|█         | 6/58 [00:01<00:11,  4.62it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=45.64 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.65 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.65 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.65 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.47it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=45.65 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.66 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.66 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.65 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.41it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=45.65 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.95it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.65 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.95it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.65 GB):  21%|██        | 12/58 [00:02<00:06,  7.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=45.65 GB):  21%|██        | 12/58 [00:02<00:06,  7.58it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=45.65 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.65 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.65 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.65 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.65 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.58it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=45.65 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=45.65 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.65 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.65 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.65 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.65 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.74it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=45.65 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.74it/s]Capturing num tokens (num_tokens=960 avail_mem=45.65 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.74it/s] Capturing num tokens (num_tokens=960 avail_mem=45.65 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.09it/s]Capturing num tokens (num_tokens=896 avail_mem=45.64 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.09it/s]Capturing num tokens (num_tokens=832 avail_mem=45.64 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.09it/s]Capturing num tokens (num_tokens=768 avail_mem=45.63 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.09it/s]Capturing num tokens (num_tokens=768 avail_mem=45.63 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.97it/s]Capturing num tokens (num_tokens=704 avail_mem=45.63 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.97it/s]

    Capturing num tokens (num_tokens=640 avail_mem=45.63 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.97it/s]Capturing num tokens (num_tokens=576 avail_mem=45.62 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.97it/s]Capturing num tokens (num_tokens=576 avail_mem=45.62 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.64it/s]Capturing num tokens (num_tokens=512 avail_mem=45.62 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.64it/s]Capturing num tokens (num_tokens=480 avail_mem=45.61 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.64it/s]Capturing num tokens (num_tokens=448 avail_mem=45.61 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.64it/s]Capturing num tokens (num_tokens=416 avail_mem=45.61 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.64it/s]Capturing num tokens (num_tokens=416 avail_mem=45.61 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.81it/s]Capturing num tokens (num_tokens=384 avail_mem=45.60 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.81it/s]

    Capturing num tokens (num_tokens=352 avail_mem=45.60 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.81it/s]Capturing num tokens (num_tokens=320 avail_mem=45.60 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.81it/s]Capturing num tokens (num_tokens=288 avail_mem=45.59 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.81it/s]Capturing num tokens (num_tokens=288 avail_mem=45.59 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.99it/s]Capturing num tokens (num_tokens=256 avail_mem=45.59 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.99it/s]Capturing num tokens (num_tokens=240 avail_mem=45.59 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.99it/s]Capturing num tokens (num_tokens=224 avail_mem=45.58 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.99it/s]Capturing num tokens (num_tokens=208 avail_mem=45.58 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.99it/s]Capturing num tokens (num_tokens=208 avail_mem=45.58 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.65it/s]Capturing num tokens (num_tokens=192 avail_mem=45.57 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.65it/s]

    Capturing num tokens (num_tokens=176 avail_mem=45.57 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.65it/s]Capturing num tokens (num_tokens=160 avail_mem=45.56 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.65it/s]Capturing num tokens (num_tokens=144 avail_mem=45.56 GB):  69%|██████▉   | 40/58 [00:03<00:00, 30.65it/s]Capturing num tokens (num_tokens=144 avail_mem=45.56 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.70it/s]Capturing num tokens (num_tokens=128 avail_mem=45.57 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.70it/s]Capturing num tokens (num_tokens=112 avail_mem=45.57 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.70it/s]Capturing num tokens (num_tokens=96 avail_mem=45.56 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.70it/s] Capturing num tokens (num_tokens=80 avail_mem=45.56 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.70it/s]Capturing num tokens (num_tokens=80 avail_mem=45.56 GB):  83%|████████▎ | 48/58 [00:03<00:00, 34.28it/s]Capturing num tokens (num_tokens=64 avail_mem=45.56 GB):  83%|████████▎ | 48/58 [00:03<00:00, 34.28it/s]

    Capturing num tokens (num_tokens=48 avail_mem=45.56 GB):  83%|████████▎ | 48/58 [00:03<00:00, 34.28it/s]Capturing num tokens (num_tokens=32 avail_mem=45.55 GB):  83%|████████▎ | 48/58 [00:03<00:00, 34.28it/s]Capturing num tokens (num_tokens=28 avail_mem=45.55 GB):  83%|████████▎ | 48/58 [00:03<00:00, 34.28it/s]Capturing num tokens (num_tokens=28 avail_mem=45.55 GB):  90%|████████▉ | 52/58 [00:03<00:00, 35.42it/s]Capturing num tokens (num_tokens=24 avail_mem=45.55 GB):  90%|████████▉ | 52/58 [00:03<00:00, 35.42it/s]Capturing num tokens (num_tokens=20 avail_mem=45.54 GB):  90%|████████▉ | 52/58 [00:03<00:00, 35.42it/s]Capturing num tokens (num_tokens=16 avail_mem=45.53 GB):  90%|████████▉ | 52/58 [00:03<00:00, 35.42it/s]Capturing num tokens (num_tokens=12 avail_mem=45.53 GB):  90%|████████▉ | 52/58 [00:03<00:00, 35.42it/s]Capturing num tokens (num_tokens=12 avail_mem=45.53 GB):  97%|█████████▋| 56/58 [00:03<00:00, 35.30it/s]Capturing num tokens (num_tokens=8 avail_mem=45.53 GB):  97%|█████████▋| 56/58 [00:03<00:00, 35.30it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=45.52 GB):  97%|█████████▋| 56/58 [00:03<00:00, 35.30it/s]Capturing num tokens (num_tokens=4 avail_mem=45.52 GB): 100%|██████████| 58/58 [00:03<00:00, 14.76it/s]


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
    Generated text: London is the capital of England
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
    
    Generated text: Alright, the user is asking for the information and population of the capital of France in JSON format. First, I need to identify the capital of France, which is Paris. Next, I should gather accurate data about Paris's population. I'll have to check the latest estimates to ensure the information is current.
    
    I remember that population figures can vary based on sources and the time of the year, so I should specify that the population figure is approximate and may fluctuate. Also, I should consider including some additional relevant details about Paris, like its location, to provide a more comprehensive JSON structure.
    
    Breaking it down, the JSON should include sections like "city" with its name and location, "population" with the approximate number, "fact" to add some interesting tidbit, and "year" for the population estimate. I'll make sure to format everything correctly with proper syntax.
    
    Finally, I'll present the information in a clear and structured way, ensuring it's easy for the user to understand and use if needed. I should keep the language simple and avoid any unnecessary jargon to make it accessible.
    </think>
    
    Here is the information and population of the capital of France (Paris) in JSON format:
    
    ```json
    {
      "city": {
        "name": "Paris",
        "location": " Île-de-France, Île-de-France, France"
      },
      "population": 2163000,
      "fact": "Paris is the most populous city in France and one of the most populous cities in Western Europe.",
      "year": 2023
    }
    ```
    
    This JSON structure provides the requested information about the capital of France, including its population, location, a notable fact, and the year of the population estimate.



```python
llm.shutdown()
```
