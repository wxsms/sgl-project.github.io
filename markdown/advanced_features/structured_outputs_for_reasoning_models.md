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

    [2026-03-05 20:43:54] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-05 20:43:54] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-05 20:43:54] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-05 20:43:58] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-05 20:43:58] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-05 20:43:58] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:41: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-05 20:44:00] INFO server_args.py:2039: Attention backend not specified. Use fa3 backend by default.
    [2026-03-05 20:44:00] INFO server_args.py:3146: Set soft_watchdog_timeout since in CI


    [2026-03-05 20:44:01] INFO utils.py:452: Successfully reserved port 38885 on host '0.0.0.0'


    [2026-03-05 20:44:05] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-05 20:44:05] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-05 20:44:05] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-05 20:44:05] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-05 20:44:05] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-05 20:44:05] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-05 20:44:09] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-05 20:44:09] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-05 20:44:09] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.36s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.28s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.29s/it]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=31.43 GB):   0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=31.43 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.35it/s]Capturing batches (bs=2 avail_mem=27.85 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.35it/s]

    Capturing batches (bs=1 avail_mem=27.85 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.35it/s]Capturing batches (bs=1 avail_mem=27.85 GB): 100%|██████████| 3/3 [00:00<00:00, 12.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:22,  2.49s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:22,  2.49s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:11,  1.27s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:11,  1.27s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:45,  1.22it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:45,  1.22it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:40,  1.34it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:40,  1.34it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:32,  1.61it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:28,  1.78it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:28,  1.78it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:25,  1.95it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:25,  1.95it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:22,  2.15it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:22,  2.15it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:20,  2.38it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:20,  2.38it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:18,  2.58it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:18,  2.58it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:16,  2.79it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:16,  2.79it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:14,  3.07it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:14,  3.07it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:13,  3.27it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:13,  3.27it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:11,  3.60it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:11,  3.60it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:10,  3.95it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:10,  3.95it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:08<00:09,  4.27it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:08<00:09,  4.27it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:08<00:08,  4.81it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:08<00:08,  4.81it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:08<00:07,  5.22it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:08<00:07,  5.22it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:08<00:06,  5.78it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:08<00:06,  5.78it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:08<00:05,  6.33it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:08<00:05,  6.33it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:08<00:05,  6.33it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:08<00:04,  7.75it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:08<00:04,  7.75it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:08<00:04,  7.75it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:09<00:03,  8.85it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:09<00:03,  8.85it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:09<00:03,  8.85it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:09<00:03,  9.94it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:09<00:03,  9.94it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:09<00:03,  9.94it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:09<00:02, 10.94it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:09<00:02, 10.94it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:09<00:02, 10.94it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:09<00:02, 12.50it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:09<00:02, 12.50it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:09<00:02, 12.50it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:09<00:01, 12.63it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:09<00:01, 12.63it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:09<00:01, 12.63it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:09<00:01, 13.86it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:09<00:01, 13.86it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:09<00:01, 13.86it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:09<00:01, 14.62it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:09<00:01, 14.62it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:09<00:01, 14.62it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:09<00:01, 15.88it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:09<00:01, 15.88it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:10<00:01, 15.88it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:10<00:01, 16.94it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:10<00:01, 16.94it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:10<00:01, 16.94it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:10<00:00, 17.51it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:10<00:00, 17.51it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:10<00:00, 17.51it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:10<00:00, 17.99it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:10<00:00, 17.99it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:10<00:00, 17.99it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:10<00:00, 18.51it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:10<00:00, 18.51it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:10<00:00, 18.51it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:10<00:00, 18.51it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:10<00:00, 19.71it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:10<00:00, 19.71it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:10<00:00, 19.71it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:10<00:00, 19.71it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:10<00:00, 20.33it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:10<00:00, 20.33it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:10<00:00, 20.33it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:10<00:00, 20.33it/s]

    Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:10<00:00, 20.33it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:10<00:00, 24.03it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:10<00:00, 24.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.42 GB):   2%|▏         | 1/58 [00:00<00:34,  1.66it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.38 GB):   2%|▏         | 1/58 [00:00<00:34,  1.66it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.38 GB):   3%|▎         | 2/58 [00:01<00:28,  1.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=23.19 GB):   3%|▎         | 2/58 [00:01<00:28,  1.99it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=23.19 GB):   5%|▌         | 3/58 [00:01<00:38,  1.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=23.34 GB):   5%|▌         | 3/58 [00:01<00:38,  1.45it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=23.34 GB):   7%|▋         | 4/58 [00:02<00:37,  1.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=22.58 GB):   7%|▋         | 4/58 [00:02<00:37,  1.44it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=22.58 GB):   9%|▊         | 5/58 [00:03<00:35,  1.50it/s]Capturing num tokens (num_tokens=5632 avail_mem=22.72 GB):   9%|▊         | 5/58 [00:03<00:35,  1.50it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=22.72 GB):  10%|█         | 6/58 [00:03<00:33,  1.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=24.35 GB):  10%|█         | 6/58 [00:03<00:33,  1.55it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=24.35 GB):  12%|█▏        | 7/58 [00:04<00:31,  1.63it/s]Capturing num tokens (num_tokens=4608 avail_mem=22.85 GB):  12%|█▏        | 7/58 [00:04<00:31,  1.63it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=22.85 GB):  14%|█▍        | 8/58 [00:04<00:28,  1.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=22.98 GB):  14%|█▍        | 8/58 [00:04<00:28,  1.77it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=22.98 GB):  16%|█▌        | 9/58 [00:05<00:25,  1.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=23.71 GB):  16%|█▌        | 9/58 [00:05<00:25,  1.89it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=23.71 GB):  17%|█▋        | 10/58 [00:05<00:24,  1.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=23.18 GB):  17%|█▋        | 10/58 [00:05<00:24,  1.99it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=23.18 GB):  19%|█▉        | 11/58 [00:06<00:23,  2.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=24.38 GB):  19%|█▉        | 11/58 [00:06<00:23,  2.03it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=24.38 GB):  21%|██        | 12/58 [00:06<00:20,  2.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=23.89 GB):  21%|██        | 12/58 [00:06<00:20,  2.23it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=23.89 GB):  22%|██▏       | 13/58 [00:06<00:18,  2.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=23.95 GB):  22%|██▏       | 13/58 [00:06<00:18,  2.42it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=23.95 GB):  24%|██▍       | 14/58 [00:07<00:16,  2.63it/s]Capturing num tokens (num_tokens=2560 avail_mem=23.93 GB):  24%|██▍       | 14/58 [00:07<00:16,  2.63it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=23.93 GB):  26%|██▌       | 15/58 [00:07<00:14,  2.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=23.54 GB):  26%|██▌       | 15/58 [00:07<00:14,  2.90it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=23.54 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=23.60 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.23it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=23.60 GB):  29%|██▉       | 17/58 [00:07<00:11,  3.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=24.39 GB):  29%|██▉       | 17/58 [00:07<00:11,  3.51it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=24.39 GB):  31%|███       | 18/58 [00:08<00:10,  3.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=24.03 GB):  31%|███       | 18/58 [00:08<00:10,  3.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=24.03 GB):  33%|███▎      | 19/58 [00:08<00:09,  4.26it/s]Capturing num tokens (num_tokens=1280 avail_mem=24.06 GB):  33%|███▎      | 19/58 [00:08<00:09,  4.26it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=24.06 GB):  34%|███▍      | 20/58 [00:08<00:08,  4.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.09 GB):  34%|███▍      | 20/58 [00:08<00:08,  4.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.09 GB):  36%|███▌      | 21/58 [00:08<00:07,  4.99it/s]Capturing num tokens (num_tokens=960 avail_mem=24.06 GB):  36%|███▌      | 21/58 [00:08<00:07,  4.99it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=24.06 GB):  38%|███▊      | 22/58 [00:08<00:06,  5.57it/s]Capturing num tokens (num_tokens=896 avail_mem=24.38 GB):  38%|███▊      | 22/58 [00:08<00:06,  5.57it/s]Capturing num tokens (num_tokens=896 avail_mem=24.38 GB):  40%|███▉      | 23/58 [00:08<00:06,  5.78it/s]Capturing num tokens (num_tokens=832 avail_mem=23.89 GB):  40%|███▉      | 23/58 [00:08<00:06,  5.78it/s]

    Capturing num tokens (num_tokens=768 avail_mem=24.32 GB):  40%|███▉      | 23/58 [00:09<00:06,  5.78it/s]Capturing num tokens (num_tokens=768 avail_mem=24.32 GB):  43%|████▎     | 25/58 [00:09<00:04,  6.94it/s]Capturing num tokens (num_tokens=704 avail_mem=24.37 GB):  43%|████▎     | 25/58 [00:09<00:04,  6.94it/s]

    Capturing num tokens (num_tokens=704 avail_mem=24.37 GB):  45%|████▍     | 26/58 [00:09<00:05,  5.89it/s]Capturing num tokens (num_tokens=640 avail_mem=24.17 GB):  45%|████▍     | 26/58 [00:09<00:05,  5.89it/s]Capturing num tokens (num_tokens=640 avail_mem=24.17 GB):  47%|████▋     | 27/58 [00:09<00:04,  6.53it/s]Capturing num tokens (num_tokens=576 avail_mem=24.21 GB):  47%|████▋     | 27/58 [00:09<00:04,  6.53it/s]

    Capturing num tokens (num_tokens=576 avail_mem=24.21 GB):  48%|████▊     | 28/58 [00:09<00:04,  7.00it/s]Capturing num tokens (num_tokens=512 avail_mem=24.03 GB):  48%|████▊     | 28/58 [00:09<00:04,  7.00it/s]Capturing num tokens (num_tokens=480 avail_mem=24.17 GB):  48%|████▊     | 28/58 [00:09<00:04,  7.00it/s]Capturing num tokens (num_tokens=480 avail_mem=24.17 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.62it/s]Capturing num tokens (num_tokens=448 avail_mem=24.33 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.62it/s]

    Capturing num tokens (num_tokens=416 avail_mem=24.32 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.62it/s]Capturing num tokens (num_tokens=416 avail_mem=24.32 GB):  55%|█████▌    | 32/58 [00:09<00:02,  9.61it/s]Capturing num tokens (num_tokens=384 avail_mem=24.34 GB):  55%|█████▌    | 32/58 [00:09<00:02,  9.61it/s]Capturing num tokens (num_tokens=352 avail_mem=24.20 GB):  55%|█████▌    | 32/58 [00:10<00:02,  9.61it/s]

    Capturing num tokens (num_tokens=352 avail_mem=24.20 GB):  59%|█████▊    | 34/58 [00:10<00:02, 11.14it/s]Capturing num tokens (num_tokens=320 avail_mem=24.30 GB):  59%|█████▊    | 34/58 [00:10<00:02, 11.14it/s]Capturing num tokens (num_tokens=288 avail_mem=24.22 GB):  59%|█████▊    | 34/58 [00:10<00:02, 11.14it/s]Capturing num tokens (num_tokens=288 avail_mem=24.22 GB):  62%|██████▏   | 36/58 [00:10<00:01, 12.11it/s]Capturing num tokens (num_tokens=256 avail_mem=24.22 GB):  62%|██████▏   | 36/58 [00:10<00:01, 12.11it/s]Capturing num tokens (num_tokens=240 avail_mem=24.13 GB):  62%|██████▏   | 36/58 [00:10<00:01, 12.11it/s]

    Capturing num tokens (num_tokens=240 avail_mem=24.13 GB):  66%|██████▌   | 38/58 [00:10<00:01, 13.35it/s]Capturing num tokens (num_tokens=224 avail_mem=24.14 GB):  66%|██████▌   | 38/58 [00:10<00:01, 13.35it/s]Capturing num tokens (num_tokens=208 avail_mem=24.24 GB):  66%|██████▌   | 38/58 [00:10<00:01, 13.35it/s]Capturing num tokens (num_tokens=208 avail_mem=24.24 GB):  69%|██████▉   | 40/58 [00:10<00:01, 14.01it/s]Capturing num tokens (num_tokens=192 avail_mem=24.23 GB):  69%|██████▉   | 40/58 [00:10<00:01, 14.01it/s]Capturing num tokens (num_tokens=176 avail_mem=24.22 GB):  69%|██████▉   | 40/58 [00:10<00:01, 14.01it/s]

    Capturing num tokens (num_tokens=176 avail_mem=24.22 GB):  72%|███████▏  | 42/58 [00:10<00:01, 15.07it/s]Capturing num tokens (num_tokens=160 avail_mem=24.21 GB):  72%|███████▏  | 42/58 [00:10<00:01, 15.07it/s]Capturing num tokens (num_tokens=144 avail_mem=24.20 GB):  72%|███████▏  | 42/58 [00:10<00:01, 15.07it/s]Capturing num tokens (num_tokens=144 avail_mem=24.20 GB):  76%|███████▌  | 44/58 [00:10<00:00, 16.03it/s]Capturing num tokens (num_tokens=128 avail_mem=24.16 GB):  76%|███████▌  | 44/58 [00:10<00:00, 16.03it/s]Capturing num tokens (num_tokens=112 avail_mem=24.19 GB):  76%|███████▌  | 44/58 [00:10<00:00, 16.03it/s]

    Capturing num tokens (num_tokens=96 avail_mem=24.18 GB):  76%|███████▌  | 44/58 [00:10<00:00, 16.03it/s] Capturing num tokens (num_tokens=96 avail_mem=24.18 GB):  81%|████████  | 47/58 [00:10<00:00, 17.49it/s]Capturing num tokens (num_tokens=80 avail_mem=24.17 GB):  81%|████████  | 47/58 [00:10<00:00, 17.49it/s]Capturing num tokens (num_tokens=64 avail_mem=24.16 GB):  81%|████████  | 47/58 [00:10<00:00, 17.49it/s]Capturing num tokens (num_tokens=64 avail_mem=24.16 GB):  84%|████████▍ | 49/58 [00:10<00:00, 18.05it/s]Capturing num tokens (num_tokens=48 avail_mem=24.15 GB):  84%|████████▍ | 49/58 [00:10<00:00, 18.05it/s]

    Capturing num tokens (num_tokens=32 avail_mem=24.14 GB):  84%|████████▍ | 49/58 [00:11<00:00, 18.05it/s]Capturing num tokens (num_tokens=28 avail_mem=24.12 GB):  84%|████████▍ | 49/58 [00:11<00:00, 18.05it/s]Capturing num tokens (num_tokens=28 avail_mem=24.12 GB):  90%|████████▉ | 52/58 [00:11<00:00, 18.87it/s]Capturing num tokens (num_tokens=24 avail_mem=24.13 GB):  90%|████████▉ | 52/58 [00:11<00:00, 18.87it/s]Capturing num tokens (num_tokens=20 avail_mem=24.12 GB):  90%|████████▉ | 52/58 [00:11<00:00, 18.87it/s]Capturing num tokens (num_tokens=16 avail_mem=24.10 GB):  90%|████████▉ | 52/58 [00:11<00:00, 18.87it/s]

    Capturing num tokens (num_tokens=16 avail_mem=24.10 GB):  95%|█████████▍| 55/58 [00:11<00:00, 19.64it/s]Capturing num tokens (num_tokens=12 avail_mem=24.09 GB):  95%|█████████▍| 55/58 [00:11<00:00, 19.64it/s]Capturing num tokens (num_tokens=8 avail_mem=24.08 GB):  95%|█████████▍| 55/58 [00:11<00:00, 19.64it/s] Capturing num tokens (num_tokens=4 avail_mem=24.07 GB):  95%|█████████▍| 55/58 [00:11<00:00, 19.64it/s]Capturing num tokens (num_tokens=4 avail_mem=24.07 GB): 100%|██████████| 58/58 [00:11<00:00, 20.54it/s]Capturing num tokens (num_tokens=4 avail_mem=24.07 GB): 100%|██████████| 58/58 [00:11<00:00,  5.10it/s]



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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants to know the current date and time along with the weather. I need to figure out how to get both pieces of information using the functions provided.<br><br>First, I'll look at the functions available. There's get_current_weather and get_current_date. I need to call both, so I'll have to make two separate function calls in the response.<br><br>Starting with get_current_date, the required parameter is timezone, which should be 'America/New_York'. I'll structure the function call with that parameter.<br><br>Next, for the weather, I need the city, state, and unit. The city is 'New York', the state is 'NY', and the unit is probably 'fahrenheit' since the user didn't specify otherwise. I'll include all three parameters in the get_current_weather function.<br><br>I should make sure each function call is properly formatted with the start and end tags, and include the parameters as a JSON object. Also, I need to remind the user to execute the Python code I provide to get the results.<br><br>Putting it all together, I'll write two function calls: one for the date and time, and another for the weather, each with their respective parameters. I'll also add a helpful reminder at the end.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York)}</function> <function=get_current_weather>{"   <br>    }</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s straightforward. \n\nNext, I need to find the population. I remember that Paris is a major city, so its population is quite large. I think it\'s over 3 million, but I\'m not exactly sure of the exact number. Maybe I should double-check that. \n\nWait, I recall that the population figure can vary depending on the source and the year. The user didn\'t specify a particular year, so I should probably go with the most recent estimate. I believe the population is around 3,500,000 as of 2023. \n\nNow, I need to structure this information into a JSON format. JSON typically uses key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" since the user mentioned France. \n\nI should make sure the keys are in English to keep it clear. The city is Paris, the population is 3,500,000, and the country is France. I\'ll format this into a JSON object. \n\nI also need to present this in a way that\'s easy to read, so I\'ll use proper syntax with quotation marks and commas in the right places. No trailing commas to avoid errors. \n\nPutting it all together, the JSON should look something like this: a dictionary with the keys and the corresponding values. I\'ll make sure to test it to ensure it\'s valid, but since I\'m just writing it out, I\'ll assume it\'s correct based on my knowledge. \n\nI think that\'s all. The user just needs the information in JSON, so this should satisfy their request.\n</think>{\n\n"name": "Paris",\n"population": 3500000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4710, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 11, 773, 1181, 7042, 374, 5008, 3460, 13, 358, 1744, 432, 594, 916, 220, 18, 3526, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 10696, 358, 1265, 1990, 15934, 429, 13, 4710, 14190, 11, 358, 19091, 429, 279, 7042, 7071, 646, 13289, 11649, 389, 279, 2530, 323, 279, 1042, 13, 576, 1196, 3207, 944, 13837, 264, 3953, 1042, 11, 773, 358, 1265, 4658, 728, 448, 279, 1429, 3213, 16045, 13, 358, 4411, 279, 7042, 374, 2163, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 4710, 7039, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 11136, 5711, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 2474, 279, 1196, 9733, 9625, 13, 4710, 40, 1265, 1281, 2704, 279, 6894, 525, 304, 6364, 311, 2506, 432, 2797, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 18, 11, 20, 15, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 13, 358, 3278, 3561, 419, 1119, 264, 4718, 1633, 13, 4710, 40, 1083, 1184, 311, 3042, 419, 304, 264, 1616, 429, 594, 4135, 311, 1349, 11, 773, 358, 3278, 990, 6169, 19482, 448, 54231, 15423, 323, 76602, 304, 279, 1290, 7482, 13, 2308, 27748, 76602, 311, 5648, 5975, 13, 4710, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 264, 10997, 448, 279, 6894, 323, 279, 12159, 2750, 13, 358, 3278, 1281, 2704, 311, 1273, 432, 311, 5978, 432, 594, 2697, 11, 714, 2474, 358, 2776, 1101, 4378, 432, 700, 11, 358, 3278, 9658, 432, 594, 4396, 3118, 389, 847, 6540, 13, 4710, 40, 1744, 429, 594, 678, 13, 576, 1196, 1101, 3880, 279, 1995, 304, 4718, 11, 773, 419, 1265, 26553, 862, 1681, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 18, 20, 15, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '56658b2933244243a08385a313a46347', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 405, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.4386703590862453, 'response_sent_to_client_ts': 1772743501.957534}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants both the general information and the population. So, I\'ll create an object with a "name" field for the capital, a "general_information" section that includes the administrative center, area, and government department, and a "population" section that includes the current population and a note about the data source.\n\nI should also add a "source" field to indicate where the population data comes from, which is the 2020 census. This makes the information more transparent and trustworthy.\n\nPutting it all together, I\'ll format the JSON with proper syntax, using double quotes for strings and ensuring that the keys are clear and descriptive. I\'ll make sure there are no typos and that the JSON is valid.\n\nFinally, I\'ll present the JSON in a code block so the user can easily copy and use it. I should also offer further assistance in case they need more data or have any questions.\n</think>{\n  "name": "Paris",\n  "population": 2174300\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 2176, 279, 4586, 1995, 323, 279, 7042, 13, 2055, 11, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 6722, 11, 264, 330, 24595, 35212, 1, 3772, 429, 5646, 279, 22707, 4126, 11, 3082, 11, 323, 3033, 9292, 11, 323, 264, 330, 44441, 1, 3772, 429, 5646, 279, 1482, 7042, 323, 264, 5185, 911, 279, 821, 2530, 382, 40, 1265, 1083, 912, 264, 330, 2427, 1, 2070, 311, 13216, 1380, 279, 7042, 821, 4041, 504, 11, 892, 374, 279, 220, 17, 15, 17, 15, 43602, 13, 1096, 3643, 279, 1995, 803, 17821, 323, 55942, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3561, 279, 4718, 448, 6169, 19482, 11, 1667, 1990, 17194, 369, 9069, 323, 22573, 429, 279, 6894, 525, 2797, 323, 52844, 13, 358, 3278, 1281, 2704, 1052, 525, 902, 13580, 966, 323, 429, 279, 4718, 374, 2697, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 304, 264, 2038, 2504, 773, 279, 1196, 646, 6707, 2975, 323, 990, 432, 13, 358, 1265, 1083, 3010, 4623, 12994, 304, 1142, 807, 1184, 803, 821, 476, 614, 894, 4755, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 198, 92, 151643], 'meta_info': {'id': '1d2ba11f50984a8b8f71be22be46bf0e', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 386, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.3249195176176727, 'response_sent_to_client_ts': 1772743504.2914493}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '370250f25dae42f0aa1c451517ad3b7a', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.288785716984421, 'response_sent_to_client_ts': 1772743504.603419}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'c1cf2a94758045ec950bd75869df09de', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.28870917623862624, 'response_sent_to_client_ts': 1772743504.6034324}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '6d8270a8ff11456b8a2a4d412842bd54', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.28866815427318215, 'response_sent_to_client_ts': 1772743504.6034372}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '4433e18452bd4c6f98929b673b4c1242', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 24.603730245959014, 'response_sent_to_client_ts': 1772743529.2140365}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user asked for the capital of France and its population in JSON format. I need to make sure I provide accurate information. First, I recall that the capital is Paris. \n\nNow, for the population, I think it\'s around 2 million, but it\'s important to get the most recent data. I remember that in 2023, Paris had a population estimate around 2,187,000. That seems about right. \n\nI should structure this into a JSON format. Oh, right, they specified JSON, so I\'ll need to use proper syntax with curly braces and keys. The keys should be "city" for the capital and "population" for the number.\n\nWait, should I include the country name as well? The user didn\'t specify, so I\'ll stick to the city and population as requested. \n\nPutting it all together, I\'ll format the response so it\'s clear and easy to read. No markdown, just plain JSON. Let me double-check to make sure there are no typos and that the numbers are correct. \n\nThat should fulfill the user\'s request accurately.\n</think>\n\n```json\n{\n  "capital": "Paris",\n  "population": 2187000\n}\n```', 'output_ids': [32313, 11, 773, 279, 1196, 4588, 369, 279, 6722, 315, 9625, 323, 1181, 7042, 304, 4718, 3561, 13, 358, 1184, 311, 1281, 2704, 358, 3410, 13382, 1995, 13, 5512, 11, 358, 19091, 429, 279, 6722, 374, 12095, 13, 4710, 7039, 11, 369, 279, 7042, 11, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 432, 594, 2989, 311, 633, 279, 1429, 3213, 821, 13, 358, 6099, 429, 304, 220, 17, 15, 17, 18, 11, 12095, 1030, 264, 7042, 16045, 2163, 220, 17, 11, 16, 23, 22, 11, 15, 15, 15, 13, 2938, 4977, 911, 1290, 13, 4710, 40, 1265, 5944, 419, 1119, 264, 4718, 3561, 13, 8670, 11, 1290, 11, 807, 5189, 4718, 11, 773, 358, 3278, 1184, 311, 990, 6169, 19482, 448, 68103, 59191, 323, 6894, 13, 576, 6894, 1265, 387, 330, 8926, 1, 369, 279, 6722, 323, 330, 44441, 1, 369, 279, 1372, 382, 14190, 11, 1265, 358, 2924, 279, 3146, 829, 438, 1632, 30, 576, 1196, 3207, 944, 13837, 11, 773, 358, 3278, 9214, 311, 279, 3283, 323, 7042, 438, 11223, 13, 4710, 97904, 432, 678, 3786, 11, 358, 3278, 3561, 279, 2033, 773, 432, 594, 2797, 323, 4135, 311, 1349, 13, 2308, 50494, 11, 1101, 14396, 4718, 13, 6771, 752, 1990, 15934, 311, 1281, 2704, 1052, 525, 902, 13580, 966, 323, 429, 279, 5109, 525, 4396, 13, 4710, 4792, 1265, 20423, 279, 1196, 594, 1681, 29257, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 23, 22, 15, 15, 15, 198, 532, 73594, 151643], 'meta_info': {'id': 'b8ac02da96e54ba3bc442bc6391497c4', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 261, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 3.3151612328365445, 'response_sent_to_client_ts': 1772743532.5390553}}</strong>



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


    [2026-03-05 20:45:34] INFO server_args.py:2039: Attention backend not specified. Use fa3 backend by default.


    [2026-03-05 20:45:34] INFO server_args.py:3146: Set soft_watchdog_timeout since in CI


    [2026-03-05 20:45:34] INFO engine.py:158: server_args=ServerArgs(model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.83, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=455656874, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser='deepseek-r1', tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.79s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:03<00:00,  1.65s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:03<00:00,  1.67s/it]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=62.65 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=62.65 GB):   5%|▌         | 1/20 [00:00<00:03,  5.87it/s]Capturing batches (bs=120 avail_mem=62.54 GB):   5%|▌         | 1/20 [00:00<00:03,  5.87it/s]Capturing batches (bs=112 avail_mem=62.54 GB):   5%|▌         | 1/20 [00:00<00:03,  5.87it/s]

    Capturing batches (bs=104 avail_mem=62.53 GB):   5%|▌         | 1/20 [00:00<00:03,  5.87it/s]Capturing batches (bs=96 avail_mem=62.53 GB):   5%|▌         | 1/20 [00:00<00:03,  5.87it/s] Capturing batches (bs=96 avail_mem=62.53 GB):  25%|██▌       | 5/20 [00:00<00:00, 17.17it/s]Capturing batches (bs=88 avail_mem=62.53 GB):  25%|██▌       | 5/20 [00:00<00:00, 17.17it/s]Capturing batches (bs=80 avail_mem=62.53 GB):  25%|██▌       | 5/20 [00:00<00:00, 17.17it/s]Capturing batches (bs=80 avail_mem=62.53 GB):  35%|███▌      | 7/20 [00:00<00:00, 18.07it/s]Capturing batches (bs=72 avail_mem=62.53 GB):  35%|███▌      | 7/20 [00:00<00:00, 18.07it/s]

    Capturing batches (bs=64 avail_mem=62.53 GB):  35%|███▌      | 7/20 [00:00<00:00, 18.07it/s]Capturing batches (bs=64 avail_mem=62.53 GB):  45%|████▌     | 9/20 [00:00<00:00, 17.50it/s]Capturing batches (bs=56 avail_mem=62.53 GB):  45%|████▌     | 9/20 [00:00<00:00, 17.50it/s]Capturing batches (bs=48 avail_mem=62.52 GB):  45%|████▌     | 9/20 [00:00<00:00, 17.50it/s]Capturing batches (bs=48 avail_mem=62.52 GB):  55%|█████▌    | 11/20 [00:00<00:00, 17.59it/s]Capturing batches (bs=40 avail_mem=62.52 GB):  55%|█████▌    | 11/20 [00:00<00:00, 17.59it/s]

    Capturing batches (bs=32 avail_mem=62.52 GB):  55%|█████▌    | 11/20 [00:00<00:00, 17.59it/s]Capturing batches (bs=32 avail_mem=62.52 GB):  65%|██████▌   | 13/20 [00:00<00:00, 17.74it/s]Capturing batches (bs=24 avail_mem=62.52 GB):  65%|██████▌   | 13/20 [00:00<00:00, 17.74it/s]Capturing batches (bs=16 avail_mem=62.52 GB):  65%|██████▌   | 13/20 [00:00<00:00, 17.74it/s]Capturing batches (bs=16 avail_mem=62.52 GB):  75%|███████▌  | 15/20 [00:00<00:00, 16.63it/s]Capturing batches (bs=12 avail_mem=62.52 GB):  75%|███████▌  | 15/20 [00:00<00:00, 16.63it/s]

    Capturing batches (bs=8 avail_mem=62.52 GB):  75%|███████▌  | 15/20 [00:00<00:00, 16.63it/s] Capturing batches (bs=4 avail_mem=62.52 GB):  75%|███████▌  | 15/20 [00:00<00:00, 16.63it/s]Capturing batches (bs=2 avail_mem=62.52 GB):  75%|███████▌  | 15/20 [00:00<00:00, 16.63it/s]Capturing batches (bs=2 avail_mem=62.52 GB):  95%|█████████▌| 19/20 [00:01<00:00, 21.34it/s]Capturing batches (bs=1 avail_mem=62.51 GB):  95%|█████████▌| 19/20 [00:01<00:00, 21.34it/s]Capturing batches (bs=1 avail_mem=62.51 GB): 100%|██████████| 20/20 [00:01<00:00, 18.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:13,  1.32s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:13,  1.32s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:44,  1.24it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:44,  1.24it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:30,  1.78it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:30,  1.78it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:21,  2.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:21,  2.41it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:16,  3.06it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:16,  3.06it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:13,  3.80it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:13,  3.80it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:10,  4.58it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:10,  4.58it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.44it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.44it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.44it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  7.06it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  7.06it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  7.06it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.51it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.51it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.51it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  9.97it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  9.97it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  9.97it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.77it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.77it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.77it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.77it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:02, 15.00it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:02, 15.00it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:02, 15.00it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:02, 15.00it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:02, 15.00it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:04<00:01, 20.02it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:04<00:01, 20.02it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:04<00:01, 20.02it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:04<00:01, 20.02it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.02it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.02it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 26.40it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 26.40it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 26.40it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 26.40it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 26.40it/s]

    Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 26.40it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 26.40it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 34.05it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 34.05it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 34.05it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 34.05it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 34.05it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 34.05it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 37.85it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 37.85it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 37.85it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 37.85it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 37.85it/s]

    Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 37.85it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s]Compiling num tokens (num_tokens=12):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s]Compiling num tokens (num_tokens=8):  78%|███████▊  | 45/58 [00:05<00:00, 36.23it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 56.65it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 56.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.51it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=61.35 GB):   2%|▏         | 1/58 [00:00<00:17,  3.20it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.32 GB):   2%|▏         | 1/58 [00:00<00:17,  3.20it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=61.32 GB):   3%|▎         | 2/58 [00:00<00:16,  3.46it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.32 GB):   3%|▎         | 2/58 [00:00<00:16,  3.46it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=61.32 GB):   5%|▌         | 3/58 [00:00<00:14,  3.71it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.32 GB):   5%|▌         | 3/58 [00:00<00:14,  3.71it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=61.32 GB):   7%|▋         | 4/58 [00:01<00:13,  4.03it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.33 GB):   7%|▋         | 4/58 [00:01<00:13,  4.03it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.33 GB):   9%|▊         | 5/58 [00:01<00:12,  4.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.33 GB):   9%|▊         | 5/58 [00:01<00:12,  4.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.33 GB):  10%|█         | 6/58 [00:01<00:11,  4.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.34 GB):  10%|█         | 6/58 [00:01<00:11,  4.70it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=61.34 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.34 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.34 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.34 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.54it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=61.34 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.07it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.35 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.07it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.35 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.34 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.51it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=61.34 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.34 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.34 GB):  21%|██        | 12/58 [00:02<00:06,  7.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.34 GB):  21%|██        | 12/58 [00:02<00:06,  7.58it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=61.34 GB):  21%|██        | 12/58 [00:02<00:06,  7.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.34 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.34 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.34 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.81it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=61.34 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.34 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.34 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.34 GB):  31%|███       | 18/58 [00:02<00:03, 10.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.34 GB):  31%|███       | 18/58 [00:02<00:03, 10.81it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=61.34 GB):  31%|███       | 18/58 [00:02<00:03, 10.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.34 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.34 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.73it/s]Capturing num tokens (num_tokens=960 avail_mem=61.34 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.73it/s] Capturing num tokens (num_tokens=896 avail_mem=61.33 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.73it/s]Capturing num tokens (num_tokens=896 avail_mem=61.33 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.02it/s]Capturing num tokens (num_tokens=832 avail_mem=61.33 GB):  40%|███▉      | 23/58 [00:02<00:02, 15.02it/s]

    Capturing num tokens (num_tokens=768 avail_mem=61.33 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.02it/s]Capturing num tokens (num_tokens=704 avail_mem=61.32 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.02it/s]Capturing num tokens (num_tokens=704 avail_mem=61.32 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.10it/s]Capturing num tokens (num_tokens=640 avail_mem=61.32 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.10it/s]Capturing num tokens (num_tokens=576 avail_mem=61.31 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.10it/s]Capturing num tokens (num_tokens=576 avail_mem=61.31 GB):  48%|████▊     | 28/58 [00:03<00:01, 17.96it/s]Capturing num tokens (num_tokens=512 avail_mem=61.29 GB):  48%|████▊     | 28/58 [00:03<00:01, 17.96it/s]

    Capturing num tokens (num_tokens=480 avail_mem=61.29 GB):  48%|████▊     | 28/58 [00:03<00:01, 17.96it/s]Capturing num tokens (num_tokens=480 avail_mem=61.29 GB):  52%|█████▏    | 30/58 [00:03<00:01, 18.32it/s]Capturing num tokens (num_tokens=448 avail_mem=61.28 GB):  52%|█████▏    | 30/58 [00:03<00:01, 18.32it/s]Capturing num tokens (num_tokens=416 avail_mem=61.26 GB):  52%|█████▏    | 30/58 [00:03<00:01, 18.32it/s]Capturing num tokens (num_tokens=384 avail_mem=60.78 GB):  52%|█████▏    | 30/58 [00:03<00:01, 18.32it/s]Capturing num tokens (num_tokens=384 avail_mem=60.78 GB):  57%|█████▋    | 33/58 [00:03<00:01, 20.17it/s]Capturing num tokens (num_tokens=352 avail_mem=60.67 GB):  57%|█████▋    | 33/58 [00:03<00:01, 20.17it/s]

    Capturing num tokens (num_tokens=320 avail_mem=60.62 GB):  57%|█████▋    | 33/58 [00:03<00:01, 20.17it/s]Capturing num tokens (num_tokens=288 avail_mem=60.61 GB):  57%|█████▋    | 33/58 [00:03<00:01, 20.17it/s]Capturing num tokens (num_tokens=256 avail_mem=60.61 GB):  57%|█████▋    | 33/58 [00:03<00:01, 20.17it/s]Capturing num tokens (num_tokens=256 avail_mem=60.61 GB):  64%|██████▍   | 37/58 [00:03<00:00, 24.99it/s]Capturing num tokens (num_tokens=240 avail_mem=60.61 GB):  64%|██████▍   | 37/58 [00:03<00:00, 24.99it/s]Capturing num tokens (num_tokens=224 avail_mem=60.60 GB):  64%|██████▍   | 37/58 [00:03<00:00, 24.99it/s]Capturing num tokens (num_tokens=208 avail_mem=60.60 GB):  64%|██████▍   | 37/58 [00:03<00:00, 24.99it/s]Capturing num tokens (num_tokens=192 avail_mem=60.59 GB):  64%|██████▍   | 37/58 [00:03<00:00, 24.99it/s]Capturing num tokens (num_tokens=192 avail_mem=60.59 GB):  71%|███████   | 41/58 [00:03<00:00, 28.70it/s]Capturing num tokens (num_tokens=176 avail_mem=60.59 GB):  71%|███████   | 41/58 [00:03<00:00, 28.70it/s]

    Capturing num tokens (num_tokens=160 avail_mem=60.58 GB):  71%|███████   | 41/58 [00:03<00:00, 28.70it/s]Capturing num tokens (num_tokens=144 avail_mem=60.58 GB):  71%|███████   | 41/58 [00:03<00:00, 28.70it/s]Capturing num tokens (num_tokens=128 avail_mem=60.59 GB):  71%|███████   | 41/58 [00:03<00:00, 28.70it/s]Capturing num tokens (num_tokens=128 avail_mem=60.59 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.58it/s]Capturing num tokens (num_tokens=112 avail_mem=60.59 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.58it/s]Capturing num tokens (num_tokens=96 avail_mem=60.58 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.58it/s] Capturing num tokens (num_tokens=80 avail_mem=60.58 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.58it/s]Capturing num tokens (num_tokens=64 avail_mem=60.58 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.58it/s]Capturing num tokens (num_tokens=64 avail_mem=60.58 GB):  84%|████████▍ | 49/58 [00:03<00:00, 33.90it/s]Capturing num tokens (num_tokens=48 avail_mem=60.58 GB):  84%|████████▍ | 49/58 [00:03<00:00, 33.90it/s]

    Capturing num tokens (num_tokens=32 avail_mem=60.57 GB):  84%|████████▍ | 49/58 [00:03<00:00, 33.90it/s]Capturing num tokens (num_tokens=28 avail_mem=60.57 GB):  84%|████████▍ | 49/58 [00:03<00:00, 33.90it/s]Capturing num tokens (num_tokens=24 avail_mem=60.57 GB):  84%|████████▍ | 49/58 [00:03<00:00, 33.90it/s]Capturing num tokens (num_tokens=24 avail_mem=60.57 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.56it/s]Capturing num tokens (num_tokens=20 avail_mem=60.56 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.56it/s]Capturing num tokens (num_tokens=16 avail_mem=60.55 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.56it/s]Capturing num tokens (num_tokens=12 avail_mem=60.55 GB):  91%|█████████▏| 53/58 [00:03<00:00, 35.56it/s]Capturing num tokens (num_tokens=8 avail_mem=60.55 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.56it/s] Capturing num tokens (num_tokens=4 avail_mem=60.54 GB):  91%|█████████▏| 53/58 [00:04<00:00, 35.56it/s]Capturing num tokens (num_tokens=4 avail_mem=60.54 GB): 100%|██████████| 58/58 [00:04<00:00, 37.68it/s]Capturing num tokens (num_tokens=4 avail_mem=60.54 GB): 100%|██████████| 58/58 [00:04<00:00, 14.26it/s]


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
    
    Generated text: Okay, so I need to figure out the population of the capital of France. The user asked for the information and population in JSON format. Hmm, let me start by recalling some basic facts about France. I know that Paris is the capital, that's for sure. Now, about the population... I remember that Paris is a very large city, but I'm not exactly sure about the exact number. I think it's over 3 million people, maybe around 3.5 million? But I'm not entirely certain.
    
    Wait, could it be more than that? I've heard that major cities in developed countries tend to have increasing populations. Maybe the population is now over 4 million? I'm a bit confused. I should probably check if there's a recent estimate available. I know that population figures can change due to births, deaths, and migration. So, the number might be an approximate figure.
    
    Also, when they say the capital, does that include the entire metropolitan area or just the city proper? I think in some contexts, the metropolitan area includes surrounding suburbs and that might have a larger population. But the user specifically asked for the capital, so I should focus on Paris itself, not the broader metropolitan area.
    
    Let me think about any recent data I might have come across. I believe that in 2020, the population of Paris was around 3,500,000. But I'm not sure if that's the most recent figure. Maybe it's a bit higher now. I should make sure to provide the most accurate information possible, even if it's an estimate.
    
    Another thing to consider is that population statistics can vary slightly depending on the source. Some organizations might report slightly different numbers based on when the data was collected. So, it's important to mention that the figure is approximate to avoid giving a misleadingly precise number.
    
    Wait, the user wants the information in JSON format. So, I need to structure the response accordingly. The JSON should include the capital name, its population, and a note explaining that the figure is approximate. That way, the user gets clear and concise information in a structured format.
    
    I should also ensure that the JSON syntax is correct, using proper quotation marks and commas. Maybe I'll write it out like this:
    
    {
      "capital": "Paris",
      "population": 3500000,
      "note": "Population figure is approximate and may vary slightly depending on the source and time of measurement."
    }
    
    Does that cover everything? It provides the capital, the population, and a note about the accuracy of the number. I think that's what the user is looking for. I'll go ahead and present it this way.
    </think>
    
    ```json
    {
      "capital": "Paris",
      "population": 3500000,
      "note": "Population figure is approximate and may vary slightly depending on the source and time of measurement."
    }
    ```



```python
llm.shutdown()
```
