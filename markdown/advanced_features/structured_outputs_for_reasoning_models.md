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

    [2026-03-06 19:27:26] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-06 19:27:26] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-06 19:27:26] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-06 19:27:31] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-06 19:27:31] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-06 19:27:31] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /root/actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-06 19:27:33] INFO server_args.py:2048: Attention backend not specified. Use fa3 backend by default.
    [2026-03-06 19:27:33] INFO server_args.py:3155: Set soft_watchdog_timeout since in CI


    [2026-03-06 19:27:39] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-06 19:27:39] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-06 19:27:39] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-06 19:27:39] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-06 19:27:39] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-06 19:27:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-06 19:27:44] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-06 19:27:44] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-06 19:27:44] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.62s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:03<00:00,  1.82s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:03<00:00,  1.79s/it]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=122.75 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=122.75 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.45it/s]Capturing batches (bs=2 avail_mem=122.69 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.45it/s]Capturing batches (bs=1 avail_mem=122.68 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.45it/s]Capturing batches (bs=1 avail_mem=122.68 GB): 100%|██████████| 3/3 [00:00<00:00, 10.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:03,  3.23s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:03,  3.23s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:27,  1.56s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:27,  1.56s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.16it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.16it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.79it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.79it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.52it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.52it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.30it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.30it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.19it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.19it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.19it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.87it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.87it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.87it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.44it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.44it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.44it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.99it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.99it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.99it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.92it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.92it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.92it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.92it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.29it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.29it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.29it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.29it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.29it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.53it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.53it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.53it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.53it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.53it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.53it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 27.44it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 27.44it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 27.44it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 27.44it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 27.44it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 27.44it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 27.44it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 34.66it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 34.66it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 34.66it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 34.66it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 34.66it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 34.66it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 34.66it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:00, 34.66it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 42.51it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 42.51it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 42.51it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 42.51it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 42.51it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 42.51it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 42.51it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 42.51it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 48.93it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 48.93it/s]

    Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 48.93it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 48.93it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 48.93it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 48.93it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 48.93it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 48.93it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:06<00:00, 48.93it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 56.52it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 56.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=105.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=105.73 GB):   2%|▏         | 1/58 [00:00<00:15,  3.58it/s]Capturing num tokens (num_tokens=7680 avail_mem=105.70 GB):   2%|▏         | 1/58 [00:00<00:15,  3.58it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=105.70 GB):   3%|▎         | 2/58 [00:00<00:15,  3.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=105.69 GB):   3%|▎         | 2/58 [00:00<00:15,  3.60it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=105.69 GB):   5%|▌         | 3/58 [00:00<00:14,  3.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=105.70 GB):   5%|▌         | 3/58 [00:00<00:14,  3.86it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=105.70 GB):   7%|▋         | 4/58 [00:01<00:13,  4.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=105.70 GB):   7%|▋         | 4/58 [00:01<00:13,  4.15it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=105.70 GB):   9%|▊         | 5/58 [00:01<00:12,  4.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=105.70 GB):   9%|▊         | 5/58 [00:01<00:12,  4.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=105.70 GB):  10%|█         | 6/58 [00:01<00:11,  4.71it/s]Capturing num tokens (num_tokens=5120 avail_mem=105.70 GB):  10%|█         | 6/58 [00:01<00:11,  4.71it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=105.70 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=105.70 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=105.70 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=105.70 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.40it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=105.70 GB):  16%|█▌        | 9/58 [00:01<00:10,  4.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=105.71 GB):  16%|█▌        | 9/58 [00:01<00:10,  4.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=105.71 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=105.70 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.49it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=105.70 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.21it/s]Capturing num tokens (num_tokens=3328 avail_mem=105.70 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.21it/s]Capturing num tokens (num_tokens=3328 avail_mem=105.70 GB):  21%|██        | 12/58 [00:02<00:06,  6.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=105.70 GB):  21%|██        | 12/58 [00:02<00:06,  6.99it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=105.70 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=105.70 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=105.70 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=105.70 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=105.69 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.13it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=105.69 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=105.69 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=105.69 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=105.69 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=105.69 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=105.68 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.44it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=105.50 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=105.50 GB):  36%|███▌      | 21/58 [00:03<00:02, 12.40it/s]Capturing num tokens (num_tokens=960 avail_mem=104.52 GB):  36%|███▌      | 21/58 [00:03<00:02, 12.40it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=104.52 GB):  36%|███▌      | 21/58 [00:03<00:02, 12.40it/s]Capturing num tokens (num_tokens=896 avail_mem=104.52 GB):  40%|███▉      | 23/58 [00:03<00:03, 10.18it/s]Capturing num tokens (num_tokens=832 avail_mem=104.51 GB):  40%|███▉      | 23/58 [00:03<00:03, 10.18it/s]

    Capturing num tokens (num_tokens=768 avail_mem=104.51 GB):  40%|███▉      | 23/58 [00:03<00:03, 10.18it/s]Capturing num tokens (num_tokens=768 avail_mem=104.51 GB):  43%|████▎     | 25/58 [00:03<00:03,  9.68it/s]Capturing num tokens (num_tokens=704 avail_mem=104.50 GB):  43%|████▎     | 25/58 [00:03<00:03,  9.68it/s]

    Capturing num tokens (num_tokens=640 avail_mem=104.48 GB):  43%|████▎     | 25/58 [00:03<00:03,  9.68it/s]Capturing num tokens (num_tokens=640 avail_mem=104.48 GB):  47%|████▋     | 27/58 [00:03<00:03,  8.13it/s]Capturing num tokens (num_tokens=576 avail_mem=104.47 GB):  47%|████▋     | 27/58 [00:03<00:03,  8.13it/s]

    Capturing num tokens (num_tokens=576 avail_mem=104.47 GB):  48%|████▊     | 28/58 [00:03<00:03,  8.13it/s]Capturing num tokens (num_tokens=512 avail_mem=103.98 GB):  48%|████▊     | 28/58 [00:03<00:03,  8.13it/s]Capturing num tokens (num_tokens=480 avail_mem=103.81 GB):  48%|████▊     | 28/58 [00:04<00:03,  8.13it/s]

    Capturing num tokens (num_tokens=480 avail_mem=103.81 GB):  52%|█████▏    | 30/58 [00:04<00:03,  8.55it/s]Capturing num tokens (num_tokens=448 avail_mem=103.80 GB):  52%|█████▏    | 30/58 [00:04<00:03,  8.55it/s]Capturing num tokens (num_tokens=448 avail_mem=103.80 GB):  53%|█████▎    | 31/58 [00:04<00:03,  8.79it/s]Capturing num tokens (num_tokens=416 avail_mem=103.80 GB):  53%|█████▎    | 31/58 [00:04<00:03,  8.79it/s]

    Capturing num tokens (num_tokens=416 avail_mem=103.80 GB):  55%|█████▌    | 32/58 [00:04<00:02,  8.93it/s]Capturing num tokens (num_tokens=384 avail_mem=103.79 GB):  55%|█████▌    | 32/58 [00:04<00:02,  8.93it/s]Capturing num tokens (num_tokens=352 avail_mem=103.79 GB):  55%|█████▌    | 32/58 [00:04<00:02,  8.93it/s]Capturing num tokens (num_tokens=352 avail_mem=103.79 GB):  59%|█████▊    | 34/58 [00:04<00:02,  9.45it/s]Capturing num tokens (num_tokens=320 avail_mem=103.78 GB):  59%|█████▊    | 34/58 [00:04<00:02,  9.45it/s]

    Capturing num tokens (num_tokens=320 avail_mem=103.78 GB):  60%|██████    | 35/58 [00:04<00:02,  9.41it/s]Capturing num tokens (num_tokens=288 avail_mem=103.78 GB):  60%|██████    | 35/58 [00:04<00:02,  9.41it/s]Capturing num tokens (num_tokens=256 avail_mem=103.77 GB):  60%|██████    | 35/58 [00:04<00:02,  9.41it/s]Capturing num tokens (num_tokens=256 avail_mem=103.77 GB):  64%|██████▍   | 37/58 [00:04<00:02,  9.72it/s]Capturing num tokens (num_tokens=240 avail_mem=103.76 GB):  64%|██████▍   | 37/58 [00:04<00:02,  9.72it/s]

    Capturing num tokens (num_tokens=224 avail_mem=103.76 GB):  64%|██████▍   | 37/58 [00:04<00:02,  9.72it/s]Capturing num tokens (num_tokens=224 avail_mem=103.76 GB):  67%|██████▋   | 39/58 [00:05<00:01,  9.70it/s]Capturing num tokens (num_tokens=208 avail_mem=103.75 GB):  67%|██████▋   | 39/58 [00:05<00:01,  9.70it/s]

    Capturing num tokens (num_tokens=208 avail_mem=103.75 GB):  69%|██████▉   | 40/58 [00:05<00:01,  9.74it/s]Capturing num tokens (num_tokens=192 avail_mem=103.75 GB):  69%|██████▉   | 40/58 [00:05<00:01,  9.74it/s]Capturing num tokens (num_tokens=176 avail_mem=103.74 GB):  69%|██████▉   | 40/58 [00:05<00:01,  9.74it/s]

    Capturing num tokens (num_tokens=176 avail_mem=103.74 GB):  72%|███████▏  | 42/58 [00:05<00:01,  9.09it/s]Capturing num tokens (num_tokens=160 avail_mem=103.73 GB):  72%|███████▏  | 42/58 [00:05<00:01,  9.09it/s]Capturing num tokens (num_tokens=160 avail_mem=103.73 GB):  74%|███████▍  | 43/58 [00:05<00:01,  9.25it/s]Capturing num tokens (num_tokens=144 avail_mem=103.73 GB):  74%|███████▍  | 43/58 [00:05<00:01,  9.25it/s]

    Capturing num tokens (num_tokens=144 avail_mem=103.73 GB):  76%|███████▌  | 44/58 [00:05<00:01,  8.21it/s]Capturing num tokens (num_tokens=128 avail_mem=103.74 GB):  76%|███████▌  | 44/58 [00:05<00:01,  8.21it/s]Capturing num tokens (num_tokens=112 avail_mem=103.73 GB):  76%|███████▌  | 44/58 [00:05<00:01,  8.21it/s]Capturing num tokens (num_tokens=112 avail_mem=103.73 GB):  79%|███████▉  | 46/58 [00:05<00:01,  9.93it/s]Capturing num tokens (num_tokens=96 avail_mem=103.73 GB):  79%|███████▉  | 46/58 [00:05<00:01,  9.93it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=103.72 GB):  79%|███████▉  | 46/58 [00:05<00:01,  9.93it/s]Capturing num tokens (num_tokens=80 avail_mem=103.72 GB):  83%|████████▎ | 48/58 [00:05<00:00, 11.31it/s]Capturing num tokens (num_tokens=64 avail_mem=103.72 GB):  83%|████████▎ | 48/58 [00:05<00:00, 11.31it/s]Capturing num tokens (num_tokens=48 avail_mem=103.71 GB):  83%|████████▎ | 48/58 [00:06<00:00, 11.31it/s]Capturing num tokens (num_tokens=48 avail_mem=103.71 GB):  86%|████████▌ | 50/58 [00:06<00:00, 13.18it/s]Capturing num tokens (num_tokens=32 avail_mem=103.71 GB):  86%|████████▌ | 50/58 [00:06<00:00, 13.18it/s]

    Capturing num tokens (num_tokens=28 avail_mem=103.70 GB):  86%|████████▌ | 50/58 [00:06<00:00, 13.18it/s]Capturing num tokens (num_tokens=24 avail_mem=103.69 GB):  86%|████████▌ | 50/58 [00:06<00:00, 13.18it/s]Capturing num tokens (num_tokens=24 avail_mem=103.69 GB):  91%|█████████▏| 53/58 [00:06<00:00, 16.08it/s]Capturing num tokens (num_tokens=20 avail_mem=103.69 GB):  91%|█████████▏| 53/58 [00:06<00:00, 16.08it/s]Capturing num tokens (num_tokens=16 avail_mem=103.68 GB):  91%|█████████▏| 53/58 [00:06<00:00, 16.08it/s]Capturing num tokens (num_tokens=12 avail_mem=103.67 GB):  91%|█████████▏| 53/58 [00:06<00:00, 16.08it/s]

    Capturing num tokens (num_tokens=12 avail_mem=103.67 GB):  97%|█████████▋| 56/58 [00:06<00:00, 18.86it/s]Capturing num tokens (num_tokens=8 avail_mem=103.67 GB):  97%|█████████▋| 56/58 [00:06<00:00, 18.86it/s] Capturing num tokens (num_tokens=4 avail_mem=103.66 GB):  97%|█████████▋| 56/58 [00:06<00:00, 18.86it/s]Capturing num tokens (num_tokens=4 avail_mem=103.66 GB): 100%|██████████| 58/58 [00:06<00:00,  9.09it/s]


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


<strong style='color: #00008B;'>reasoing_content: Alright, I need to figure out how to respond to the user's query. They're in New York and want the current date and time, along with the weather. <br><br>First, I'll look at the functions available. There's 'get_current_date' and 'get_current_weather'. The user mentioned New York, but I should check if it's a city or state abbreviation. New York is a state, so I'll use 'NY' for the state parameter in 'get_current_weather'. <br><br>For the date, I'll use 'get_current_date' with the timezone set to 'America/New_York'. <br><br>I need to call both functions and include their parameters. I'll structure the response using the specified format, making sure each function call is on a single line with the correct parameters. <br><br>I'll also remember to add the sources where I got the information, even if it's just using the available functions. <br><br>Putting it all together, I'll send the two function calls as per the instructions.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "celsius"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, the JSON will have the city name, population, and a concise description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid any errors.\n\nFinally, I\'ll present the JSON to the user, ensuring it\'s clear and well-structured. I\'ll double-check the population number to confirm it\'s up to date and accurate. That should fulfill the user\'s request effectively.\n</think>{\n\n"name": "Paris",\n"population": 2174300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 279, 4718, 686, 614, 279, 3283, 829, 11, 7042, 11, 323, 264, 63594, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 894, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 22573, 432, 594, 2797, 323, 1632, 12, 51143, 13, 358, 3278, 1990, 15934, 279, 7042, 1372, 311, 7683, 432, 594, 705, 311, 2400, 323, 13382, 13, 2938, 1265, 20423, 279, 1196, 594, 1681, 13444, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '63e05ddfbb7341f5873f80da48e98e72', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 10.138862991239876, 'response_sent_to_client_ts': 1772825313.4144857}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user is a student working on a project or maybe a developer integrating this data into an app. Either way, providing the information in a structured JSON format is likely helpful for their needs. I should ensure the data is accurate to avoid any confusion.\n\nAlso, considering the population figure, I should note that it\'s an approximate number because population figures can change over time due to various factors like births, deaths, and migration. It\'s good to be transparent about that so the user knows the data is current as of a specific point in time.\n\nIn summary, I need to provide a clear, concise JSON response with the correct capital and population, making sure the syntax is correct and the information is accurate.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 374, 264, 5458, 3238, 389, 264, 2390, 476, 7196, 264, 15754, 53852, 419, 821, 1119, 458, 906, 13, 20988, 1616, 11, 8241, 279, 1995, 304, 264, 32930, 4718, 3561, 374, 4363, 10950, 369, 862, 3880, 13, 358, 1265, 5978, 279, 821, 374, 13382, 311, 5648, 894, 21340, 382, 13394, 11, 12831, 279, 7042, 7071, 11, 358, 1265, 5185, 429, 432, 594, 458, 44868, 1372, 1576, 7042, 12396, 646, 2297, 916, 882, 4152, 311, 5257, 9363, 1075, 65232, 11, 16375, 11, 323, 11906, 13, 1084, 594, 1661, 311, 387, 17821, 911, 429, 773, 279, 1196, 8788, 279, 821, 374, 1482, 438, 315, 264, 3151, 1459, 304, 882, 382, 641, 12126, 11, 358, 1184, 311, 3410, 264, 2797, 11, 63594, 4718, 2033, 448, 279, 4396, 6722, 323, 7042, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 1995, 374, 13382, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': '721ac73e13ee48a6a587756f7513281b', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 456, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 2.2435304699465632, 'response_sent_to_client_ts': 1772825315.6678824}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '363f4a2e999144caaec4190256f4adbd', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.13059328915551305, 'response_sent_to_client_ts': 1772825315.820149}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '01454350afb549b895bdadeedb281481', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.1305344020947814, 'response_sent_to_client_ts': 1772825315.8201587}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '327e0ae178b54503857fe4bb8c2906bf', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.13048763386905193, 'response_sent_to_client_ts': 1772825315.8201635}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\) \n\\( n \\) \\( m \\)', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8, 715, 44292, 308, 1124, 8, 17767, 296, 1124, 8], 'meta_info': {'id': '9880822316434db19a5d44aa3f2567c6', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 13.60866368515417, 'response_sent_to_client_ts': 1772825329.4343982}}


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


<strong style='color: #00008B;'>{'text': 'Okay, so I need to help the user by giving them the information and population of the capital of France in JSON format. Alright, first, let\'s break this down. The user, me, need to provide the exact information about the capital city of France. \n\nHmm, I know that the capital of France is Paris. That\'s pretty much a given. Now, I should confirm the population. Wait, population figures can change over time. I think the current population is around 2 million. But I\'m not 100% sure if it\'s 2.16 million or something else. Maybe I should look that up or remember from recent data.\n\nWait, in my knowledge cutoff in 2023, I think the population is approximately 2,165,546. Let me make sure I got that right. Yeah, so I\'ll include that number. \n\nNow, the user wants this in JSON format. JSON is a data interchange format, so I need to structure it properly. The main points are that the capital is Paris, the country is France, and the population is 2,165,546. \n\nI should present this in a JSON format. So I\'ll create a main object with a key "capital" containing an object with "name", "country", and "population" fields. That way, it\'s organized and easy to read.\n\nI should also provide an example of how to use the JSON data in an application for clarity. That will help the user understand how to access the information. I\'ll make sure the example is straightforward, using console.log to display the JSON content.\n\nWait, do I need to include any units for the population? Like, should it be 2,165,546 people or just the number? I think it\'s just the number, but adding a note might be helpful. Oh, right, in the JSON example, I don\'t add explanations, but in the response, I can mention that it\'s the approximate population.\n\nDid I cover everything? The main points: capital is Paris, country is France, population is correct, and it\'s in JSON format with an example. I think that covers what the user asked for. Let me put it all together now.\n</think>\n\nHere is the information about the capital of France in JSON format:\n\n```json\n{\n  "capital": {\n    "name": "Paris",\n    "country": "France",\n    "population": 2165546\n  }\n}\n```\n\nExample code to use this JSON data:\n\n```javascript\nconst capitalData = {\n  "capital": {\n    "name": "Paris",\n    "country": "France",\n    "population": 2165546\n  }\n};\n\nconsole.log(capitalData);\n```', 'output_ids': [32313, 11, 773, 358, 1184, 311, 1492, 279, 1196, 553, 7086, 1105, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 97593, 11, 1156, 11, 1077, 594, 1438, 419, 1495, 13, 576, 1196, 11, 752, 11, 1184, 311, 3410, 279, 4734, 1995, 911, 279, 6722, 3283, 315, 9625, 13, 4710, 80022, 11, 358, 1414, 429, 279, 6722, 315, 9625, 374, 12095, 13, 2938, 594, 5020, 1753, 264, 2661, 13, 4695, 11, 358, 1265, 7683, 279, 7042, 13, 13824, 11, 7042, 12396, 646, 2297, 916, 882, 13, 358, 1744, 279, 1482, 7042, 374, 2163, 220, 17, 3526, 13, 1988, 358, 2776, 537, 220, 16, 15, 15, 4, 2704, 421, 432, 594, 220, 17, 13, 16, 21, 3526, 476, 2494, 770, 13, 10696, 358, 1265, 1401, 429, 705, 476, 6099, 504, 3213, 821, 382, 14190, 11, 304, 847, 6540, 44279, 304, 220, 17, 15, 17, 18, 11, 358, 1744, 279, 7042, 374, 13187, 220, 17, 11, 16, 21, 20, 11, 20, 19, 21, 13, 6771, 752, 1281, 2704, 358, 2684, 429, 1290, 13, 21607, 11, 773, 358, 3278, 2924, 429, 1372, 13, 4710, 7039, 11, 279, 1196, 6801, 419, 304, 4718, 3561, 13, 4718, 374, 264, 821, 51263, 3561, 11, 773, 358, 1184, 311, 5944, 432, 10277, 13, 576, 1887, 3501, 525, 429, 279, 6722, 374, 12095, 11, 279, 3146, 374, 9625, 11, 323, 279, 7042, 374, 220, 17, 11, 16, 21, 20, 11, 20, 19, 21, 13, 4710, 40, 1265, 3042, 419, 304, 264, 4718, 3561, 13, 2055, 358, 3278, 1855, 264, 1887, 1633, 448, 264, 1376, 330, 65063, 1, 8482, 458, 1633, 448, 330, 606, 497, 330, 11141, 497, 323, 330, 44441, 1, 5043, 13, 2938, 1616, 11, 432, 594, 16645, 323, 4135, 311, 1349, 382, 40, 1265, 1083, 3410, 458, 3110, 315, 1246, 311, 990, 279, 4718, 821, 304, 458, 3766, 369, 31273, 13, 2938, 686, 1492, 279, 1196, 3535, 1246, 311, 2615, 279, 1995, 13, 358, 3278, 1281, 2704, 279, 3110, 374, 30339, 11, 1667, 2339, 1665, 311, 3037, 279, 4718, 2213, 382, 14190, 11, 653, 358, 1184, 311, 2924, 894, 8153, 369, 279, 7042, 30, 8909, 11, 1265, 432, 387, 220, 17, 11, 16, 21, 20, 11, 20, 19, 21, 1251, 476, 1101, 279, 1372, 30, 358, 1744, 432, 594, 1101, 279, 1372, 11, 714, 7842, 264, 5185, 2578, 387, 10950, 13, 8670, 11, 1290, 11, 304, 279, 4718, 3110, 11, 358, 1513, 944, 912, 40841, 11, 714, 304, 279, 2033, 11, 358, 646, 6286, 429, 432, 594, 279, 44868, 7042, 382, 6986, 358, 3421, 4297, 30, 576, 1887, 3501, 25, 6722, 374, 12095, 11, 3146, 374, 9625, 11, 7042, 374, 4396, 11, 323, 432, 594, 304, 4718, 3561, 448, 458, 3110, 13, 358, 1744, 429, 14521, 1128, 279, 1196, 4588, 369, 13, 6771, 752, 2182, 432, 678, 3786, 1431, 624, 151649, 271, 8420, 374, 279, 1995, 911, 279, 6722, 315, 9625, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 341, 262, 330, 606, 788, 330, 59604, 756, 262, 330, 11141, 788, 330, 49000, 756, 262, 330, 44441, 788, 220, 17, 16, 21, 20, 20, 19, 21, 198, 220, 456, 532, 13874, 19324, 13314, 2038, 311, 990, 419, 4718, 821, 1447, 73594, 14073, 198, 1024, 6722, 1043, 284, 341, 220, 330, 65063, 788, 341, 262, 330, 606, 788, 330, 59604, 756, 262, 330, 11141, 788, 330, 49000, 756, 262, 330, 44441, 788, 220, 17, 16, 21, 20, 20, 19, 21, 198, 220, 456, 2315, 5354, 1665, 51386, 2174, 1043, 317, 73594, 151643], 'meta_info': {'id': '06d8d40ae68d4f5d95a719a03782a7f9', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 583, 'cached_tokens': 22, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 6.1217183149419725, 'response_sent_to_client_ts': 1772825335.564394}}</strong>



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


    [2026-03-06 19:28:58] INFO server_args.py:2048: Attention backend not specified. Use fa3 backend by default.


    [2026-03-06 19:28:58] INFO server_args.py:3155: Set soft_watchdog_timeout since in CI


    [2026-03-06 19:28:58] INFO engine.py:177: server_args=ServerArgs(model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.903, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=283234276, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='error', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser='deepseek-r1', tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method=None, kt_cpuinfer=None, kt_threadpool_count=None, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
    <frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.25s/it]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.24s/it]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.24s/it]
    


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=122.61 GB):   0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=122.61 GB):   5%|▌         | 1/20 [00:00<00:03,  5.62it/s]Capturing batches (bs=120 avail_mem=122.49 GB):   5%|▌         | 1/20 [00:00<00:03,  5.62it/s]

    Capturing batches (bs=112 avail_mem=122.49 GB):   5%|▌         | 1/20 [00:00<00:03,  5.62it/s]Capturing batches (bs=104 avail_mem=122.49 GB):   5%|▌         | 1/20 [00:00<00:03,  5.62it/s]Capturing batches (bs=104 avail_mem=122.49 GB):  20%|██        | 4/20 [00:00<00:00, 16.00it/s]Capturing batches (bs=96 avail_mem=122.48 GB):  20%|██        | 4/20 [00:00<00:00, 16.00it/s] Capturing batches (bs=88 avail_mem=122.48 GB):  20%|██        | 4/20 [00:00<00:00, 16.00it/s]Capturing batches (bs=80 avail_mem=122.47 GB):  20%|██        | 4/20 [00:00<00:00, 16.00it/s]

    Capturing batches (bs=80 avail_mem=122.47 GB):  35%|███▌      | 7/20 [00:00<00:00, 15.58it/s]Capturing batches (bs=72 avail_mem=122.47 GB):  35%|███▌      | 7/20 [00:00<00:00, 15.58it/s]Capturing batches (bs=64 avail_mem=122.46 GB):  35%|███▌      | 7/20 [00:00<00:00, 15.58it/s]Capturing batches (bs=56 avail_mem=122.46 GB):  35%|███▌      | 7/20 [00:00<00:00, 15.58it/s]Capturing batches (bs=56 avail_mem=122.46 GB):  50%|█████     | 10/20 [00:00<00:00, 17.55it/s]Capturing batches (bs=48 avail_mem=122.45 GB):  50%|█████     | 10/20 [00:00<00:00, 17.55it/s]

    Capturing batches (bs=40 avail_mem=121.39 GB):  50%|█████     | 10/20 [00:00<00:00, 17.55it/s]Capturing batches (bs=40 avail_mem=121.39 GB):  60%|██████    | 12/20 [00:00<00:00, 15.55it/s]Capturing batches (bs=32 avail_mem=121.38 GB):  60%|██████    | 12/20 [00:00<00:00, 15.55it/s]Capturing batches (bs=24 avail_mem=121.22 GB):  60%|██████    | 12/20 [00:00<00:00, 15.55it/s]Capturing batches (bs=16 avail_mem=108.39 GB):  60%|██████    | 12/20 [00:00<00:00, 15.55it/s]

    Capturing batches (bs=16 avail_mem=108.39 GB):  75%|███████▌  | 15/20 [00:00<00:00, 16.17it/s]Capturing batches (bs=12 avail_mem=107.41 GB):  75%|███████▌  | 15/20 [00:00<00:00, 16.17it/s]Capturing batches (bs=8 avail_mem=107.40 GB):  75%|███████▌  | 15/20 [00:00<00:00, 16.17it/s] Capturing batches (bs=4 avail_mem=107.40 GB):  75%|███████▌  | 15/20 [00:01<00:00, 16.17it/s]Capturing batches (bs=2 avail_mem=107.39 GB):  75%|███████▌  | 15/20 [00:01<00:00, 16.17it/s]Capturing batches (bs=2 avail_mem=107.39 GB):  95%|█████████▌| 19/20 [00:01<00:00, 20.52it/s]Capturing batches (bs=1 avail_mem=107.39 GB):  95%|█████████▌| 19/20 [00:01<00:00, 20.52it/s]Capturing batches (bs=1 avail_mem=107.39 GB): 100%|██████████| 20/20 [00:01<00:00, 17.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:04,  3.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:04,  3.24s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:29,  1.60s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:29,  1.60s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:52,  1.04it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:52,  1.04it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:35,  1.54it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:35,  1.54it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.12it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.12it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.76it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.76it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.49it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.49it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.27it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.27it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.17it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.17it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.17it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.88it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.88it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.88it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:05,  8.41it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:05,  8.41it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.41it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04, 10.00it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04, 10.00it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04, 10.00it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.93it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.93it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:12<00:03, 11.93it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:12<00:50,  1.29s/it]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:12<00:50,  1.29s/it]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:12<00:50,  1.29s/it]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:12<00:50,  1.29s/it] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:12<00:50,  1.29s/it]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:12<00:23,  1.47it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:12<00:23,  1.47it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:12<00:23,  1.47it/s]

    Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:13<00:23,  1.47it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:13<00:23,  1.47it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:13<00:23,  1.47it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:13<00:11,  2.66it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:13<00:11,  2.66it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:13<00:11,  2.66it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:13<00:11,  2.66it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:13<00:11,  2.66it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:13<00:11,  2.66it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:13<00:11,  2.66it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:13<00:11,  2.66it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:13<00:04,  4.94it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:13<00:04,  4.94it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:13<00:04,  4.94it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:13<00:04,  4.94it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:13<00:04,  4.94it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:13<00:04,  4.94it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:13<00:04,  4.94it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:13<00:04,  4.94it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:13<00:04,  4.94it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:13<00:01,  8.35it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:13<00:01,  8.35it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:13<00:01,  8.35it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:13<00:01,  8.35it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:13<00:01,  8.35it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:13<00:01,  8.35it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:13<00:01,  8.35it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:13<00:01,  8.35it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:13<00:01,  8.35it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:13<00:01,  8.35it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:13<00:00, 13.33it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:13<00:00, 13.33it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:13<00:00, 13.33it/s]

    Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:13<00:00, 13.33it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:13<00:00, 13.33it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:13<00:00, 13.33it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:13<00:00, 13.33it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:13<00:00,  4.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=100.85 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=100.85 GB):   2%|▏         | 1/58 [00:00<00:16,  3.53it/s]Capturing num tokens (num_tokens=7680 avail_mem=100.82 GB):   2%|▏         | 1/58 [00:00<00:16,  3.53it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=100.82 GB):   3%|▎         | 2/58 [00:00<00:15,  3.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=100.82 GB):   3%|▎         | 2/58 [00:00<00:15,  3.65it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=100.82 GB):   5%|▌         | 3/58 [00:00<00:14,  3.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=100.82 GB):   5%|▌         | 3/58 [00:00<00:14,  3.72it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=100.82 GB):   7%|▋         | 4/58 [00:01<00:15,  3.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=100.79 GB):   7%|▋         | 4/58 [00:01<00:15,  3.40it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=100.79 GB):   9%|▊         | 5/58 [00:01<00:15,  3.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=100.79 GB):   9%|▊         | 5/58 [00:01<00:15,  3.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=100.79 GB):  10%|█         | 6/58 [00:01<00:13,  3.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=100.79 GB):  10%|█         | 6/58 [00:01<00:13,  3.97it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=100.79 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=100.79 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.12it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=100.79 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=100.79 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=100.79 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=100.80 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.70it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=100.80 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=100.79 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=100.79 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=100.79 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=100.79 GB):  21%|██        | 12/58 [00:02<00:06,  6.87it/s]Capturing num tokens (num_tokens=3072 avail_mem=100.79 GB):  21%|██        | 12/58 [00:02<00:06,  6.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=100.79 GB):  21%|██        | 12/58 [00:02<00:06,  6.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=100.79 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=100.78 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.25it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=100.78 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=100.78 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=100.78 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.41it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=100.78 GB):  29%|██▉       | 17/58 [00:03<00:04,  8.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=100.78 GB):  29%|██▉       | 17/58 [00:03<00:04,  8.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=100.78 GB):  29%|██▉       | 17/58 [00:03<00:04,  8.37it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=100.78 GB):  33%|███▎      | 19/58 [00:03<00:04,  8.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=100.77 GB):  33%|███▎      | 19/58 [00:03<00:04,  8.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=100.77 GB):  34%|███▍      | 20/58 [00:03<00:04,  8.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=100.77 GB):  34%|███▍      | 20/58 [00:03<00:04,  8.87it/s]Capturing num tokens (num_tokens=960 avail_mem=100.77 GB):  34%|███▍      | 20/58 [00:03<00:04,  8.87it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=100.76 GB):  34%|███▍      | 20/58 [00:03<00:04,  8.87it/s]Capturing num tokens (num_tokens=896 avail_mem=100.76 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.63it/s]Capturing num tokens (num_tokens=832 avail_mem=100.76 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.63it/s]Capturing num tokens (num_tokens=768 avail_mem=100.75 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.63it/s]Capturing num tokens (num_tokens=704 avail_mem=100.74 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.63it/s]Capturing num tokens (num_tokens=704 avail_mem=100.74 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.09it/s]Capturing num tokens (num_tokens=640 avail_mem=100.74 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.09it/s]

    Capturing num tokens (num_tokens=576 avail_mem=100.74 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.09it/s]Capturing num tokens (num_tokens=576 avail_mem=100.74 GB):  48%|████▊     | 28/58 [00:03<00:02, 13.63it/s]Capturing num tokens (num_tokens=512 avail_mem=100.73 GB):  48%|████▊     | 28/58 [00:03<00:02, 13.63it/s]Capturing num tokens (num_tokens=480 avail_mem=100.73 GB):  48%|████▊     | 28/58 [00:03<00:02, 13.63it/s]Capturing num tokens (num_tokens=448 avail_mem=100.72 GB):  48%|████▊     | 28/58 [00:03<00:02, 13.63it/s]Capturing num tokens (num_tokens=416 avail_mem=100.71 GB):  48%|████▊     | 28/58 [00:03<00:02, 13.63it/s]Capturing num tokens (num_tokens=416 avail_mem=100.71 GB):  55%|█████▌    | 32/58 [00:03<00:01, 18.20it/s]Capturing num tokens (num_tokens=384 avail_mem=100.71 GB):  55%|█████▌    | 32/58 [00:03<00:01, 18.20it/s]Capturing num tokens (num_tokens=352 avail_mem=100.70 GB):  55%|█████▌    | 32/58 [00:03<00:01, 18.20it/s]

    Capturing num tokens (num_tokens=320 avail_mem=100.70 GB):  55%|█████▌    | 32/58 [00:04<00:01, 18.20it/s]Capturing num tokens (num_tokens=288 avail_mem=100.69 GB):  55%|█████▌    | 32/58 [00:04<00:01, 18.20it/s]Capturing num tokens (num_tokens=288 avail_mem=100.69 GB):  62%|██████▏   | 36/58 [00:04<00:00, 22.05it/s]Capturing num tokens (num_tokens=256 avail_mem=100.68 GB):  62%|██████▏   | 36/58 [00:04<00:00, 22.05it/s]Capturing num tokens (num_tokens=240 avail_mem=100.68 GB):  62%|██████▏   | 36/58 [00:04<00:00, 22.05it/s]Capturing num tokens (num_tokens=224 avail_mem=100.67 GB):  62%|██████▏   | 36/58 [00:04<00:00, 22.05it/s]Capturing num tokens (num_tokens=208 avail_mem=100.67 GB):  62%|██████▏   | 36/58 [00:04<00:00, 22.05it/s]Capturing num tokens (num_tokens=208 avail_mem=100.67 GB):  69%|██████▉   | 40/58 [00:04<00:00, 25.31it/s]Capturing num tokens (num_tokens=192 avail_mem=100.66 GB):  69%|██████▉   | 40/58 [00:04<00:00, 25.31it/s]

    Capturing num tokens (num_tokens=176 avail_mem=100.66 GB):  69%|██████▉   | 40/58 [00:04<00:00, 25.31it/s]Capturing num tokens (num_tokens=160 avail_mem=100.65 GB):  69%|██████▉   | 40/58 [00:04<00:00, 25.31it/s]Capturing num tokens (num_tokens=144 avail_mem=100.65 GB):  69%|██████▉   | 40/58 [00:04<00:00, 25.31it/s]Capturing num tokens (num_tokens=144 avail_mem=100.65 GB):  76%|███████▌  | 44/58 [00:04<00:00, 27.78it/s]Capturing num tokens (num_tokens=128 avail_mem=100.65 GB):  76%|███████▌  | 44/58 [00:04<00:00, 27.78it/s]Capturing num tokens (num_tokens=112 avail_mem=100.65 GB):  76%|███████▌  | 44/58 [00:04<00:00, 27.78it/s]Capturing num tokens (num_tokens=96 avail_mem=100.64 GB):  76%|███████▌  | 44/58 [00:04<00:00, 27.78it/s] Capturing num tokens (num_tokens=80 avail_mem=100.64 GB):  76%|███████▌  | 44/58 [00:04<00:00, 27.78it/s]

    Capturing num tokens (num_tokens=80 avail_mem=100.64 GB):  83%|████████▎ | 48/58 [00:04<00:00, 29.43it/s]Capturing num tokens (num_tokens=64 avail_mem=100.63 GB):  83%|████████▎ | 48/58 [00:04<00:00, 29.43it/s]Capturing num tokens (num_tokens=48 avail_mem=100.63 GB):  83%|████████▎ | 48/58 [00:04<00:00, 29.43it/s]Capturing num tokens (num_tokens=32 avail_mem=100.62 GB):  83%|████████▎ | 48/58 [00:04<00:00, 29.43it/s]Capturing num tokens (num_tokens=28 avail_mem=100.62 GB):  83%|████████▎ | 48/58 [00:04<00:00, 29.43it/s]Capturing num tokens (num_tokens=28 avail_mem=100.62 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.80it/s]Capturing num tokens (num_tokens=24 avail_mem=100.61 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.80it/s]Capturing num tokens (num_tokens=20 avail_mem=100.60 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.80it/s]Capturing num tokens (num_tokens=16 avail_mem=100.60 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.80it/s]

    Capturing num tokens (num_tokens=12 avail_mem=100.59 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.80it/s]Capturing num tokens (num_tokens=12 avail_mem=100.59 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.72it/s]Capturing num tokens (num_tokens=8 avail_mem=100.58 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.72it/s] Capturing num tokens (num_tokens=4 avail_mem=100.58 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.72it/s]Capturing num tokens (num_tokens=4 avail_mem=100.58 GB): 100%|██████████| 58/58 [00:04<00:00, 12.25it/s]


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
    
    Generated text: Alright, so the user asked me to provide the information and population of the capital of France in JSON format. Hmm, let me break this down.
    
    First, I need to identify what the user is asking for. They want the capital city of France, which I know is Paris. Then, they also need the population. I should make sure the population figure is up to date because demographics can change over time.
    
    I wonder if the user is a student working on a project or maybe someone planning a trip. Knowing the population could be useful for research or travel planning. Maybe they need this data for a presentation or a report. Either way, I should provide accurate information.
    
    Now, the user specifically asked for the JSON format. JSON is a data interchange format, so I need to structure the information properly. I'll start with the capital city, then include the population, and maybe add some key points about the city to make it more informative.
    
    I should check the current population of Paris. Let me recall the latest figures. I think it's around 2 million, but I'm not 100% sure. Wait, no, I remember that Paris's population is over 2 million. In fact, the population was approximately 2,165,000 as of 2021, but it can fluctuate. I should mention that the population is approximate and can vary based on the year.
    
    I also need to structure the JSON correctly. Typically, it's an object with key-value pairs. So, I'll have a key like "capital" with the value "Paris", and another key like "population" with the number. Maybe add a "description" key to give a bit more context about Paris.
    
    Wait, the user didn't specify the format beyond JSON, so perhaps just the basic info is enough. I should make sure the response is clear and concise. No need for extra fluff.
    
    I should also consider if the user might need more details. Maybe they want statistics about the city, like area or number of residents per square kilometer. But since they only asked for population, I'll stick to that.
    
    Lastly, I'll present the JSON in a code block so it's easy to read. Make sure the syntax is correct, with proper commas and braces. No trailing commas, as that could cause errors in parsing.
    
    Okay, putting it all together, I'll structure the JSON with the necessary keys and provide the population figure accurately. I'll mention that the population is approximate to cover any potential inaccuracies.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "capital": "Paris",
      "population": 2165000,
      "description": "Paris is the capital city of France and one of the most populated cities in the world."
    }
    ```



```python
llm.shutdown()
```
