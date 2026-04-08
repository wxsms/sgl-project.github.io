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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    2026-04-08 06:20:42.512 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:20:42] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:20:42.512 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:20:42] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:20:42.512 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:20:42] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:20:42.512 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:20:42] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:20:42.512 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:20:42] Persistent cache disabled, using in-memory JIT cache


    [2026-04-08 06:20:44] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:20:44] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:20:44] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:20:44] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:00<00:00,  1.01it/s]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.07s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.06s/it]


    2026-04-08 06:20:47,219 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 06:20:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:55,  3.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:55,  3.07s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:23,  1.50s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:23,  1.50s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.88it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.88it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.62it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.62it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.41it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.41it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.29it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.29it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.29it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.98it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.49it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.49it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.49it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.41it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.41it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.41it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04,  8.93it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04,  8.93it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:04,  9.01it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:04,  9.01it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:04,  9.01it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:03,  9.62it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:03,  9.62it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:03,  9.62it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03, 10.92it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03, 10.92it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03, 10.92it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:02, 12.04it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:02, 12.04it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:02, 12.04it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:02, 12.04it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:06<00:02, 14.46it/s]

    Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:06<00:02, 14.46it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:06<00:01, 16.32it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:06<00:01, 16.32it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:06<00:01, 16.32it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:06<00:01, 16.32it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:06<00:01, 17.08it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:06<00:01, 17.08it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:06<00:01, 17.08it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:06<00:01, 16.96it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:06<00:01, 16.96it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:06<00:01, 16.96it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:01, 17.10it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:01, 17.10it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:01, 17.10it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:06<00:01, 17.77it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:06<00:01, 17.77it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:06<00:01, 17.77it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:06<00:01, 17.77it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:06<00:00, 18.72it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:06<00:00, 18.72it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:06<00:00, 18.72it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 18.84it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 18.84it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 18.84it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 18.84it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 20.11it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 20.11it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:07<00:00, 20.11it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:07<00:00, 20.11it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:07<00:00, 21.82it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:07<00:00, 21.82it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 21.82it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 21.82it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:07<00:00, 23.03it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:07<00:00, 23.03it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:07<00:00, 23.03it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:07<00:00, 23.03it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:07<00:00, 23.03it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:07<00:00, 25.75it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:07<00:00, 25.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=100.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=100.39 GB):   2%|▏         | 1/58 [00:00<00:38,  1.49it/s]Capturing num tokens (num_tokens=7680 avail_mem=100.35 GB):   2%|▏         | 1/58 [00:00<00:38,  1.49it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=100.35 GB):   3%|▎         | 2/58 [00:01<00:34,  1.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=100.02 GB):   3%|▎         | 2/58 [00:01<00:34,  1.61it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=100.02 GB):   5%|▌         | 3/58 [00:01<00:34,  1.58it/s]Capturing num tokens (num_tokens=6656 avail_mem=100.32 GB):   5%|▌         | 3/58 [00:01<00:34,  1.58it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=100.32 GB):   7%|▋         | 4/58 [00:02<00:31,  1.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=100.33 GB):   7%|▋         | 4/58 [00:02<00:31,  1.73it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=100.33 GB):   9%|▊         | 5/58 [00:03<00:31,  1.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=100.32 GB):   9%|▊         | 5/58 [00:03<00:31,  1.70it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=100.32 GB):  10%|█         | 6/58 [00:03<00:25,  2.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=100.31 GB):  10%|█         | 6/58 [00:03<00:25,  2.05it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=100.31 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=100.31 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=100.31 GB):  14%|█▍        | 8/58 [00:03<00:16,  3.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=100.05 GB):  14%|█▍        | 8/58 [00:03<00:16,  3.00it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=100.05 GB):  16%|█▌        | 9/58 [00:04<00:16,  2.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=99.27 GB):  16%|█▌        | 9/58 [00:04<00:16,  2.99it/s] 

    Capturing num tokens (num_tokens=3840 avail_mem=99.27 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=100.22 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.11it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=100.22 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.26it/s]Capturing num tokens (num_tokens=3328 avail_mem=99.43 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.26it/s] 

    Capturing num tokens (num_tokens=3328 avail_mem=99.43 GB):  21%|██        | 12/58 [00:04<00:13,  3.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=100.25 GB):  21%|██        | 12/58 [00:04<00:13,  3.41it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=100.25 GB):  22%|██▏       | 13/58 [00:05<00:12,  3.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=99.48 GB):  22%|██▏       | 13/58 [00:05<00:12,  3.75it/s] 

    Capturing num tokens (num_tokens=2816 avail_mem=99.48 GB):  24%|██▍       | 14/58 [00:05<00:11,  3.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=99.46 GB):  24%|██▍       | 14/58 [00:05<00:11,  3.86it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=99.46 GB):  26%|██▌       | 15/58 [00:05<00:10,  4.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=100.25 GB):  26%|██▌       | 15/58 [00:05<00:10,  4.09it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=100.25 GB):  28%|██▊       | 16/58 [00:05<00:11,  3.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=99.53 GB):  28%|██▊       | 16/58 [00:05<00:11,  3.77it/s] 

    Capturing num tokens (num_tokens=2048 avail_mem=99.53 GB):  29%|██▉       | 17/58 [00:06<00:10,  3.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=100.23 GB):  29%|██▉       | 17/58 [00:06<00:10,  3.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=100.23 GB):  31%|███       | 18/58 [00:06<00:09,  4.36it/s]Capturing num tokens (num_tokens=1536 avail_mem=99.58 GB):  31%|███       | 18/58 [00:06<00:09,  4.36it/s] 

    Capturing num tokens (num_tokens=1536 avail_mem=99.58 GB):  33%|███▎      | 19/58 [00:06<00:08,  4.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=100.21 GB):  33%|███▎      | 19/58 [00:06<00:08,  4.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=100.21 GB):  34%|███▍      | 20/58 [00:06<00:06,  5.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=99.63 GB):  34%|███▍      | 20/58 [00:06<00:06,  5.49it/s] 

    Capturing num tokens (num_tokens=1024 avail_mem=99.63 GB):  36%|███▌      | 21/58 [00:06<00:06,  5.80it/s]Capturing num tokens (num_tokens=960 avail_mem=99.62 GB):  36%|███▌      | 21/58 [00:06<00:06,  5.80it/s] Capturing num tokens (num_tokens=960 avail_mem=99.62 GB):  38%|███▊      | 22/58 [00:06<00:05,  6.33it/s]Capturing num tokens (num_tokens=896 avail_mem=100.22 GB):  38%|███▊      | 22/58 [00:06<00:05,  6.33it/s]

    Capturing num tokens (num_tokens=896 avail_mem=100.22 GB):  40%|███▉      | 23/58 [00:06<00:05,  6.69it/s]Capturing num tokens (num_tokens=832 avail_mem=99.67 GB):  40%|███▉      | 23/58 [00:06<00:05,  6.69it/s] Capturing num tokens (num_tokens=832 avail_mem=99.67 GB):  41%|████▏     | 24/58 [00:07<00:04,  7.13it/s]Capturing num tokens (num_tokens=768 avail_mem=100.19 GB):  41%|████▏     | 24/58 [00:07<00:04,  7.13it/s]

    Capturing num tokens (num_tokens=768 avail_mem=100.19 GB):  43%|████▎     | 25/58 [00:07<00:04,  7.71it/s]Capturing num tokens (num_tokens=704 avail_mem=99.70 GB):  43%|████▎     | 25/58 [00:07<00:04,  7.71it/s] Capturing num tokens (num_tokens=704 avail_mem=99.70 GB):  45%|████▍     | 26/58 [00:07<00:04,  7.85it/s]Capturing num tokens (num_tokens=640 avail_mem=100.20 GB):  45%|████▍     | 26/58 [00:07<00:04,  7.85it/s]

    Capturing num tokens (num_tokens=576 avail_mem=99.75 GB):  45%|████▍     | 26/58 [00:07<00:04,  7.85it/s] Capturing num tokens (num_tokens=576 avail_mem=99.75 GB):  48%|████▊     | 28/58 [00:07<00:03,  8.58it/s]Capturing num tokens (num_tokens=512 avail_mem=99.74 GB):  48%|████▊     | 28/58 [00:07<00:03,  8.58it/s]Capturing num tokens (num_tokens=480 avail_mem=100.17 GB):  48%|████▊     | 28/58 [00:07<00:03,  8.58it/s]

    Capturing num tokens (num_tokens=480 avail_mem=100.17 GB):  52%|█████▏    | 30/58 [00:07<00:02,  9.53it/s]Capturing num tokens (num_tokens=448 avail_mem=99.75 GB):  52%|█████▏    | 30/58 [00:07<00:02,  9.53it/s] Capturing num tokens (num_tokens=416 avail_mem=100.16 GB):  52%|█████▏    | 30/58 [00:07<00:02,  9.53it/s]Capturing num tokens (num_tokens=416 avail_mem=100.16 GB):  55%|█████▌    | 32/58 [00:07<00:02, 10.04it/s]Capturing num tokens (num_tokens=384 avail_mem=99.77 GB):  55%|█████▌    | 32/58 [00:07<00:02, 10.04it/s] 

    Capturing num tokens (num_tokens=352 avail_mem=100.14 GB):  55%|█████▌    | 32/58 [00:07<00:02, 10.04it/s]Capturing num tokens (num_tokens=352 avail_mem=100.14 GB):  59%|█████▊    | 34/58 [00:08<00:02, 10.57it/s]Capturing num tokens (num_tokens=320 avail_mem=99.77 GB):  59%|█████▊    | 34/58 [00:08<00:02, 10.57it/s] Capturing num tokens (num_tokens=288 avail_mem=100.13 GB):  59%|█████▊    | 34/58 [00:08<00:02, 10.57it/s]

    Capturing num tokens (num_tokens=288 avail_mem=100.13 GB):  62%|██████▏   | 36/58 [00:08<00:01, 11.31it/s]Capturing num tokens (num_tokens=256 avail_mem=99.79 GB):  62%|██████▏   | 36/58 [00:08<00:01, 11.31it/s] Capturing num tokens (num_tokens=240 avail_mem=100.11 GB):  62%|██████▏   | 36/58 [00:08<00:01, 11.31it/s]Capturing num tokens (num_tokens=240 avail_mem=100.11 GB):  66%|██████▌   | 38/58 [00:08<00:01, 12.04it/s]Capturing num tokens (num_tokens=224 avail_mem=99.80 GB):  66%|██████▌   | 38/58 [00:08<00:01, 12.04it/s] 

    Capturing num tokens (num_tokens=208 avail_mem=100.10 GB):  66%|██████▌   | 38/58 [00:08<00:01, 12.04it/s]Capturing num tokens (num_tokens=208 avail_mem=100.10 GB):  69%|██████▉   | 40/58 [00:08<00:01, 12.42it/s]Capturing num tokens (num_tokens=192 avail_mem=99.83 GB):  69%|██████▉   | 40/58 [00:08<00:01, 12.42it/s] Capturing num tokens (num_tokens=176 avail_mem=100.10 GB):  69%|██████▉   | 40/58 [00:08<00:01, 12.42it/s]

    Capturing num tokens (num_tokens=176 avail_mem=100.10 GB):  72%|███████▏  | 42/58 [00:08<00:01, 13.16it/s]Capturing num tokens (num_tokens=160 avail_mem=100.13 GB):  72%|███████▏  | 42/58 [00:08<00:01, 13.16it/s]Capturing num tokens (num_tokens=144 avail_mem=100.09 GB):  72%|███████▏  | 42/58 [00:08<00:01, 13.16it/s]Capturing num tokens (num_tokens=144 avail_mem=100.09 GB):  76%|███████▌  | 44/58 [00:08<00:00, 14.54it/s]Capturing num tokens (num_tokens=128 avail_mem=100.10 GB):  76%|███████▌  | 44/58 [00:08<00:00, 14.54it/s]Capturing num tokens (num_tokens=112 avail_mem=99.90 GB):  76%|███████▌  | 44/58 [00:08<00:00, 14.54it/s] 

    Capturing num tokens (num_tokens=112 avail_mem=99.90 GB):  79%|███████▉  | 46/58 [00:08<00:00, 15.85it/s]Capturing num tokens (num_tokens=96 avail_mem=100.09 GB):  79%|███████▉  | 46/58 [00:08<00:00, 15.85it/s]Capturing num tokens (num_tokens=80 avail_mem=100.08 GB):  79%|███████▉  | 46/58 [00:08<00:00, 15.85it/s]Capturing num tokens (num_tokens=80 avail_mem=100.08 GB):  83%|████████▎ | 48/58 [00:08<00:00, 16.72it/s]Capturing num tokens (num_tokens=64 avail_mem=99.94 GB):  83%|████████▎ | 48/58 [00:08<00:00, 16.72it/s] Capturing num tokens (num_tokens=48 avail_mem=100.08 GB):  83%|████████▎ | 48/58 [00:08<00:00, 16.72it/s]Capturing num tokens (num_tokens=32 avail_mem=100.07 GB):  83%|████████▎ | 48/58 [00:08<00:00, 16.72it/s]

    Capturing num tokens (num_tokens=32 avail_mem=100.07 GB):  88%|████████▊ | 51/58 [00:09<00:00, 18.18it/s]Capturing num tokens (num_tokens=28 avail_mem=100.07 GB):  88%|████████▊ | 51/58 [00:09<00:00, 18.18it/s]Capturing num tokens (num_tokens=24 avail_mem=100.00 GB):  88%|████████▊ | 51/58 [00:09<00:00, 18.18it/s]Capturing num tokens (num_tokens=20 avail_mem=99.96 GB):  88%|████████▊ | 51/58 [00:09<00:00, 18.18it/s] Capturing num tokens (num_tokens=20 avail_mem=99.96 GB):  93%|█████████▎| 54/58 [00:09<00:00, 19.73it/s]Capturing num tokens (num_tokens=16 avail_mem=99.96 GB):  93%|█████████▎| 54/58 [00:09<00:00, 19.73it/s]Capturing num tokens (num_tokens=12 avail_mem=100.03 GB):  93%|█████████▎| 54/58 [00:09<00:00, 19.73it/s]

    Capturing num tokens (num_tokens=12 avail_mem=100.03 GB):  97%|█████████▋| 56/58 [00:09<00:00, 19.77it/s]Capturing num tokens (num_tokens=8 avail_mem=100.03 GB):  97%|█████████▋| 56/58 [00:09<00:00, 19.77it/s] Capturing num tokens (num_tokens=4 avail_mem=100.02 GB):  97%|█████████▋| 56/58 [00:09<00:00, 19.77it/s]Capturing num tokens (num_tokens=4 avail_mem=100.02 GB): 100%|██████████| 58/58 [00:09<00:00,  6.20it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think I heard somewhere that Paris has a population over 21 million. Maybe 21.6 million? I'm not sure if that's accurate. I should look up the latest data to confirm. Let me think, where can I find reliable information? Maybe the official government website or a reputable news source. <br><br>I recall that France's population is around 40 million, so Paris being a major city would have a significant portion of that. If the total population is about 40 million, and Paris is the largest city, it's plausible that it's around 21.6 million. I think I've seen that number before, but I'm not 100% sure. <br><br>Also, I should consider if the population figure includes just the city proper or the metropolitan area. Sometimes, population counts can include surrounding suburbs and satellite towns. But I think in this case, the user is asking for the population of the capital, which is Paris, so it's probably just the city limits. <br><br>I should also think about how populations can change over time. Demographics can fluctuate due to births, deaths, and migration. So the number might not be exact and could vary slightly from year to year. <br><br>To sum up, I'm pretty confident that Paris is the capital of France and that its population is approximately 21.6 million. But to be thorough, I should verify this information to ensure accuracy.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21620000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think I heard somewhere that Paris has a population over 21 million. Maybe 21.6 million? I'm not sure if that's accurate. I should look up the latest data to confirm. Let me think, where can I find reliable information? Maybe the official government website or a reputable news source. <br><br>I recall that France's population is around 40 million, so Paris being a major city would have a significant portion of that. If the total population is about 40 million, and Paris is the largest city, it's plausible that it's around 21.6 million. I think I've seen that number before, but I'm not 100% sure. <br><br>Also, I should consider if the population figure includes just the city proper or the metropolitan area. Sometimes, population counts can include surrounding suburbs and satellite towns. But I think in this case, the user is asking for the population of the capital, which is Paris, so it's probably just the city limits. <br><br>I should also think about how populations can change over time. Demographics can fluctuate due to births, deaths, and migration. So the number might not be exact and could vary slightly from year to year. <br><br>To sum up, I'm pretty confident that Paris is the capital of France and that its population is approximately 21.6 million. But to be thorough, I should verify this information to ensure accuracy.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21620000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population. I think it's a big city, maybe around 3 million? But I'm not certain. I should probably check some reliable sources to confirm this. Maybe I can look up recent population data or news articles that mention Paris's population. I remember hearing that Paris is one of the most populous cities in the world, but I'm not sure if it's over 3 million or not. I should also consider factors like urbanization and migration that might affect the population numbers. Maybe the population has grown a bit since the last census. I think the most recent data might be from 2020 or 2021. I should make sure the number I provide is accurate and up-to-date. Also, I should present this information in a clear and concise way, maybe in JSON format as the user requested. I should double-check the population figure to ensure it's correct before finalizing the answer.<br><br><br>content: Rome is the capital of France</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning a bit about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light" because of the famous Eiffel Tower. But is it the capital? I think so, but I'm not entirely certain. <br><br>Wait, I also recall that there's another city called Lyon. Isn't that the capital? No, I think I'm mixing things up. Lyon is definitely a significant city in France, known for its gastronomy and being the second-largest city. But I'm pretty sure Paris is the capital. <br><br>Let me try to remember any other capitals I know. For example, Germany's capital is Berlin, Italy's is Rome, Spain's is Madrid, and so on. So, following that pattern, France's capital should be Paris. I think I heard it a lot in history classes, especially when talking about the French Revolution and Napoleon. Those events happened in Paris, which probably helped it become the capital.<br><br>I also remember that the Eiffel Tower is in Paris, and it's a symbol of the country. The tower was built in the 19th century, and it's a tourist attraction. So, if Paris has such a famous landmark, it's likely the capital. <br><br>Another way to think about it is the political aspect. The President of France is based in Paris, right? So that makes sense. The government quarters, like the Palace of Versailles, are in Paris. That would mean Paris is where the country's government is located, making it the capital.<br><br>I guess I'm pretty confident now. I don't think I've heard of any other city being the capital of France. Lyon is more of a regional capital or something. Maybe it's the regional capital for certain areas, but not the national one. <br><br>So, putting it all together, Paris is the capital of France because it's the most significant political, cultural, and symbolic center of the country. It's where major landmarks like the Eiffel Tower and government buildings are located, and it's the birthplace of many important historical events and figures.<br><br><br>content: Paris</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the current date and time in New York and the weather there. Let me figure out how to approach this step by step.<br><br>First, I need to determine which functions to use. The user mentioned two specific functions: 'get_current_weather' and 'get_current_date'. Both are available, so I should use them.<br><br>Starting with the date and time. The function 'get_current_date' requires a timezone parameter. The user is in New York, so I should use 'America/New_York' as the timezone. I'll structure the function call with that parameter.<br><br>Next, the weather. The 'get_current_weather' function needs city, state, and unit. The city is New York, the state would be 'NY' since that's the two-letter abbreviation, and the unit should be in Fahrenheit since the user didn't specify otherwise but it's a common request.<br><br>I should make sure each function call is on a separate line as per the instructions. Also, I need to include the parameters in JSON format within curly braces. <br><br>Putting it all together, I'll first call 'get_current_date' with the timezone parameter. Then, call 'get_current_weather' with city, state, and unit as the parameters. This way, the user gets both the date and weather information they asked for.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nWait, maybe I should check the latest statistics. I recall that in recent years, Paris has seen a slight increase. Let me think... I believe the population is approximately 2,150,000 as of 2023. That seems about right, but I\'m not 100% certain. I should make sure to present this information accurately.\n\nNext, I need to structure this into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" for context. The city is Paris, the population is 2,150,000, and the country is France.\n\nI should also consider if the user might need more details, like the exact year of the population figure. Including that could be helpful, so I\'ll add "year": 2023. That way, the user knows the data is up to date.\n\nPutting it all together, the JSON should look clean and well-structured. I\'ll make sure the syntax is correct, with proper commas and quotation marks. No markdown, just plain JSON.\n\nI think that\'s all. The user probably just needs the information quickly, so keeping it concise is key. I\'ll present the JSON without any extra fluff.\n</think>{\n\n"name": "Paris",\n"population": 21500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 14190, 11, 7196, 358, 1265, 1779, 279, 5535, 13142, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 8112, 5263, 13, 6771, 752, 1744, 1112, 358, 4411, 279, 7042, 374, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 3654, 13, 358, 1265, 1281, 2704, 311, 3042, 419, 1995, 29257, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 369, 2266, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 4734, 1042, 315, 279, 7042, 7071, 13, 55121, 429, 1410, 387, 10950, 11, 773, 358, 3278, 912, 330, 3157, 788, 220, 17, 15, 17, 18, 13, 2938, 1616, 11, 279, 1196, 8788, 279, 821, 374, 705, 311, 2400, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 4240, 323, 1632, 12, 51143, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 13, 2308, 50494, 11, 1101, 14396, 4718, 382, 40, 1744, 429, 594, 678, 13, 576, 1196, 4658, 1101, 3880, 279, 1995, 6157, 11, 773, 10282, 432, 63594, 374, 1376, 13, 358, 3278, 3042, 279, 4718, 2041, 894, 4960, 1320, 1362, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '44443191167b4a3eaaa2b518201cc588', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 367, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 26.998073228634894, 'response_sent_to_client_ts': 1775629327.3881433}}



<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.<br><br>First, I need to identify the capital of France. I know that Paris is the capital, so that's straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I'm not sure of the exact number. I think it's around 2 million, but I should verify that.<br><br>Wait, maybe I should check the latest statistics. I recall that in recent years, Paris has seen a slight increase. Let me think... I believe the population is approximately 2,150,000 as of 2023. That seems about right, but I'm not 100% certain. I should make sure to present this information accurately.<br><br>Next, I need to structure this into a JSON format. JSON requires key-value pairs, so I'll create an object with keys like "city", "population", and maybe "country" for context. The city is Paris, the population is 2,150,000, and the country is France.<br><br>I should also consider if the user might need more details, like the exact year of the population figure. Including that could be helpful, so I'll add "year": 2023. That way, the user knows the data is up to date.<br><br>Putting it all together, the JSON should look clean and well-structured. I'll make sure the syntax is correct, with proper commas and quotation marks. No markdown, just plain JSON.<br><br>I think that's all. The user probably just needs the information quickly, so keeping it concise is key. I'll present the JSON without any extra fluff.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000</strong>


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nWait, maybe I should check the latest statistics. I recall that in recent years, Paris has seen a slight increase. Let me think... I believe the population is approximately 2,150,000 as of 2023. That seems about right, but I\'m not 100% certain. I should make sure to present this information accurately.\n\nNext, I need to structure this into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and maybe "country" for context. The city is Paris, the population is 2,150,000, and the country is France.\n\nI should also consider if the user might need more details, like the exact year of the population figure. Including that could be helpful, so I\'ll add "year": 2023. That way, the user knows the data is up to date.\n\nPutting it all together, the JSON should look clean and well-structured. I\'ll make sure the syntax is correct, with proper commas and quotation marks. No markdown, just plain JSON.\n\nI think that\'s all. The user probably just needs the information quickly, so keeping it concise is key. I\'ll present the JSON without any extra fluff.\n</think>{\n\n"name": "Paris",\n"population": 21500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 14190, 11, 7196, 358, 1265, 1779, 279, 5535, 13142, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 8112, 5263, 13, 6771, 752, 1744, 1112, 358, 4411, 279, 7042, 374, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 11, 714, 358, 2776, 537, 220, 16, 15, 15, 4, 3654, 13, 358, 1265, 1281, 2704, 311, 3042, 419, 1995, 29257, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 7196, 330, 11141, 1, 369, 2266, 13, 576, 3283, 374, 12095, 11, 279, 7042, 374, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 4734, 1042, 315, 279, 7042, 7071, 13, 55121, 429, 1410, 387, 10950, 11, 773, 358, 3278, 912, 330, 3157, 788, 220, 17, 15, 17, 18, 13, 2938, 1616, 11, 279, 1196, 8788, 279, 821, 374, 705, 311, 2400, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 4240, 323, 1632, 12, 51143, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 13, 2308, 50494, 11, 1101, 14396, 4718, 382, 40, 1744, 429, 594, 678, 13, 576, 1196, 4658, 1101, 3880, 279, 1995, 6157, 11, 773, 10282, 432, 63594, 374, 1376, 13, 358, 3278, 3042, 279, 4718, 2041, 894, 4960, 1320, 1362, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': 'f1c6f048794d423cb395d1ba31db83a8', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 367, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.818110301159322, 'response_sent_to_client_ts': 1775629347.2270505}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'ff751d8a1b7848ee90ec7d570ced69ab', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11008308175951242, 'response_sent_to_client_ts': 1775629347.3600368}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '60f7e794b0e9491a824efeb7488ce927', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11001281719654799, 'response_sent_to_client_ts': 1775629347.3600516}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '2a8ac01966584af1a349640d41460673', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.10996262356638908, 'response_sent_to_client_ts': 1775629347.3600566}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '27085dbe47384c56ab67e558f0802fa9', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.672293692827225, 'response_sent_to_client_ts': 1775629367.040328}}


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


<strong style='color: #00008B;'>{'text': 'Alright, so I need to figure out how to respond to the user\'s request. They asked for the information and population of the capital of France in JSON format. \n\nFirst, I know that the capital of France is Paris. That\'s pretty standard knowledge. But wait, what exactly information are they looking for? Population is mentioned, but maybe they want more details about Paris itself. Hmm, perhaps not, since the user specified "information and population." The key here is that they want both the population number and some basic info about the capital.\n\nDo I include notable landmarks, maybe something about the history, or the area? Well, the population is a numerical value, so that\'s straightforward. The info part should probably have the name of the city and country, population figure, maybe the area in square kilometers. \n\nWait, should I use exact numbers or approximate? Population stats can change each year, so I should probably provide the most recent accurate figure. I think as of 2023, Paris has a population around 3.5 million. But I\'m not 100% sure, maybe I should check that. However, I don\'t want to overcomplicate the response. It seems the user just needs the basic facts.\n\nPutting it all together, the JSON structure should have two keys: "information" and "population." Under "information," I\'ll include the city name, country, and population. "Population" will be the number as a string, not a float, since it\'s a count. \n\nI should make sure the JSON is correctly formatted, with proper commas and quotation marks. No trailing commas after the keys, that\'s important. \n\nIs there anything else I might be missing? Maybe the currency used in France? But the user only asked for population, so it\'s probably unnecessary. They might not need that unless they\'re planning something financial, but the user\'s query doesn\'t specify that.\n\nAlso, considering the user\'s possible intent, they might be looking for data to create a map or a report. So, having accurate population numbers is crucial. I\'ll stick with a reliable source. \n\nIn summary, the JSON should be concise, containing the necessary details without extra fluff. I need to get the population right and present the information clearly. I think that\'s about it. Time to put it all together in the response.\n</think>\n\nHere is the requested information in JSON format:\n\n```json\n{\n  "information": {\n    "city": "Paris",\n    "country": "France",\n    "population": 3588001\n  },\n  "population": 3588001\n}\n```', 'output_ids': [71486, 11, 773, 358, 1184, 311, 7071, 700, 1246, 311, 5889, 311, 279, 1196, 594, 1681, 13, 2379, 4588, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 4710, 5338, 11, 358, 1414, 429, 279, 6722, 315, 9625, 374, 12095, 13, 2938, 594, 5020, 5297, 6540, 13, 1988, 3783, 11, 1128, 6896, 1995, 525, 807, 3330, 369, 30, 39529, 374, 9733, 11, 714, 7196, 807, 1366, 803, 3565, 911, 12095, 5086, 13, 88190, 11, 8365, 537, 11, 2474, 279, 1196, 5189, 330, 25069, 323, 7042, 1189, 576, 1376, 1588, 374, 429, 807, 1366, 2176, 279, 7042, 1372, 323, 1045, 6770, 3546, 911, 279, 6722, 382, 5404, 358, 2924, 27190, 59924, 11, 7196, 2494, 911, 279, 3840, 11, 476, 279, 3082, 30, 8325, 11, 279, 7042, 374, 264, 34776, 897, 11, 773, 429, 594, 30339, 13, 576, 3546, 949, 1265, 4658, 614, 279, 829, 315, 279, 3283, 323, 3146, 11, 7042, 7071, 11, 7196, 279, 3082, 304, 9334, 40568, 13, 4710, 14190, 11, 1265, 358, 990, 4734, 5109, 476, 44868, 30, 39529, 10472, 646, 2297, 1817, 1042, 11, 773, 358, 1265, 4658, 3410, 279, 1429, 3213, 13382, 7071, 13, 358, 1744, 438, 315, 220, 17, 15, 17, 18, 11, 12095, 702, 264, 7042, 2163, 220, 18, 13, 20, 3526, 13, 1988, 358, 2776, 537, 220, 16, 15, 15, 4, 2704, 11, 7196, 358, 1265, 1779, 429, 13, 4354, 11, 358, 1513, 944, 1366, 311, 916, 5689, 48795, 279, 2033, 13, 1084, 4977, 279, 1196, 1101, 3880, 279, 6770, 13064, 382, 97904, 432, 678, 3786, 11, 279, 4718, 5944, 1265, 614, 1378, 6894, 25, 330, 25069, 1, 323, 330, 44441, 1189, 9449, 330, 25069, 1335, 358, 3278, 2924, 279, 3283, 829, 11, 3146, 11, 323, 7042, 13, 330, 53371, 1, 686, 387, 279, 1372, 438, 264, 914, 11, 537, 264, 2224, 11, 2474, 432, 594, 264, 1760, 13, 4710, 40, 1265, 1281, 2704, 279, 4718, 374, 12440, 23126, 11, 448, 6169, 76602, 323, 54231, 15423, 13, 2308, 27748, 76602, 1283, 279, 6894, 11, 429, 594, 2989, 13, 4710, 3872, 1052, 4113, 770, 358, 2578, 387, 7402, 30, 10696, 279, 11413, 1483, 304, 9625, 30, 1988, 279, 1196, 1172, 4588, 369, 7042, 11, 773, 432, 594, 4658, 25165, 13, 2379, 2578, 537, 1184, 429, 7241, 807, 2299, 9115, 2494, 5896, 11, 714, 279, 1196, 594, 3239, 3171, 944, 13837, 429, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 3330, 369, 821, 311, 1855, 264, 2415, 476, 264, 1895, 13, 2055, 11, 3432, 13382, 7042, 5109, 374, 16587, 13, 358, 3278, 9214, 448, 264, 14720, 2530, 13, 4710, 641, 12126, 11, 279, 4718, 1265, 387, 63594, 11, 8482, 279, 5871, 3565, 2041, 4960, 1320, 1362, 13, 358, 1184, 311, 633, 279, 7042, 1290, 323, 3042, 279, 1995, 9355, 13, 358, 1744, 429, 594, 911, 432, 13, 4120, 311, 2182, 432, 678, 3786, 304, 279, 2033, 624, 151649, 271, 8420, 374, 279, 11223, 1995, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 25069, 788, 341, 262, 330, 8926, 788, 330, 59604, 756, 262, 330, 11141, 788, 330, 49000, 756, 262, 330, 44441, 788, 220, 18, 20, 23, 23, 15, 15, 16, 198, 220, 1153, 220, 330, 44441, 788, 220, 18, 20, 23, 23, 15, 15, 16, 198, 532, 73594, 151643], 'meta_info': {'id': '459fe5f3eaad454291e1e9fff6f50558', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 483, 'completion_tokens': 547, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.345306672155857, 'response_sent_to_client_ts': 1775629372.3955638}}</strong>



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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    2026-04-08 06:23:10.134 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:23:10] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:23:10.134 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:23:10] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:23:10.134 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:23:10] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:23:10.134 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:23:10] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:23:10.134 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:23:10] Persistent cache disabled, using in-memory JIT cache


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:00<00:00,  1.01it/s]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.07s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.06s/it]


    2026-04-08 06:23:15,168 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 06:23:15] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:55,  3.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:55,  3.07s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:27,  1.56s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:27,  1.56s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:56,  1.02s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:56,  1.02s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:40,  1.32it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:25,  2.05it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:25,  2.05it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:21,  2.43it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:21,  2.43it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:17,  2.79it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:17,  2.79it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:15,  3.21it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:15,  3.21it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:12,  4.00it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:12,  4.00it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:10,  4.36it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:10,  4.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:09,  4.78it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:09,  4.78it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:08,  5.21it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:08,  5.21it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:07,  5.65it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:07,  5.65it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:06,  6.17it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:06,  6.17it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:06,  6.95it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:06,  6.95it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:06<00:06,  6.95it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:06<00:04,  9.52it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:06<00:04,  9.52it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:06<00:04,  9.52it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:06<00:04,  9.52it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:06<00:02, 14.14it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:06<00:02, 14.14it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:06<00:02, 14.14it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:06<00:02, 14.14it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:06<00:02, 14.14it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:06<00:01, 20.19it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:06<00:01, 20.19it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:06<00:01, 20.19it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:06<00:01, 20.19it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:06<00:01, 20.19it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:07<00:01, 20.19it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:01, 27.56it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:01, 27.56it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:01, 27.56it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:01, 27.56it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:01, 27.56it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:01, 27.56it/s]

    Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:01, 27.56it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 34.83it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 34.83it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 34.83it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 34.83it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 34.83it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 34.83it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 34.83it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:07<00:00, 39.90it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:07<00:00, 39.90it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:07<00:00, 39.90it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:07<00:00, 39.90it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:07<00:00, 39.90it/s]

    Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:07<00:00, 39.90it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:07<00:00, 42.13it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:07<00:00, 42.13it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:07<00:00, 42.13it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:07<00:00, 42.13it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:07<00:00, 42.13it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:07<00:00, 42.13it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:07<00:00, 42.13it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:07<00:00, 46.20it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:07<00:00, 46.20it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:07<00:00, 46.20it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:07<00:00, 46.20it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:07<00:00, 46.20it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:07<00:00, 46.20it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=102.52 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=102.52 GB):   2%|▏         | 1/58 [00:00<00:31,  1.82it/s]Capturing num tokens (num_tokens=7680 avail_mem=102.48 GB):   2%|▏         | 1/58 [00:00<00:31,  1.82it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=102.48 GB):   3%|▎         | 2/58 [00:00<00:22,  2.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=102.48 GB):   3%|▎         | 2/58 [00:00<00:22,  2.52it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=102.48 GB):   5%|▌         | 3/58 [00:01<00:18,  2.98it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.14 GB):   5%|▌         | 3/58 [00:01<00:18,  2.98it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=120.14 GB):   7%|▋         | 4/58 [00:01<00:15,  3.49it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.14 GB):   7%|▋         | 4/58 [00:01<00:15,  3.49it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.14 GB):   9%|▊         | 5/58 [00:01<00:14,  3.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.15 GB):   9%|▊         | 5/58 [00:01<00:14,  3.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.15 GB):  10%|█         | 6/58 [00:01<00:13,  3.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.15 GB):  10%|█         | 6/58 [00:01<00:13,  3.96it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=120.15 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.15 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.15 GB):  14%|█▍        | 8/58 [00:02<00:09,  5.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.16 GB):  14%|█▍        | 8/58 [00:02<00:09,  5.05it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=120.16 GB):  16%|█▌        | 9/58 [00:13<03:01,  3.70s/it]Capturing num tokens (num_tokens=3840 avail_mem=119.87 GB):  16%|█▌        | 9/58 [00:13<03:01,  3.70s/it]

    Capturing num tokens (num_tokens=3840 avail_mem=119.87 GB):  17%|█▋        | 10/58 [00:13<02:06,  2.63s/it]Capturing num tokens (num_tokens=3584 avail_mem=119.86 GB):  17%|█▋        | 10/58 [00:13<02:06,  2.63s/it]

    Capturing num tokens (num_tokens=3584 avail_mem=119.86 GB):  19%|█▉        | 11/58 [00:13<01:29,  1.90s/it]Capturing num tokens (num_tokens=3328 avail_mem=119.86 GB):  19%|█▉        | 11/58 [00:13<01:29,  1.90s/it]Capturing num tokens (num_tokens=3328 avail_mem=119.86 GB):  21%|██        | 12/58 [00:14<01:02,  1.36s/it]Capturing num tokens (num_tokens=3072 avail_mem=119.86 GB):  21%|██        | 12/58 [00:14<01:02,  1.36s/it]

    Capturing num tokens (num_tokens=3072 avail_mem=119.86 GB):  22%|██▏       | 13/58 [00:14<00:44,  1.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.86 GB):  22%|██▏       | 13/58 [00:14<00:44,  1.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.86 GB):  24%|██▍       | 14/58 [00:14<00:31,  1.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.86 GB):  24%|██▍       | 14/58 [00:14<00:31,  1.39it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=119.86 GB):  24%|██▍       | 14/58 [00:14<00:31,  1.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=119.86 GB):  28%|██▊       | 16/58 [00:14<00:17,  2.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=119.86 GB):  28%|██▊       | 16/58 [00:14<00:17,  2.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.86 GB):  28%|██▊       | 16/58 [00:14<00:17,  2.33it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=119.86 GB):  31%|███       | 18/58 [00:14<00:11,  3.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=119.85 GB):  31%|███       | 18/58 [00:14<00:11,  3.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=119.85 GB):  31%|███       | 18/58 [00:14<00:11,  3.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=119.85 GB):  34%|███▍      | 20/58 [00:14<00:07,  4.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=119.85 GB):  34%|███▍      | 20/58 [00:14<00:07,  4.85it/s]Capturing num tokens (num_tokens=960 avail_mem=119.85 GB):  34%|███▍      | 20/58 [00:14<00:07,  4.85it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=119.85 GB):  38%|███▊      | 22/58 [00:14<00:05,  6.53it/s]Capturing num tokens (num_tokens=896 avail_mem=119.85 GB):  38%|███▊      | 22/58 [00:14<00:05,  6.53it/s]Capturing num tokens (num_tokens=832 avail_mem=119.84 GB):  38%|███▊      | 22/58 [00:14<00:05,  6.53it/s]Capturing num tokens (num_tokens=768 avail_mem=119.84 GB):  38%|███▊      | 22/58 [00:14<00:05,  6.53it/s]Capturing num tokens (num_tokens=768 avail_mem=119.84 GB):  43%|████▎     | 25/58 [00:15<00:03,  9.52it/s]Capturing num tokens (num_tokens=704 avail_mem=119.83 GB):  43%|████▎     | 25/58 [00:15<00:03,  9.52it/s]Capturing num tokens (num_tokens=640 avail_mem=119.83 GB):  43%|████▎     | 25/58 [00:15<00:03,  9.52it/s]Capturing num tokens (num_tokens=576 avail_mem=119.83 GB):  43%|████▎     | 25/58 [00:15<00:03,  9.52it/s]

    Capturing num tokens (num_tokens=576 avail_mem=119.83 GB):  48%|████▊     | 28/58 [00:15<00:02, 12.69it/s]Capturing num tokens (num_tokens=512 avail_mem=119.82 GB):  48%|████▊     | 28/58 [00:15<00:02, 12.69it/s]Capturing num tokens (num_tokens=480 avail_mem=119.82 GB):  48%|████▊     | 28/58 [00:15<00:02, 12.69it/s]Capturing num tokens (num_tokens=448 avail_mem=119.82 GB):  48%|████▊     | 28/58 [00:15<00:02, 12.69it/s]Capturing num tokens (num_tokens=416 avail_mem=119.81 GB):  48%|████▊     | 28/58 [00:15<00:02, 12.69it/s]Capturing num tokens (num_tokens=416 avail_mem=119.81 GB):  55%|█████▌    | 32/58 [00:15<00:01, 17.22it/s]Capturing num tokens (num_tokens=384 avail_mem=119.81 GB):  55%|█████▌    | 32/58 [00:15<00:01, 17.22it/s]Capturing num tokens (num_tokens=352 avail_mem=119.80 GB):  55%|█████▌    | 32/58 [00:15<00:01, 17.22it/s]Capturing num tokens (num_tokens=320 avail_mem=119.47 GB):  55%|█████▌    | 32/58 [00:15<00:01, 17.22it/s]

    Capturing num tokens (num_tokens=320 avail_mem=119.47 GB):  60%|██████    | 35/58 [00:15<00:01, 18.89it/s]Capturing num tokens (num_tokens=288 avail_mem=119.51 GB):  60%|██████    | 35/58 [00:15<00:01, 18.89it/s]Capturing num tokens (num_tokens=256 avail_mem=119.76 GB):  60%|██████    | 35/58 [00:15<00:01, 18.89it/s]Capturing num tokens (num_tokens=240 avail_mem=119.76 GB):  60%|██████    | 35/58 [00:15<00:01, 18.89it/s]Capturing num tokens (num_tokens=240 avail_mem=119.76 GB):  66%|██████▌   | 38/58 [00:15<00:01, 18.51it/s]Capturing num tokens (num_tokens=224 avail_mem=119.75 GB):  66%|██████▌   | 38/58 [00:15<00:01, 18.51it/s]

    Capturing num tokens (num_tokens=208 avail_mem=119.74 GB):  66%|██████▌   | 38/58 [00:15<00:01, 18.51it/s]Capturing num tokens (num_tokens=192 avail_mem=119.74 GB):  66%|██████▌   | 38/58 [00:15<00:01, 18.51it/s]Capturing num tokens (num_tokens=192 avail_mem=119.74 GB):  71%|███████   | 41/58 [00:15<00:00, 19.12it/s]Capturing num tokens (num_tokens=176 avail_mem=119.74 GB):  71%|███████   | 41/58 [00:15<00:00, 19.12it/s]Capturing num tokens (num_tokens=160 avail_mem=119.73 GB):  71%|███████   | 41/58 [00:15<00:00, 19.12it/s]Capturing num tokens (num_tokens=144 avail_mem=119.72 GB):  71%|███████   | 41/58 [00:15<00:00, 19.12it/s]

    Capturing num tokens (num_tokens=144 avail_mem=119.72 GB):  76%|███████▌  | 44/58 [00:15<00:00, 19.67it/s]Capturing num tokens (num_tokens=128 avail_mem=119.73 GB):  76%|███████▌  | 44/58 [00:15<00:00, 19.67it/s]Capturing num tokens (num_tokens=112 avail_mem=119.72 GB):  76%|███████▌  | 44/58 [00:15<00:00, 19.67it/s]Capturing num tokens (num_tokens=96 avail_mem=119.71 GB):  76%|███████▌  | 44/58 [00:15<00:00, 19.67it/s] Capturing num tokens (num_tokens=96 avail_mem=119.71 GB):  81%|████████  | 47/58 [00:15<00:00, 21.26it/s]Capturing num tokens (num_tokens=80 avail_mem=119.71 GB):  81%|████████  | 47/58 [00:15<00:00, 21.26it/s]Capturing num tokens (num_tokens=64 avail_mem=119.70 GB):  81%|████████  | 47/58 [00:15<00:00, 21.26it/s]

    Capturing num tokens (num_tokens=48 avail_mem=119.70 GB):  81%|████████  | 47/58 [00:16<00:00, 21.26it/s]Capturing num tokens (num_tokens=48 avail_mem=119.70 GB):  86%|████████▌ | 50/58 [00:16<00:00, 20.97it/s]Capturing num tokens (num_tokens=32 avail_mem=119.69 GB):  86%|████████▌ | 50/58 [00:16<00:00, 20.97it/s]Capturing num tokens (num_tokens=28 avail_mem=119.63 GB):  86%|████████▌ | 50/58 [00:16<00:00, 20.97it/s]Capturing num tokens (num_tokens=24 avail_mem=119.64 GB):  86%|████████▌ | 50/58 [00:16<00:00, 20.97it/s]

    Capturing num tokens (num_tokens=24 avail_mem=119.64 GB):  91%|█████████▏| 53/58 [00:16<00:00, 16.76it/s]Capturing num tokens (num_tokens=20 avail_mem=119.67 GB):  91%|█████████▏| 53/58 [00:16<00:00, 16.76it/s]Capturing num tokens (num_tokens=16 avail_mem=119.66 GB):  91%|█████████▏| 53/58 [00:16<00:00, 16.76it/s]Capturing num tokens (num_tokens=16 avail_mem=119.66 GB):  95%|█████████▍| 55/58 [00:16<00:00, 17.18it/s]Capturing num tokens (num_tokens=12 avail_mem=119.64 GB):  95%|█████████▍| 55/58 [00:16<00:00, 17.18it/s]Capturing num tokens (num_tokens=8 avail_mem=119.63 GB):  95%|█████████▍| 55/58 [00:16<00:00, 17.18it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=119.63 GB):  98%|█████████▊| 57/58 [00:16<00:00, 17.74it/s]Capturing num tokens (num_tokens=4 avail_mem=119.62 GB):  98%|█████████▊| 57/58 [00:16<00:00, 17.74it/s]Capturing num tokens (num_tokens=4 avail_mem=119.62 GB): 100%|██████████| 58/58 [00:16<00:00,  3.50it/s]


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
        "population": 138000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
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
    
    Generated text: Okay, so I need to figure out the information and population of the capital of France. Hmm, let me start by recalling what I know about France. I know that Paris is the capital, but I'm not exactly sure about its current population. I remember that it's a major city with a significant population, but I don't know the exact number.
    
    I think I've heard it's around 10 million or something like that. Wait, maybe it's a bit more? I'm not certain. I should try to verify. I can't access the internet, so I have to go by what I remember or what I can deduce logically.
    
    I know that major European cities like London, which is the capital of the UK, has a population of about 9 million. Maybe Paris is similar or a bit larger. I've heard that cities in France often have populations in the millions, but not too many. So maybe Paris is around 10 to 12 million?
    
    I also remember that Paris is a very dense city. It has a lot of people living in the same area, so the population density is high. But I'm not sure if that affects the total population. Maybe the area is large enough that even with high density, the total doesn't get too high.
    
    I think the population might be around 10.5 million. I've heard that number before, but I'm not 100% sure. It might vary a bit depending on the source. To be precise, perhaps I should go with a rounded number, like 11 million. But I'm still uncertain.
    
    Let me try to think about other capitals in similar countries. For example, Tokyo in Japan is one of the most populous cities in the country, but Japan's population is around 126 million, so each city can't be too large. France's population is about 40 million, so dividing that among its 12 capitals would average around 3.3 million each. That suggests that Paris, being a major city, might be higher than that average.
    
    Wait, maybe I'm overcomplicating it. I think I was correct the first time. Paris's population is around 10.5 million. I've heard it referred to as about 10.5 million, so that seems plausible. I should just go with that number for now.
    
    So, putting it all together, the capital of France is Paris, and its population is approximately 10.5 million. I should present this information in JSON format as the user requested.
    </think>
    
    The capital of France is Paris, and its approximate population is 10.5 million.
    
    ```json
    {
      "capital": "Paris",
      "population": 10500000
    }
    ```



```python
llm.shutdown()
```
