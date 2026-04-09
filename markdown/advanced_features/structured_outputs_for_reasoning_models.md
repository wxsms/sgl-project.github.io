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


    [2026-04-09 07:16:54] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 07:16:54] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 07:16:54] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 07:16:54] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.25s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.60s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.55s/it]


    2026-04-09 07:16:58,475 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 07:16:58] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:50,  2.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:50,  2.99s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:34,  1.68s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:34,  1.68s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:55,  1.01s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:55,  1.01s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:26,  2.03it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:26,  2.03it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:15,  3.36it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:15,  3.36it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:12,  4.14it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:12,  4.14it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  4.99it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  4.99it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  4.99it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:07,  6.65it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:07,  6.65it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:07,  6.65it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:05,  8.12it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:05,  8.12it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:05,  8.12it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.64it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.64it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.64it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.55it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 14.79it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 14.79it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 14.79it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 14.79it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 14.79it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 19.70it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 19.70it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 19.70it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 19.70it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 19.04it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 19.04it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 19.04it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 19.04it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 19.46it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 19.46it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:06<00:01, 19.46it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:06<00:01, 19.46it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:06<00:01, 19.51it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:06<00:01, 19.51it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:06<00:01, 19.51it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:06<00:01, 19.51it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:06<00:01, 20.30it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:06<00:01, 20.30it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:06<00:01, 20.30it/s]

    Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:06<00:01, 20.30it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:06<00:00, 21.93it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:06<00:00, 21.93it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:06<00:00, 21.93it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:06<00:00, 21.93it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:06<00:00, 22.47it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:06<00:00, 22.47it/s]

    Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:06<00:00, 22.47it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:06<00:00, 22.47it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 22.94it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 22.94it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 22.94it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 22.94it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:06<00:00, 23.72it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:06<00:00, 23.72it/s]

    Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:06<00:00, 23.72it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:06<00:00, 23.72it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 24.19it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 24.19it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 24.19it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 24.19it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 24.19it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 26.42it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 26.42it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:07<00:00, 26.42it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:07<00:00, 26.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  8.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=44.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=44.17 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.14 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.14 GB):   3%|▎         | 2/58 [00:01<00:34,  1.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=44.14 GB):   3%|▎         | 2/58 [00:01<00:34,  1.64it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=44.14 GB):   5%|▌         | 3/58 [00:01<00:30,  1.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.15 GB):   5%|▌         | 3/58 [00:01<00:30,  1.78it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=44.15 GB):   7%|▋         | 4/58 [00:02<00:27,  1.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.15 GB):   7%|▋         | 4/58 [00:02<00:27,  1.98it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.15 GB):   9%|▊         | 5/58 [00:02<00:23,  2.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.16 GB):   9%|▊         | 5/58 [00:02<00:23,  2.22it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=44.16 GB):  10%|█         | 6/58 [00:02<00:20,  2.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.16 GB):  10%|█         | 6/58 [00:02<00:20,  2.56it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=44.16 GB):  12%|█▏        | 7/58 [00:03<00:17,  2.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.16 GB):  12%|█▏        | 7/58 [00:03<00:17,  2.86it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=44.16 GB):  14%|█▍        | 8/58 [00:03<00:15,  3.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.16 GB):  14%|█▍        | 8/58 [00:03<00:15,  3.17it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=44.16 GB):  16%|█▌        | 9/58 [00:03<00:13,  3.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.16 GB):  16%|█▌        | 9/58 [00:03<00:13,  3.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.16 GB):  17%|█▋        | 10/58 [00:03<00:12,  3.98it/s]Capturing num tokens (num_tokens=3584 avail_mem=44.16 GB):  17%|█▋        | 10/58 [00:03<00:12,  3.98it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=44.16 GB):  19%|█▉        | 11/58 [00:03<00:09,  4.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.16 GB):  19%|█▉        | 11/58 [00:03<00:09,  4.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.16 GB):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.97 GB):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=42.97 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.11 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.11 GB):  24%|██▍       | 14/58 [00:04<00:08,  4.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.13 GB):  24%|██▍       | 14/58 [00:04<00:08,  4.89it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=44.13 GB):  26%|██▌       | 15/58 [00:04<00:08,  4.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.14 GB):  26%|██▌       | 15/58 [00:04<00:08,  4.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.14 GB):  28%|██▊       | 16/58 [00:04<00:08,  4.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.13 GB):  28%|██▊       | 16/58 [00:04<00:08,  4.98it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=44.13 GB):  29%|██▉       | 17/58 [00:04<00:07,  5.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.21 GB):  29%|██▉       | 17/58 [00:04<00:07,  5.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.21 GB):  31%|███       | 18/58 [00:05<00:07,  5.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.72 GB):  31%|███       | 18/58 [00:05<00:07,  5.39it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=43.72 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.27 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.27 GB):  34%|███▍      | 20/58 [00:05<00:06,  6.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.27 GB):  34%|███▍      | 20/58 [00:05<00:06,  6.17it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=43.27 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.78it/s]Capturing num tokens (num_tokens=960 avail_mem=44.13 GB):  36%|███▌      | 21/58 [00:05<00:05,  6.78it/s] Capturing num tokens (num_tokens=960 avail_mem=44.13 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.23it/s]Capturing num tokens (num_tokens=896 avail_mem=43.34 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.23it/s]

    Capturing num tokens (num_tokens=896 avail_mem=43.34 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.41it/s]Capturing num tokens (num_tokens=832 avail_mem=44.21 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.41it/s]Capturing num tokens (num_tokens=768 avail_mem=43.40 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.41it/s]

    Capturing num tokens (num_tokens=768 avail_mem=43.40 GB):  43%|████▎     | 25/58 [00:06<00:04,  8.23it/s]Capturing num tokens (num_tokens=704 avail_mem=43.39 GB):  43%|████▎     | 25/58 [00:06<00:04,  8.23it/s]Capturing num tokens (num_tokens=704 avail_mem=43.39 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.53it/s]Capturing num tokens (num_tokens=640 avail_mem=44.12 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.53it/s]Capturing num tokens (num_tokens=576 avail_mem=43.45 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.53it/s]

    Capturing num tokens (num_tokens=576 avail_mem=43.45 GB):  48%|████▊     | 28/58 [00:06<00:03,  7.90it/s]Capturing num tokens (num_tokens=512 avail_mem=44.11 GB):  48%|████▊     | 28/58 [00:06<00:03,  7.90it/s]Capturing num tokens (num_tokens=480 avail_mem=43.68 GB):  48%|████▊     | 28/58 [00:06<00:03,  7.90it/s]Capturing num tokens (num_tokens=480 avail_mem=43.68 GB):  52%|█████▏    | 30/58 [00:06<00:03,  8.97it/s]Capturing num tokens (num_tokens=448 avail_mem=43.51 GB):  52%|█████▏    | 30/58 [00:06<00:03,  8.97it/s]

    Capturing num tokens (num_tokens=448 avail_mem=43.51 GB):  53%|█████▎    | 31/58 [00:06<00:03,  7.33it/s]Capturing num tokens (num_tokens=416 avail_mem=44.11 GB):  53%|█████▎    | 31/58 [00:06<00:03,  7.33it/s]Capturing num tokens (num_tokens=384 avail_mem=43.58 GB):  53%|█████▎    | 31/58 [00:06<00:03,  7.33it/s]Capturing num tokens (num_tokens=384 avail_mem=43.58 GB):  57%|█████▋    | 33/58 [00:06<00:03,  8.19it/s]Capturing num tokens (num_tokens=352 avail_mem=44.10 GB):  57%|█████▋    | 33/58 [00:06<00:03,  8.19it/s]

    Capturing num tokens (num_tokens=320 avail_mem=43.60 GB):  57%|█████▋    | 33/58 [00:07<00:03,  8.19it/s]Capturing num tokens (num_tokens=320 avail_mem=43.60 GB):  60%|██████    | 35/58 [00:07<00:02,  9.30it/s]Capturing num tokens (num_tokens=288 avail_mem=44.09 GB):  60%|██████    | 35/58 [00:07<00:02,  9.30it/s]Capturing num tokens (num_tokens=256 avail_mem=43.63 GB):  60%|██████    | 35/58 [00:07<00:02,  9.30it/s]

    Capturing num tokens (num_tokens=256 avail_mem=43.63 GB):  64%|██████▍   | 37/58 [00:07<00:02, 10.33it/s]Capturing num tokens (num_tokens=240 avail_mem=44.08 GB):  64%|██████▍   | 37/58 [00:07<00:02, 10.33it/s]Capturing num tokens (num_tokens=224 avail_mem=43.65 GB):  64%|██████▍   | 37/58 [00:07<00:02, 10.33it/s]Capturing num tokens (num_tokens=224 avail_mem=43.65 GB):  67%|██████▋   | 39/58 [00:07<00:01, 11.30it/s]Capturing num tokens (num_tokens=208 avail_mem=44.08 GB):  67%|██████▋   | 39/58 [00:07<00:01, 11.30it/s]Capturing num tokens (num_tokens=192 avail_mem=43.68 GB):  67%|██████▋   | 39/58 [00:07<00:01, 11.30it/s]

    Capturing num tokens (num_tokens=192 avail_mem=43.68 GB):  71%|███████   | 41/58 [00:07<00:01, 12.23it/s]Capturing num tokens (num_tokens=176 avail_mem=44.07 GB):  71%|███████   | 41/58 [00:07<00:01, 12.23it/s]Capturing num tokens (num_tokens=160 avail_mem=43.81 GB):  71%|███████   | 41/58 [00:07<00:01, 12.23it/s]Capturing num tokens (num_tokens=160 avail_mem=43.81 GB):  74%|███████▍  | 43/58 [00:07<00:01, 13.15it/s]Capturing num tokens (num_tokens=144 avail_mem=44.07 GB):  74%|███████▍  | 43/58 [00:07<00:01, 13.15it/s]Capturing num tokens (num_tokens=128 avail_mem=43.74 GB):  74%|███████▍  | 43/58 [00:07<00:01, 13.15it/s]

    Capturing num tokens (num_tokens=128 avail_mem=43.74 GB):  78%|███████▊  | 45/58 [00:07<00:00, 14.09it/s]Capturing num tokens (num_tokens=112 avail_mem=44.07 GB):  78%|███████▊  | 45/58 [00:07<00:00, 14.09it/s]Capturing num tokens (num_tokens=96 avail_mem=43.77 GB):  78%|███████▊  | 45/58 [00:07<00:00, 14.09it/s] Capturing num tokens (num_tokens=96 avail_mem=43.77 GB):  81%|████████  | 47/58 [00:07<00:00, 14.94it/s]Capturing num tokens (num_tokens=80 avail_mem=44.06 GB):  81%|████████  | 47/58 [00:07<00:00, 14.94it/s]Capturing num tokens (num_tokens=64 avail_mem=44.06 GB):  81%|████████  | 47/58 [00:08<00:00, 14.94it/s]

    Capturing num tokens (num_tokens=64 avail_mem=44.06 GB):  84%|████████▍ | 49/58 [00:08<00:00, 15.56it/s]Capturing num tokens (num_tokens=48 avail_mem=43.82 GB):  84%|████████▍ | 49/58 [00:08<00:00, 15.56it/s]Capturing num tokens (num_tokens=32 avail_mem=44.05 GB):  84%|████████▍ | 49/58 [00:08<00:00, 15.56it/s]Capturing num tokens (num_tokens=32 avail_mem=44.05 GB):  88%|████████▊ | 51/58 [00:08<00:00, 16.32it/s]Capturing num tokens (num_tokens=28 avail_mem=43.91 GB):  88%|████████▊ | 51/58 [00:08<00:00, 16.32it/s]Capturing num tokens (num_tokens=24 avail_mem=44.03 GB):  88%|████████▊ | 51/58 [00:08<00:00, 16.32it/s]

    Capturing num tokens (num_tokens=24 avail_mem=44.03 GB):  91%|█████████▏| 53/58 [00:08<00:00, 16.81it/s]Capturing num tokens (num_tokens=20 avail_mem=44.04 GB):  91%|█████████▏| 53/58 [00:08<00:00, 16.81it/s]

    Capturing num tokens (num_tokens=16 avail_mem=44.03 GB):  91%|█████████▏| 53/58 [00:08<00:00, 16.81it/s]Capturing num tokens (num_tokens=16 avail_mem=44.03 GB):  95%|█████████▍| 55/58 [00:08<00:00, 11.03it/s]Capturing num tokens (num_tokens=12 avail_mem=43.92 GB):  95%|█████████▍| 55/58 [00:08<00:00, 11.03it/s]Capturing num tokens (num_tokens=8 avail_mem=44.02 GB):  95%|█████████▍| 55/58 [00:08<00:00, 11.03it/s] Capturing num tokens (num_tokens=8 avail_mem=44.02 GB):  98%|█████████▊| 57/58 [00:08<00:00, 12.64it/s]Capturing num tokens (num_tokens=4 avail_mem=44.01 GB):  98%|█████████▊| 57/58 [00:08<00:00, 12.64it/s]

    Capturing num tokens (num_tokens=4 avail_mem=44.01 GB): 100%|██████████| 58/58 [00:08<00:00,  6.63it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should check if that's correct.<br><br>Wait, I think the population might have changed a bit over the years. I recall reading somewhere that Paris has grown a lot, especially with the influx of people moving there for work. But I'm not sure if it's exactly 21 million or maybe a bit more. I should look up the latest data to confirm.<br><br>I also wonder if the population figure includes just the city proper or the entire metropolitan area. Sometimes, people talk about the metro area, which can be much larger. But the question specifically asks for the population of the capital, so I think it refers to the city limits. Still, I should make sure.<br><br>Another thing to consider is that population figures can vary depending on the source. Some might cite estimates from government agencies, while others might use more recent data from censuses or surveys. I should find a reliable source to get the most accurate number.<br><br>I think the population of Paris is around 21 million, but I'm not 100% sure. Maybe I should think about other major cities in France to compare. For example, Lyon is another big city, but it's much smaller. I believe its population is around 1.2 million. That gives me a sense that Paris is significantly larger.<br><br>Also, considering the economic activities in Paris, like the fashion industry and the entertainment sector, it makes sense that it's the capital and has a large population. The city hosts a lot of events, conventions, and businesses, which would attract a diverse population.<br><br>I should also think about the historical growth of Paris. It's been a major city for centuries, so its population has been increasing steadily. I think it's safe to say that it's over 20 million, but I'm still not certain about the exact number.<br><br>In summary, I'm pretty confident that the capital of France is Paris, and its population is around 21 million. However, to be precise, I should look up the latest statistics to confirm the exact figure.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21538000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should check if that's correct.<br><br>Wait, I think the population might have changed a bit over the years. I recall reading somewhere that it's over 21 million, but I'm not sure if it's 21.5 or 22 million. Also, I should consider whether the population figure is as of a specific year, like 2021 or 2022, because populations can increase over time.<br><br>I also wonder if there are any other factors to consider, like whether the population includes just the city proper or the broader metropolitan area. Sometimes, population figures can include surrounding regions. But I think in this case, since the user asked for the population of the capital, it's probably referring to the city limits.<br><br>Another thing to consider is the source of the data. Is it from a reliable government website or a recent census? I should make sure the information is up-to-date and accurate. Maybe I can cross-reference this with a recent source to confirm the population number.<br><br>So, putting it all together, I'm pretty confident that Paris is the capital of France, and its population is around 21.5 million. I'll go with that and present it in the JSON format as requested.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21500000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light." People go there for museums, landmarks like the Eiffel Tower, and it's a cultural hub. But is it the capital?<br><br>Wait, I think the capital is the official seat of government, right? So maybe Paris is both the capital and the most famous city. But I'm not entirely certain. I recall that some countries have their capital in a different city than their main tourist attraction. For example, I think Brazil's capital is not Rio de Janeiro, which is more famous. So maybe France is like that too.<br><br>Let me try to remember any specific information. I think the French government declares Paris as the capital. I've heard that before. Also, I remember learning that the Eiffel Tower is in Paris, which is a symbol of the city, but not necessarily the government building. The government buildings are probably in another part of the city or in a different city altogether.<br><br>Wait, no, I think the government buildings are in Paris. Maybe the Palace of Consultation or something like that. I'm not sure of the exact name, but I know it's in Paris. So if the government is based there, then Paris would be the capital. That makes sense because the capital is where all the official government work happens.<br><br>I also think that other capitals, like Germany's Berlin or Italy's Rome, are more than just tourist spots. So, yes, Paris being the capital of France fits into that pattern. It's both the main city and the government center.<br><br>I guess I'm pretty confident now. Paris is the capital of France because it's where the government is located, even though it's also a major tourist destination. So the answer should be Paris.<br><br><br>content: Paris</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants to know the current date and time, as well as the weather. Let me figure out how to approach this.<br><br>First, I need to determine which functions to use. The user mentioned two possibilities: get_current_date and get_current_weather. Both are available, so I can call them separately.<br><br>I should start by getting the current date and time. The get_current_date function requires a timezone parameter. Since the user is in New York, I'll use 'America/New_York' as the timezone. I'll structure the function call as <function=get_current_date '{"timezone": "America/New_York"}' and make sure it's on one line.<br><br>Next, for the weather, I'll use get_current_weather. This function needs the city, state, and unit. The city is New York, the state is NY, and I'll choose Fahrenheit since the user didn't specify. So the parameters will be {'city': 'New York', 'state': 'NY', 'unit': 'fahrenheit'}. I'll format this similarly to the date function.<br><br>I should also remember to include the sources in the response. For the date, it's from get_current_date, and for the weather, it's from get_current_weather. Each result should have a source line below it.<br><br>Putting it all together, I'll send two separate function calls, each with their parameters in the specified JSON format, and add the sources at the end of each response.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br>Source: get_current_date  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>  <br>Source: get_current_weather</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not exactly sure of the exact number. I think it\'s around 2 million, but I should double-check that.\n\nWait, maybe I should look up the latest statistics to be accurate. I recall that in recent years, Paris has seen a steady increase due to urban development and immigration. Let me see... I think the population is approximately 2,170,000 as of 2023. That seems about right.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the city name, its population, and maybe the country it\'s in. So, the keys could be "city", "population", and "country".\n\nPutting it all together, the JSON should look something like this: {"city": "Paris", "population": 2170000, "country": "France"}. I should make sure the syntax is correct, with proper commas and quotation marks.\n\nI should also consider if the user might need more details, like the metropolitan area population or other statistics, but since they specifically asked for the capital, I\'ll stick to that. Maybe mention that the data is up to date as of 2023 to provide context.\n\nAlright, I think that\'s all. I\'ll present the JSON and offer further assistance in case they need more information.\n</think>{\n  "name": "Paris",\n  "population": 2170000\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 382, 14190, 11, 7196, 358, 1265, 1401, 705, 279, 5535, 13142, 311, 387, 13382, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 24020, 5263, 4152, 311, 15662, 4401, 323, 15093, 13, 6771, 752, 1490, 1112, 358, 1744, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 3283, 829, 11, 1181, 7042, 11, 323, 7196, 279, 3146, 432, 594, 304, 13, 2055, 11, 279, 6894, 1410, 387, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 11436, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 5212, 8926, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 11, 330, 11141, 788, 330, 49000, 1, 7810, 358, 1265, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 57406, 3082, 7042, 476, 1008, 13142, 11, 714, 2474, 807, 11689, 4588, 369, 279, 6722, 11, 358, 3278, 9214, 311, 429, 13, 10696, 6286, 429, 279, 821, 374, 705, 311, 2400, 438, 315, 220, 17, 15, 17, 18, 311, 3410, 2266, 382, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3042, 279, 4718, 323, 3010, 4623, 12994, 304, 1142, 807, 1184, 803, 1995, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 198, 92, 151643], 'meta_info': {'id': 'fd7f68b65d7c4a6a938bf34b141b5689', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 369, 'completion_tokens': 392, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.127751404885203, 'response_sent_to_client_ts': 1775719066.673414}}



<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.<br><br>First, I need to identify the capital of France. I know that Paris is the capital, so that's straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I'm not exactly sure of the exact number. I think it's around 2 million, but I should double-check that.<br><br>Wait, maybe I should look up the latest statistics to be accurate. I recall that in recent years, Paris has seen a steady increase due to urban development and immigration. Let me see... I think the population is approximately 2,170,000 as of 2023. That seems about right.<br><br>Now, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the city name, its population, and maybe the country it's in. So, the keys could be "city", "population", and "country".<br><br>Putting it all together, the JSON should look something like this: {"city": "Paris", "population": 2170000, "country": "France"}. I should make sure the syntax is correct, with proper commas and quotation marks.<br><br>I should also consider if the user might need more details, like the metropolitan area population or other statistics, but since they specifically asked for the capital, I'll stick to that. Maybe mention that the data is up to date as of 2023 to provide context.<br><br>Alright, I think that's all. I'll present the JSON and offer further assistance in case they need more information.<br><br><br>content: {<br>  "name": "Paris",<br>  "population": 2170000<br>}</strong>


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not exactly sure of the exact number. I think it\'s around 2 million, but I should double-check that.\n\nWait, maybe I should look up the latest statistics to be accurate. I recall that in recent years, Paris has seen a steady increase due to urban development and immigration. Let me see... I think the population is approximately 2,170,000 as of 2023. That seems about right.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the city name, its population, and maybe the country it\'s in. So, the keys could be "city", "population", and "country".\n\nPutting it all together, the JSON should look something like this: {"city": "Paris", "population": 2170000, "country": "France"}. I should make sure the syntax is correct, with proper commas and quotation marks.\n\nI should also consider if the user might need more details, like the metropolitan area population or other statistics, but since they specifically asked for the capital, I\'ll stick to that. Maybe mention that the data is up to date as of 2023 to provide context.\n\nAlright, I think that\'s all. I\'ll present the JSON and offer further assistance in case they need more information.\n</think>{\n  "name": "Paris",\n  "population": 2170000\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 382, 14190, 11, 7196, 358, 1265, 1401, 705, 279, 5535, 13142, 311, 387, 13382, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 24020, 5263, 4152, 311, 15662, 4401, 323, 15093, 13, 6771, 752, 1490, 1112, 358, 1744, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 3283, 829, 11, 1181, 7042, 11, 323, 7196, 279, 3146, 432, 594, 304, 13, 2055, 11, 279, 6894, 1410, 387, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 11436, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 5212, 8926, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 11, 330, 11141, 788, 330, 49000, 1, 7810, 358, 1265, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 57406, 3082, 7042, 476, 1008, 13142, 11, 714, 2474, 807, 11689, 4588, 369, 279, 6722, 11, 358, 3278, 9214, 311, 429, 13, 10696, 6286, 429, 279, 821, 374, 705, 311, 2400, 438, 315, 220, 17, 15, 17, 18, 311, 3410, 2266, 382, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3042, 279, 4718, 323, 3010, 4623, 12994, 304, 1142, 807, 1184, 803, 1995, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 198, 92, 151643], 'meta_info': {'id': 'a68d30b60b85490cb2a295f64b6dccf0', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 369, 'completion_tokens': 392, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.594631304033101, 'response_sent_to_client_ts': 1775719072.2769318}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '4ee12182b27443628e30ff6bb1bc9455', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.22580321482382715, 'response_sent_to_client_ts': 1775719072.5737748}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'b19f2e61c81246c39056ecb8a966b32f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.22570887091569602, 'response_sent_to_client_ts': 1775719072.5737884}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '66b86c2a3e6041668417f803014dac2c', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.22567095910198987, 'response_sent_to_client_ts': 1775719072.573794}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '7a74cf75ec944298922e10399ee2aa3c', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 24.585766836069524, 'response_sent_to_client_ts': 1775719097.1663013}}


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


<strong style='color: #00008B;'>{'text': 'Alright, so the user is asking for the information and population of France\'s capital, which is Paris. They specifically want it in JSON format. Hmm, I should first make sure I have the correct details about Paris. \n\nI know that Paris is the administrative capital, but sometimes people confuse it with the political capital. The official name is Paris City, right? And the region is Île-de-France. I should include that information too because it\'s relevant.\n\nNow, population stats. I think the population is approximately 2.1 million, but that\'s an average from recent years. It can fluctuate a bit depending on various factors like trends. So mentioning that it\'s an approximate figure makes sense.\n\nFor the ZIP code, Paris is spread out, so it doesn\'t have a single ZIP code. Instead, it has multiple postal codes, so I should list the most commonly used ones. 75001 is the central postcode, and there are others around it. Including that adds completeness to the response.\n\nThe date when Paris became the capital is important too. It was decided in 18th-century reforms, and I think it was the result of the defeat of the French highlights during the Napoleonic Wars. That context might help the user understand why Paris was chosen as the capital.\n\nPutting it all together, I should structure it in a JSON format as the user requested. I\'ll use the public .json() method to get accurate data for the population estimate. Let me also recall that the latest data is around 2023, so I should mention that the figure is approximate.\n\nI should check if there might be any other data points the user might need but the query seems straightforward. Maybe they\'re looking for a simple data extraction for another purpose, like integration into an app or a report. So providing clear and concise information in JSON should meet their needs.\n\nJust to be thorough, I\'ll outline each field: administrativeCapital, region, population, zip, capitalsDates, and WHYbecameCapital. Each of these should cover the essential aspects the user is asking for.\n\nI think that\'s all. Now, I can format it into JSON and present it back to the user.\n</think>\n\nHere is the information and population of the capital of France, Paris, in JSON format:\n\n```json\n{\n  "administrativeCapital": "Paris",\n  "region": "Île-de-France",\n  "population": "Approximately 2.1 million (most recent estimate)",\n  "zip": "75001 (and surrounding postal codes)",\n  "capitalsDates": " established as the administrative capital on July 29, 1790",\n  "WHYbecameCapital": "Paris was chosen as the capital after a series of reforms during the 18th century, culminating in its ascension as the administrative center following the defeat of the French highlights in the Napoleonic Wars in 1815"\n}\n```\n\nLet me know if you need further details!', 'output_ids': [71486, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 9625, 594, 6722, 11, 892, 374, 12095, 13, 2379, 11689, 1366, 432, 304, 4718, 3561, 13, 88190, 11, 358, 1265, 1156, 1281, 2704, 358, 614, 279, 4396, 3565, 911, 12095, 13, 4710, 40, 1414, 429, 12095, 374, 279, 22707, 6722, 11, 714, 7025, 1251, 58117, 432, 448, 279, 4948, 6722, 13, 576, 3946, 829, 374, 12095, 4311, 11, 1290, 30, 1597, 279, 5537, 374, 59108, 273, 6810, 7276, 34106, 13, 358, 1265, 2924, 429, 1995, 2238, 1576, 432, 594, 9760, 382, 7039, 11, 7042, 10472, 13, 358, 1744, 279, 7042, 374, 13187, 220, 17, 13, 16, 3526, 11, 714, 429, 594, 458, 5461, 504, 3213, 1635, 13, 1084, 646, 38288, 6292, 264, 2699, 11649, 389, 5257, 9363, 1075, 18339, 13, 2055, 44291, 429, 432, 594, 458, 44868, 7071, 3643, 5530, 382, 2461, 279, 56674, 2038, 11, 12095, 374, 8865, 700, 11, 773, 432, 3171, 944, 614, 264, 3175, 56674, 2038, 13, 12090, 11, 432, 702, 5248, 39754, 13912, 11, 773, 358, 1265, 1140, 279, 1429, 16626, 1483, 6174, 13, 220, 22, 20, 15, 15, 16, 374, 279, 8622, 72836, 11, 323, 1052, 525, 3800, 2163, 432, 13, 55121, 429, 11367, 79314, 311, 279, 2033, 382, 785, 2400, 979, 12095, 6116, 279, 6722, 374, 2989, 2238, 13, 1084, 572, 6635, 304, 220, 16, 23, 339, 33357, 30243, 11, 323, 358, 1744, 432, 572, 279, 1102, 315, 279, 18019, 315, 279, 8585, 21314, 2337, 279, 28476, 1263, 14011, 14968, 13, 2938, 2266, 2578, 1492, 279, 1196, 3535, 3170, 12095, 572, 11882, 438, 279, 6722, 382, 97904, 432, 678, 3786, 11, 358, 1265, 5944, 432, 304, 264, 4718, 3561, 438, 279, 1196, 11223, 13, 358, 3278, 990, 279, 584, 659, 2236, 368, 1714, 311, 633, 13382, 821, 369, 279, 7042, 16045, 13, 6771, 752, 1083, 19091, 429, 279, 5535, 821, 374, 2163, 220, 17, 15, 17, 18, 11, 773, 358, 1265, 6286, 429, 279, 7071, 374, 44868, 382, 40, 1265, 1779, 421, 1052, 2578, 387, 894, 1008, 821, 3501, 279, 1196, 2578, 1184, 714, 279, 3239, 4977, 30339, 13, 10696, 807, 2299, 3330, 369, 264, 4285, 821, 32189, 369, 2441, 7428, 11, 1075, 17590, 1119, 458, 906, 476, 264, 1895, 13, 2055, 8241, 2797, 323, 63594, 1995, 304, 4718, 1265, 3367, 862, 3880, 382, 9952, 311, 387, 17423, 11, 358, 3278, 21101, 1817, 2070, 25, 22707, 63593, 11, 5537, 11, 7042, 11, 10308, 11, 92999, 55238, 11, 323, 81694, 16692, 373, 63593, 13, 8886, 315, 1493, 1265, 3421, 279, 7565, 13566, 279, 1196, 374, 10161, 369, 382, 40, 1744, 429, 594, 678, 13, 4695, 11, 358, 646, 3561, 432, 1119, 4718, 323, 3042, 432, 1182, 311, 279, 1196, 624, 151649, 271, 8420, 374, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 11, 12095, 11, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 68849, 1388, 63593, 788, 330, 59604, 756, 220, 330, 3943, 788, 330, 71807, 273, 6810, 7276, 34106, 756, 220, 330, 44441, 788, 330, 69520, 7108, 220, 17, 13, 16, 3526, 320, 3562, 3213, 16045, 15752, 220, 330, 9964, 788, 330, 22, 20, 15, 15, 16, 320, 437, 14590, 39754, 13912, 15752, 220, 330, 11346, 16265, 55238, 788, 330, 9555, 438, 279, 22707, 6722, 389, 5768, 220, 17, 24, 11, 220, 16, 22, 24, 15, 756, 220, 330, 19890, 56, 16692, 373, 63593, 788, 330, 59604, 572, 11882, 438, 279, 6722, 1283, 264, 4013, 315, 30243, 2337, 279, 220, 16, 23, 339, 9294, 11, 11695, 64283, 304, 1181, 14601, 2645, 438, 279, 22707, 4126, 2701, 279, 18019, 315, 279, 8585, 21314, 304, 279, 28476, 1263, 14011, 14968, 304, 220, 16, 23, 16, 20, 698, 532, 13874, 19324, 10061, 752, 1414, 421, 498, 1184, 4623, 3565, 0, 151643], 'meta_info': {'id': 'db1a68dfd65446fda9c4afc86f22234c', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 450, 'completion_tokens': 621, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.569903088035062, 'response_sent_to_client_ts': 1775719101.7451465}}</strong>



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


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.22s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.55s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.50s/it]


    2026-04-09 07:18:39,462 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 07:18:39] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:50,  2.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:50,  2.99s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:31,  1.63s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:31,  1.63s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:27,  1.88it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:27,  1.88it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:23,  2.20it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:23,  2.20it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:19,  2.57it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:19,  2.57it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:16,  2.93it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:16,  2.93it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:14,  3.41it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:14,  3.41it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:12,  3.77it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:12,  3.77it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:10,  4.36it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:10,  4.36it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:09,  4.68it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:09,  4.68it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:09,  4.79it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:09,  4.79it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:08,  4.82it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:08,  4.82it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:08,  5.03it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:08,  5.03it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:07,  5.53it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:07,  5.53it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:07<00:06,  5.75it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:07<00:06,  5.75it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:07<00:06,  6.04it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:07<00:06,  6.04it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:07<00:06,  6.04it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:07<00:04,  8.85it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:07<00:04,  8.85it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:07<00:04,  8.85it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:03, 10.58it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:03, 10.58it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:03, 10.58it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:07<00:03, 10.58it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:07<00:03, 10.58it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:07<00:01, 16.65it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:07<00:01, 16.65it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:08<00:01, 16.65it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:08<00:01, 16.65it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:08<00:01, 19.25it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:08<00:01, 19.25it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:08<00:01, 19.25it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:08<00:01, 19.25it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:08<00:01, 20.96it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:08<00:01, 20.96it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:08<00:01, 20.96it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:08<00:01, 20.96it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:08<00:00, 22.36it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:08<00:00, 22.36it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:08<00:00, 22.36it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:08<00:00, 22.36it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:08<00:00, 22.36it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:08<00:00, 22.36it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:08<00:00, 27.15it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:08<00:00, 27.15it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:08<00:00, 27.15it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:08<00:00, 27.15it/s]

    Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:08<00:00, 27.15it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:08<00:00, 29.43it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:08<00:00, 29.43it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:08<00:00, 29.43it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:08<00:00, 29.43it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:08<00:00, 29.17it/s]

    Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:08<00:00, 29.17it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:08<00:00, 30.98it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:08<00:00, 30.98it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:08<00:00, 30.98it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:08<00:00, 30.98it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:08<00:00, 30.98it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:08<00:00, 33.38it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:08<00:00, 33.38it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:08<00:00, 33.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=39.55 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=39.55 GB):   2%|▏         | 1/58 [00:00<00:24,  2.29it/s]Capturing num tokens (num_tokens=7680 avail_mem=39.61 GB):   2%|▏         | 1/58 [00:00<00:24,  2.29it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=39.61 GB):   3%|▎         | 2/58 [00:00<00:22,  2.50it/s]Capturing num tokens (num_tokens=7168 avail_mem=39.60 GB):   3%|▎         | 2/58 [00:00<00:22,  2.50it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=39.60 GB):   5%|▌         | 3/58 [00:01<00:21,  2.59it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.08 GB):   5%|▌         | 3/58 [00:01<00:21,  2.59it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=44.08 GB):   7%|▋         | 4/58 [00:01<00:18,  2.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.07 GB):   7%|▋         | 4/58 [00:01<00:18,  2.90it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.07 GB):   9%|▊         | 5/58 [00:01<00:16,  3.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.06 GB):   9%|▊         | 5/58 [00:01<00:16,  3.16it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=44.06 GB):  10%|█         | 6/58 [00:01<00:14,  3.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.05 GB):  10%|█         | 6/58 [00:01<00:14,  3.48it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=44.05 GB):  12%|█▏        | 7/58 [00:02<00:13,  3.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.07 GB):  12%|█▏        | 7/58 [00:02<00:13,  3.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.07 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.18it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.06 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.18it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=44.06 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.06 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.06 GB):  17%|█▋        | 10/58 [00:02<00:09,  4.96it/s]Capturing num tokens (num_tokens=3584 avail_mem=44.02 GB):  17%|█▋        | 10/58 [00:02<00:09,  4.96it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=44.02 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.02 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.33it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=44.02 GB):  21%|██        | 12/58 [00:03<00:09,  4.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=44.03 GB):  21%|██        | 12/58 [00:03<00:09,  4.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=44.03 GB):  22%|██▏       | 13/58 [00:03<00:08,  5.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=44.02 GB):  22%|██▏       | 13/58 [00:03<00:08,  5.30it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=44.02 GB):  22%|██▏       | 13/58 [00:03<00:08,  5.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.02 GB):  26%|██▌       | 15/58 [00:03<00:06,  7.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.02 GB):  26%|██▌       | 15/58 [00:03<00:06,  7.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.02 GB):  26%|██▌       | 15/58 [00:03<00:06,  7.13it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=44.02 GB):  29%|██▉       | 17/58 [00:03<00:04,  8.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.02 GB):  29%|██▉       | 17/58 [00:03<00:04,  8.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.02 GB):  29%|██▉       | 17/58 [00:03<00:04,  8.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.02 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.02 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.02 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.89it/s]Capturing num tokens (num_tokens=960 avail_mem=44.01 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.89it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=44.01 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.40it/s]Capturing num tokens (num_tokens=896 avail_mem=44.01 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.40it/s]Capturing num tokens (num_tokens=832 avail_mem=44.01 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.40it/s]Capturing num tokens (num_tokens=768 avail_mem=44.00 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.40it/s]Capturing num tokens (num_tokens=768 avail_mem=44.00 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Capturing num tokens (num_tokens=704 avail_mem=44.00 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Capturing num tokens (num_tokens=640 avail_mem=44.00 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Capturing num tokens (num_tokens=576 avail_mem=43.99 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]

    Capturing num tokens (num_tokens=512 avail_mem=43.99 GB):  43%|████▎     | 25/58 [00:04<00:01, 17.76it/s]Capturing num tokens (num_tokens=512 avail_mem=43.99 GB):  50%|█████     | 29/58 [00:04<00:01, 21.80it/s]Capturing num tokens (num_tokens=480 avail_mem=43.99 GB):  50%|█████     | 29/58 [00:04<00:01, 21.80it/s]Capturing num tokens (num_tokens=448 avail_mem=43.98 GB):  50%|█████     | 29/58 [00:04<00:01, 21.80it/s]Capturing num tokens (num_tokens=416 avail_mem=43.98 GB):  50%|█████     | 29/58 [00:04<00:01, 21.80it/s]Capturing num tokens (num_tokens=384 avail_mem=43.98 GB):  50%|█████     | 29/58 [00:04<00:01, 21.80it/s]Capturing num tokens (num_tokens=384 avail_mem=43.98 GB):  57%|█████▋    | 33/58 [00:04<00:00, 25.31it/s]Capturing num tokens (num_tokens=352 avail_mem=43.97 GB):  57%|█████▋    | 33/58 [00:04<00:00, 25.31it/s]Capturing num tokens (num_tokens=320 avail_mem=43.97 GB):  57%|█████▋    | 33/58 [00:04<00:00, 25.31it/s]Capturing num tokens (num_tokens=288 avail_mem=43.96 GB):  57%|█████▋    | 33/58 [00:04<00:00, 25.31it/s]

    Capturing num tokens (num_tokens=256 avail_mem=43.96 GB):  57%|█████▋    | 33/58 [00:04<00:00, 25.31it/s]Capturing num tokens (num_tokens=256 avail_mem=43.96 GB):  64%|██████▍   | 37/58 [00:04<00:00, 28.50it/s]Capturing num tokens (num_tokens=240 avail_mem=43.95 GB):  64%|██████▍   | 37/58 [00:04<00:00, 28.50it/s]Capturing num tokens (num_tokens=224 avail_mem=43.95 GB):  64%|██████▍   | 37/58 [00:04<00:00, 28.50it/s]Capturing num tokens (num_tokens=208 avail_mem=43.95 GB):  64%|██████▍   | 37/58 [00:04<00:00, 28.50it/s]Capturing num tokens (num_tokens=192 avail_mem=43.94 GB):  64%|██████▍   | 37/58 [00:04<00:00, 28.50it/s]Capturing num tokens (num_tokens=192 avail_mem=43.94 GB):  71%|███████   | 41/58 [00:04<00:00, 30.01it/s]Capturing num tokens (num_tokens=176 avail_mem=43.94 GB):  71%|███████   | 41/58 [00:04<00:00, 30.01it/s]Capturing num tokens (num_tokens=160 avail_mem=43.94 GB):  71%|███████   | 41/58 [00:04<00:00, 30.01it/s]Capturing num tokens (num_tokens=144 avail_mem=43.93 GB):  71%|███████   | 41/58 [00:04<00:00, 30.01it/s]

    Capturing num tokens (num_tokens=128 avail_mem=43.94 GB):  71%|███████   | 41/58 [00:04<00:00, 30.01it/s]Capturing num tokens (num_tokens=112 avail_mem=43.94 GB):  71%|███████   | 41/58 [00:04<00:00, 30.01it/s]Capturing num tokens (num_tokens=112 avail_mem=43.94 GB):  79%|███████▉  | 46/58 [00:04<00:00, 32.19it/s]Capturing num tokens (num_tokens=96 avail_mem=43.93 GB):  79%|███████▉  | 46/58 [00:04<00:00, 32.19it/s] Capturing num tokens (num_tokens=80 avail_mem=43.93 GB):  79%|███████▉  | 46/58 [00:04<00:00, 32.19it/s]Capturing num tokens (num_tokens=64 avail_mem=43.93 GB):  79%|███████▉  | 46/58 [00:04<00:00, 32.19it/s]

    Capturing num tokens (num_tokens=48 avail_mem=43.92 GB):  79%|███████▉  | 46/58 [00:04<00:00, 32.19it/s]Capturing num tokens (num_tokens=48 avail_mem=43.92 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.96it/s]Capturing num tokens (num_tokens=32 avail_mem=43.92 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.96it/s]Capturing num tokens (num_tokens=28 avail_mem=43.92 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.96it/s]Capturing num tokens (num_tokens=24 avail_mem=43.91 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.96it/s]

    Capturing num tokens (num_tokens=24 avail_mem=43.91 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.55it/s]Capturing num tokens (num_tokens=20 avail_mem=43.91 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.55it/s]Capturing num tokens (num_tokens=16 avail_mem=43.91 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.55it/s]Capturing num tokens (num_tokens=12 avail_mem=43.90 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.55it/s]Capturing num tokens (num_tokens=8 avail_mem=43.90 GB):  91%|█████████▏| 53/58 [00:05<00:00, 22.55it/s] Capturing num tokens (num_tokens=8 avail_mem=43.90 GB):  98%|█████████▊| 57/58 [00:05<00:00, 25.95it/s]Capturing num tokens (num_tokens=4 avail_mem=43.90 GB):  98%|█████████▊| 57/58 [00:05<00:00, 25.95it/s]Capturing num tokens (num_tokens=4 avail_mem=43.90 GB): 100%|██████████| 58/58 [00:05<00:00, 11.47it/s]


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
    
    Generated text: Okay, so I need to figure out the population of the capital of France and present it in JSON format. Let me start by identifying what the capital of France is. I remember from school that Paris is the capital of France. 
    
    Now, I need to find out the current population of Paris. I'm not exactly sure of the exact number, but I think it's a large city. Maybe around 2 million? Wait, I think it's more than that. I recall hearing that major European cities like Paris, Berlin, and Madrid have populations over 3 million. 
    
    Let me try to recall or estimate. I believe Paris has a population of around 2.2 million. But I'm not 100% certain. Maybe I should think about other sources or recent data. If I remember correctly, the population figures can vary slightly depending on the source and the year. 
    
    Assuming that Paris has approximately 2.2 million people, I should format this information into a JSON structure. JSON typically uses key-value pairs, so I'll need a key for the capital and a key for the population. 
    
    So, the JSON structure would look something like this:
    {
      "capital": "Paris",
      "population": 2200000
    }
    
    Wait, but I should double-check the population. Maybe it's higher now. I think it's around 2.17 million as of recent estimates. But for simplicity, 2.2 million is a good approximation. 
    
    I also need to make sure the syntax is correct. JSON requires that keys and string values are in quotes, while numbers don't need quotes. So, "capital" is a string, and "population" is a number. 
    
    Putting it all together, the JSON should clearly state that the capital is Paris and its population is approximately 2.2 million. 
    
    I don't think I need to include units or anything else unless specified. So, the final JSON should be concise and accurate.
    </think>
    
    ```json
    {
      "capital": "Paris",
      "population": 2200000
    }
    ```



```python
llm.shutdown()
```
