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


    [2026-04-12 14:19:22] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-12 14:19:22] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-12 14:19:22] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-12 14:19:22] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.21s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.55s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.50s/it]


    2026-04-12 14:19:26,684 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-12 14:19:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:47,  2.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:47,  2.94s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:36,  1.73s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:36,  1.73s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:43,  1.24it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:43,  1.24it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:25,  2.05it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:25,  2.05it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:20,  2.52it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:20,  2.52it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:16,  3.04it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:16,  3.04it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:13,  3.61it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:13,  3.61it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:11,  4.17it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:11,  4.17it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:09,  4.78it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:09,  4.78it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:08,  5.51it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:08,  5.51it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:06<00:08,  5.51it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:06,  7.30it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:06,  7.30it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:06<00:06,  7.30it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:04,  9.12it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:04,  9.12it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:06<00:04,  9.12it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:06<00:03, 10.38it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:06<00:03, 10.38it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:06<00:03, 10.38it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:04,  8.74it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:04,  8.74it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:06<00:04,  8.67it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:06<00:04,  8.67it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:06<00:04,  8.67it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:03,  9.22it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:03,  9.22it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:03,  9.22it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:07<00:03, 10.02it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:07<00:03, 10.02it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:07<00:03, 10.02it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:07<00:02, 11.06it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:07<00:02, 11.06it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:07<00:02, 11.06it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:02, 11.95it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:02, 11.95it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:02, 11.95it/s]

    Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:02, 11.95it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:07<00:01, 14.14it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:07<00:01, 14.14it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:07<00:01, 14.14it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:07<00:01, 14.14it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:01, 16.91it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:01, 16.91it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:01, 16.91it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:01, 16.91it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:01, 16.91it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:07<00:00, 21.93it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:07<00:00, 21.93it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:07<00:00, 21.93it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:07<00:00, 21.93it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:07<00:00, 21.93it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:08<00:00, 26.16it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:08<00:00, 26.16it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:08<00:00, 26.16it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:08<00:00, 26.16it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:08<00:00, 24.63it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:08<00:00, 24.63it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:08<00:00, 24.63it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:08<00:00, 24.63it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 25.83it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 25.83it/s]

    Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 25.83it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 25.83it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 25.83it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:08<00:00, 29.01it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:08<00:00, 29.01it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:08<00:00, 29.01it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:08<00:00, 29.01it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:08<00:00, 29.01it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:08<00:00, 29.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00, 34.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.42 GB):   2%|▏         | 1/58 [00:00<00:34,  1.67it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.12 GB):   2%|▏         | 1/58 [00:00<00:34,  1.67it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.12 GB):   3%|▎         | 2/58 [00:01<00:30,  1.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.60 GB):   3%|▎         | 2/58 [00:01<00:30,  1.85it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.60 GB):   5%|▌         | 3/58 [00:01<00:28,  1.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.67 GB):   5%|▌         | 3/58 [00:01<00:28,  1.91it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.67 GB):   7%|▋         | 4/58 [00:02<00:26,  2.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.14 GB):   7%|▋         | 4/58 [00:02<00:26,  2.08it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.14 GB):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.78 GB):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.78 GB):  10%|█         | 6/58 [00:02<00:20,  2.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.15 GB):  10%|█         | 6/58 [00:02<00:20,  2.54it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=44.15 GB):  12%|█▏        | 7/58 [00:02<00:18,  2.81it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.88 GB):  12%|█▏        | 7/58 [00:02<00:18,  2.81it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.88 GB):  14%|█▍        | 8/58 [00:03<00:15,  3.18it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.15 GB):  14%|█▍        | 8/58 [00:03<00:15,  3.18it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=44.15 GB):  16%|█▌        | 9/58 [00:03<00:13,  3.51it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.15 GB):  16%|█▌        | 9/58 [00:03<00:13,  3.51it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.15 GB):  17%|█▋        | 10/58 [00:03<00:12,  3.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=44.02 GB):  17%|█▋        | 10/58 [00:03<00:12,  3.93it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=44.02 GB):  19%|█▉        | 11/58 [00:03<00:10,  4.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.14 GB):  19%|█▉        | 11/58 [00:03<00:10,  4.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.14 GB):  21%|██        | 12/58 [00:03<00:09,  4.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=44.13 GB):  21%|██        | 12/58 [00:03<00:09,  4.88it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=44.13 GB):  22%|██▏       | 13/58 [00:04<00:08,  5.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=44.04 GB):  22%|██▏       | 13/58 [00:04<00:08,  5.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=44.04 GB):  24%|██▍       | 14/58 [00:04<00:07,  5.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.12 GB):  24%|██▍       | 14/58 [00:04<00:07,  5.97it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=44.12 GB):  26%|██▌       | 15/58 [00:04<00:06,  6.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.11 GB):  26%|██▌       | 15/58 [00:04<00:06,  6.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.11 GB):  28%|██▊       | 16/58 [00:04<00:05,  7.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.11 GB):  28%|██▊       | 16/58 [00:04<00:05,  7.33it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=44.11 GB):  29%|██▉       | 17/58 [00:04<00:05,  7.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.11 GB):  29%|██▉       | 17/58 [00:04<00:05,  7.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.09 GB):  29%|██▉       | 17/58 [00:04<00:05,  7.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.09 GB):  33%|███▎      | 19/58 [00:04<00:04,  9.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.07 GB):  33%|███▎      | 19/58 [00:04<00:04,  9.55it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=44.08 GB):  33%|███▎      | 19/58 [00:04<00:04,  9.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.08 GB):  36%|███▌      | 21/58 [00:04<00:03, 11.38it/s]Capturing num tokens (num_tokens=960 avail_mem=44.08 GB):  36%|███▌      | 21/58 [00:04<00:03, 11.38it/s] Capturing num tokens (num_tokens=896 avail_mem=44.07 GB):  36%|███▌      | 21/58 [00:04<00:03, 11.38it/s]Capturing num tokens (num_tokens=896 avail_mem=44.07 GB):  40%|███▉      | 23/58 [00:04<00:02, 13.14it/s]Capturing num tokens (num_tokens=832 avail_mem=44.06 GB):  40%|███▉      | 23/58 [00:04<00:02, 13.14it/s]

    Capturing num tokens (num_tokens=768 avail_mem=44.03 GB):  40%|███▉      | 23/58 [00:04<00:02, 13.14it/s]Capturing num tokens (num_tokens=768 avail_mem=44.03 GB):  43%|████▎     | 25/58 [00:05<00:02, 13.80it/s]Capturing num tokens (num_tokens=704 avail_mem=44.05 GB):  43%|████▎     | 25/58 [00:05<00:02, 13.80it/s]Capturing num tokens (num_tokens=640 avail_mem=44.05 GB):  43%|████▎     | 25/58 [00:05<00:02, 13.80it/s]Capturing num tokens (num_tokens=576 avail_mem=44.04 GB):  43%|████▎     | 25/58 [00:05<00:02, 13.80it/s]Capturing num tokens (num_tokens=576 avail_mem=44.04 GB):  48%|████▊     | 28/58 [00:05<00:01, 16.39it/s]Capturing num tokens (num_tokens=512 avail_mem=44.01 GB):  48%|████▊     | 28/58 [00:05<00:01, 16.39it/s]

    Capturing num tokens (num_tokens=480 avail_mem=44.03 GB):  48%|████▊     | 28/58 [00:05<00:01, 16.39it/s]Capturing num tokens (num_tokens=448 avail_mem=44.03 GB):  48%|████▊     | 28/58 [00:05<00:01, 16.39it/s]Capturing num tokens (num_tokens=448 avail_mem=44.03 GB):  53%|█████▎    | 31/58 [00:05<00:01, 19.33it/s]Capturing num tokens (num_tokens=416 avail_mem=44.02 GB):  53%|█████▎    | 31/58 [00:05<00:01, 19.33it/s]Capturing num tokens (num_tokens=384 avail_mem=44.01 GB):  53%|█████▎    | 31/58 [00:05<00:01, 19.33it/s]Capturing num tokens (num_tokens=352 avail_mem=44.00 GB):  53%|█████▎    | 31/58 [00:05<00:01, 19.33it/s]Capturing num tokens (num_tokens=352 avail_mem=44.00 GB):  59%|█████▊    | 34/58 [00:05<00:01, 20.99it/s]Capturing num tokens (num_tokens=320 avail_mem=43.99 GB):  59%|█████▊    | 34/58 [00:05<00:01, 20.99it/s]

    Capturing num tokens (num_tokens=288 avail_mem=43.99 GB):  59%|█████▊    | 34/58 [00:05<00:01, 20.99it/s]Capturing num tokens (num_tokens=256 avail_mem=43.98 GB):  59%|█████▊    | 34/58 [00:05<00:01, 20.99it/s]Capturing num tokens (num_tokens=256 avail_mem=43.98 GB):  64%|██████▍   | 37/58 [00:05<00:00, 22.61it/s]Capturing num tokens (num_tokens=240 avail_mem=43.97 GB):  64%|██████▍   | 37/58 [00:05<00:00, 22.61it/s]Capturing num tokens (num_tokens=224 avail_mem=43.97 GB):  64%|██████▍   | 37/58 [00:05<00:00, 22.61it/s]Capturing num tokens (num_tokens=208 avail_mem=43.96 GB):  64%|██████▍   | 37/58 [00:05<00:00, 22.61it/s]Capturing num tokens (num_tokens=208 avail_mem=43.96 GB):  69%|██████▉   | 40/58 [00:05<00:00, 23.92it/s]Capturing num tokens (num_tokens=192 avail_mem=43.95 GB):  69%|██████▉   | 40/58 [00:05<00:00, 23.92it/s]

    Capturing num tokens (num_tokens=176 avail_mem=43.95 GB):  69%|██████▉   | 40/58 [00:05<00:00, 23.92it/s]Capturing num tokens (num_tokens=160 avail_mem=43.94 GB):  69%|██████▉   | 40/58 [00:05<00:00, 23.92it/s]Capturing num tokens (num_tokens=144 avail_mem=43.93 GB):  69%|██████▉   | 40/58 [00:05<00:00, 23.92it/s]Capturing num tokens (num_tokens=144 avail_mem=43.93 GB):  76%|███████▌  | 44/58 [00:05<00:00, 26.75it/s]Capturing num tokens (num_tokens=128 avail_mem=43.94 GB):  76%|███████▌  | 44/58 [00:05<00:00, 26.75it/s]Capturing num tokens (num_tokens=112 avail_mem=43.94 GB):  76%|███████▌  | 44/58 [00:05<00:00, 26.75it/s]Capturing num tokens (num_tokens=96 avail_mem=43.93 GB):  76%|███████▌  | 44/58 [00:05<00:00, 26.75it/s] Capturing num tokens (num_tokens=80 avail_mem=43.93 GB):  76%|███████▌  | 44/58 [00:05<00:00, 26.75it/s]Capturing num tokens (num_tokens=64 avail_mem=43.93 GB):  76%|███████▌  | 44/58 [00:05<00:00, 26.75it/s]Capturing num tokens (num_tokens=64 avail_mem=43.93 GB):  84%|████████▍ | 49/58 [00:05<00:00, 31.60it/s]Capturing num tokens (num_tokens=48 avail_mem=43.92 GB):  84%|████████▍ | 49/58 [00:05<00:00, 31.60it/s]

    Capturing num tokens (num_tokens=32 avail_mem=43.92 GB):  84%|████████▍ | 49/58 [00:05<00:00, 31.60it/s]Capturing num tokens (num_tokens=28 avail_mem=43.92 GB):  84%|████████▍ | 49/58 [00:05<00:00, 31.60it/s]Capturing num tokens (num_tokens=24 avail_mem=43.91 GB):  84%|████████▍ | 49/58 [00:05<00:00, 31.60it/s]Capturing num tokens (num_tokens=20 avail_mem=43.91 GB):  84%|████████▍ | 49/58 [00:05<00:00, 31.60it/s]Capturing num tokens (num_tokens=20 avail_mem=43.91 GB):  93%|█████████▎| 54/58 [00:05<00:00, 34.73it/s]Capturing num tokens (num_tokens=16 avail_mem=43.91 GB):  93%|█████████▎| 54/58 [00:05<00:00, 34.73it/s]Capturing num tokens (num_tokens=12 avail_mem=43.90 GB):  93%|█████████▎| 54/58 [00:06<00:00, 34.73it/s]Capturing num tokens (num_tokens=8 avail_mem=43.90 GB):  93%|█████████▎| 54/58 [00:06<00:00, 34.73it/s] Capturing num tokens (num_tokens=4 avail_mem=43.90 GB):  93%|█████████▎| 54/58 [00:06<00:00, 34.73it/s]

    Capturing num tokens (num_tokens=4 avail_mem=43.90 GB): 100%|██████████| 58/58 [00:06<00:00, 35.10it/s]Capturing num tokens (num_tokens=4 avail_mem=43.90 GB): 100%|██████████| 58/58 [00:06<00:00,  9.54it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is in New York and wants the current date and time along with the weather. I need to figure out how to get this using the provided functions.<br><br>First, I should use the get_current_date function. It requires a timezone parameter. Since the user is in New York, I'll set the timezone to 'America/New_York'. That should give me the correct date and time.<br><br>Next, for the weather, I'll use get_current_weather. I need a city and state. The user is in New York, which is NY, so the state is 'NY'. The city is 'New York'. I should also specify the unit as 'fahrenheit' since the user didn't mention the desired unit, but Fahrenheit is commonly used in the US.<br><br>I should structure the messages correctly, calling each function with the appropriate parameters. I'll make sure each function call is on its own line with the correct start and end tags as per the instructions.<br><br>I also need to remember to include the sources in the response. So, I'll mention that the date and time are from get_current_date and the weather info from get_current_weather, citing the respective functions.<br><br>Putting it all together, I'll send two separate function calls, one for the date and time and another for the weather, each with their parameters included properly.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>  <br><br>Sources:  <br>- Date and time information from get_current_date function  <br>- Weather information from get_current_weather function</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not exactly sure of the exact number. I think it\'s around 2 million, but I should double-check that.\n\nWait, maybe I should look up the latest statistics to be accurate. I recall that in recent years, Paris has seen a steady increase due to urban development and immigration. Let me see... I think the population is approximately 2,170,000 as of 2023. That seems about right.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the city name, its population, and maybe the country it\'s in. So, the keys could be "city", "population", and "country".\n\nPutting it all together, the JSON should look something like this: {"city": "Paris", "population": 2170000, "country": "France"}. I should make sure the syntax is correct, with proper commas and quotation marks.\n\nI should also consider if the user might need more details, like the metropolitan area population or other statistics, but since they specifically asked for the capital, I\'ll stick to that. Maybe mention that the data is up to date as of 2023 to provide context.\n\nAlright, I think that\'s all. I\'ll present the JSON and offer further assistance in case they need more information.\n</think>{\n  "name": "Paris",\n  "population": 2170000\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 382, 14190, 11, 7196, 358, 1265, 1401, 705, 279, 5535, 13142, 311, 387, 13382, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 24020, 5263, 4152, 311, 15662, 4401, 323, 15093, 13, 6771, 752, 1490, 1112, 358, 1744, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 3283, 829, 11, 1181, 7042, 11, 323, 7196, 279, 3146, 432, 594, 304, 13, 2055, 11, 279, 6894, 1410, 387, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 11436, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 5212, 8926, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 11, 330, 11141, 788, 330, 49000, 1, 7810, 358, 1265, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 57406, 3082, 7042, 476, 1008, 13142, 11, 714, 2474, 807, 11689, 4588, 369, 279, 6722, 11, 358, 3278, 9214, 311, 429, 13, 10696, 6286, 429, 279, 821, 374, 705, 311, 2400, 438, 315, 220, 17, 15, 17, 18, 311, 3410, 2266, 382, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3042, 279, 4718, 323, 3010, 4623, 12994, 304, 1142, 807, 1184, 803, 1995, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 198, 92, 151643], 'meta_info': {'id': '97de342adfcb4fb1ba6fd2c503ee9701', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 369, 'completion_tokens': 392, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 2.822132061002776, 'response_sent_to_client_ts': 1776003609.698623}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see... Yes, according to the latest data, the population is approximately 2,174,300 as of 2023. That seems accurate.\n\nNext, I need to structure this information into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and "country". The city is Paris, the population is the number I found, and the country is France.\n\nI should make sure the JSON syntax is correct. Each key should be in quotes, and the values as well. Also, the entire structure should be enclosed in curly braces. I\'ll format it properly to ensure there are no syntax errors.\n\nPutting it all together, the JSON object will have the city, population, and country. I\'ll present this to the user, making sure it\'s clear and easy to understand. I don\'t think the user needs anything else, but if they have more questions, I can provide additional information.\n\nI should also consider if the user might need the population figure in a different format, like a string or an integer, but since they asked for JSON, I\'ll stick with the number as an integer. \n\nAlright, I think that\'s all. I\'ll format the JSON correctly and present it to the user.\n</think>{\n\n"name": "Paris",\n"population": 2174300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 1112, 7414, 11, 4092, 311, 279, 5535, 821, 11, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 13382, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 3263, 576, 3283, 374, 12095, 11, 279, 7042, 374, 279, 1372, 358, 1730, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 8886, 1376, 1265, 387, 304, 17194, 11, 323, 279, 2750, 438, 1632, 13, 7281, 11, 279, 4453, 5944, 1265, 387, 43810, 304, 68103, 59191, 13, 358, 3278, 3561, 432, 10277, 311, 5978, 1052, 525, 902, 19482, 5975, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 686, 614, 279, 3283, 11, 7042, 11, 323, 3146, 13, 358, 3278, 3042, 419, 311, 279, 1196, 11, 3259, 2704, 432, 594, 2797, 323, 4135, 311, 3535, 13, 358, 1513, 944, 1744, 279, 1196, 3880, 4113, 770, 11, 714, 421, 807, 614, 803, 4755, 11, 358, 646, 3410, 5107, 1995, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 279, 7042, 7071, 304, 264, 2155, 3561, 11, 1075, 264, 914, 476, 458, 7546, 11, 714, 2474, 807, 4588, 369, 4718, 11, 358, 3278, 9214, 448, 279, 1372, 438, 458, 7546, 13, 4710, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3561, 279, 4718, 12440, 323, 3042, 432, 311, 279, 1196, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '3b7ddfc3eac04298be6379375940aa9f', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 383, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 30.885475787799805, 'response_sent_to_client_ts': 1776003640.5942848}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '9988bc772400456788dcb8b47240659a', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.08557496592402458, 'response_sent_to_client_ts': 1776003640.712107}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '1014c38d3e3a432d8a03bf1f81fb9a41', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.08552356087602675, 'response_sent_to_client_ts': 1776003640.7121124}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'ddc743a5b51a499db882a39b83021657', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.08548442111350596, 'response_sent_to_client_ts': 1776003640.7121153}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '0b5b3818d2fb4754bea1143b2ff7823e', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 14.820977156981826, 'response_sent_to_client_ts': 1776003655.538972}}


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


<strong style='color: #00008B;'>{'text': 'Alright, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, which is pretty straightforward. But for the population, I\'m a bit unsure. I think it\'s a large city, larger than other capitals like New York or Tokyo, but I\'m not exactly sure how many people live there.\n\nLet me start by recalling some general knowledge. I remember that Paris is one of the most populous cities in the world, but how much exactly? I\'ve heard it\'s over 30 million people, but I\'m not certain. Maybe it\'s around 21 million? I think I\'ve seen numbers that say 21.6 million, so maybe I should go with that. Wait, could it be higher now? I know that population figures can change over time due to births, deaths, and migration.\n\nI should also consider the official sources. I think the government or the census would have the most accurate figures. Maybe I can find the latest data from recent years. However, since I can\'t access the internet right now, I\'ll have to go with my memory or any information I\'ve been given before.\n\nAnother thing to think about is that sometimes people refer to the metropolitan area of Paris as being much larger. I\'ve heard that the metro area is around 50 million, but the actual city proper is smaller. So if I\'m only considering the city limits, it\'s probably closer to 21 million. \n\nI also recall a myth that Paris is the third-largest city in the world, but I think that\'s incorrect. Actually, I believe Cali in Colombia and some others in Asia are larger. So sticking with the population figures for the city proper, 21.6 million seems right. \n\nI should also mention that city limits vs. metropolitan area is a key differentiator here. People sometimes get confused between the two, so clarifying that is probably a good idea. It might help anyone reading the information to understand the context better.\n\nPutting it all together, the capital of France is Paris, and its population is approximately 21.6 million. To present this in JSON format, I need to structure it neatly. I think the keys should be "country" and "capital", with their respective values being "France" and "Paris". Then, under "population", I can include both the approximate figure and some additional notes for clarity.\n\nI should make sure to format the JSON correctly, using proper syntax with quotation marks and commas. Also, I should avoid any markdown as per the instructions, keeping it plain text. \n\nDouble-checking my numbers, I think 21.6 million is a solid estimate for the city proper. If I were unsure, I might mention the metropolitan area population as well, but since the question just asks for the population, the city proper seems appropriate.\n\nIn summary, Paris is the capital, and its population is about 21.6 million, mainly living within the city limits. The JSON should reflect this information clearly and concisely.\n</think>\n\n```json\n{\n  "country": "France",\n  "capital": "Paris",\n  "population": {\n    "approximate": "21.6 million",\n    "notes": "Population as of recent estimates; excludes metropolitan area, which is larger."\n  }\n}\n```', 'output_ids': [71486, 11, 773, 358, 1184, 311, 7071, 700, 279, 6722, 315, 9625, 323, 1181, 7042, 13, 358, 1414, 429, 279, 6722, 315, 9625, 374, 12095, 11, 892, 374, 5020, 30339, 13, 1988, 369, 279, 7042, 11, 358, 2776, 264, 2699, 42903, 13, 358, 1744, 432, 594, 264, 3460, 3283, 11, 8131, 1091, 1008, 92999, 1075, 1532, 4261, 476, 26194, 11, 714, 358, 2776, 537, 6896, 2704, 1246, 1657, 1251, 3887, 1052, 382, 10061, 752, 1191, 553, 88646, 1045, 4586, 6540, 13, 358, 6099, 429, 12095, 374, 825, 315, 279, 1429, 94451, 9720, 304, 279, 1879, 11, 714, 1246, 1753, 6896, 30, 358, 3003, 6617, 432, 594, 916, 220, 18, 15, 3526, 1251, 11, 714, 358, 2776, 537, 3654, 13, 10696, 432, 594, 2163, 220, 17, 16, 3526, 30, 358, 1744, 358, 3003, 3884, 5109, 429, 1977, 220, 17, 16, 13, 21, 3526, 11, 773, 7196, 358, 1265, 728, 448, 429, 13, 13824, 11, 1410, 432, 387, 5080, 1431, 30, 358, 1414, 429, 7042, 12396, 646, 2297, 916, 882, 4152, 311, 65232, 11, 16375, 11, 323, 11906, 382, 40, 1265, 1083, 2908, 279, 3946, 8173, 13, 358, 1744, 279, 3033, 476, 279, 43602, 1035, 614, 279, 1429, 13382, 12396, 13, 10696, 358, 646, 1477, 279, 5535, 821, 504, 3213, 1635, 13, 4354, 11, 2474, 358, 646, 944, 2615, 279, 7602, 1290, 1431, 11, 358, 3278, 614, 311, 728, 448, 847, 4938, 476, 894, 1995, 358, 3003, 1012, 2661, 1573, 382, 14037, 3166, 311, 1744, 911, 374, 429, 7025, 1251, 8300, 311, 279, 57406, 3082, 315, 12095, 438, 1660, 1753, 8131, 13, 358, 3003, 6617, 429, 279, 33482, 3082, 374, 2163, 220, 20, 15, 3526, 11, 714, 279, 5042, 3283, 6169, 374, 9155, 13, 2055, 421, 358, 2776, 1172, 12831, 279, 3283, 13388, 11, 432, 594, 4658, 12128, 311, 220, 17, 16, 3526, 13, 4710, 40, 1083, 19091, 264, 21163, 429, 12095, 374, 279, 4843, 66967, 3283, 304, 279, 1879, 11, 714, 358, 1744, 429, 594, 15114, 13, 33763, 11, 358, 4411, 3319, 72, 304, 38033, 323, 1045, 3800, 304, 13622, 525, 8131, 13, 2055, 36972, 448, 279, 7042, 12396, 369, 279, 3283, 6169, 11, 220, 17, 16, 13, 21, 3526, 4977, 1290, 13, 4710, 40, 1265, 1083, 6286, 429, 3283, 13388, 6165, 13, 57406, 3082, 374, 264, 1376, 2155, 36122, 1588, 13, 8853, 7025, 633, 21815, 1948, 279, 1378, 11, 773, 19497, 7766, 429, 374, 4658, 264, 1661, 4522, 13, 1084, 2578, 1492, 5489, 5290, 279, 1995, 311, 3535, 279, 2266, 2664, 382, 97904, 432, 678, 3786, 11, 279, 6722, 315, 9625, 374, 12095, 11, 323, 1181, 7042, 374, 13187, 220, 17, 16, 13, 21, 3526, 13, 2014, 3042, 419, 304, 4718, 3561, 11, 358, 1184, 311, 5944, 432, 62166, 13, 358, 1744, 279, 6894, 1265, 387, 330, 11141, 1, 323, 330, 65063, 497, 448, 862, 19511, 2750, 1660, 330, 49000, 1, 323, 330, 59604, 3263, 5005, 11, 1212, 330, 44441, 497, 358, 646, 2924, 2176, 279, 44868, 7071, 323, 1045, 5107, 8388, 369, 31273, 382, 40, 1265, 1281, 2704, 311, 3561, 279, 4718, 12440, 11, 1667, 6169, 19482, 448, 54231, 15423, 323, 76602, 13, 7281, 11, 358, 1265, 5648, 894, 50494, 438, 817, 279, 11221, 11, 10282, 432, 14396, 1467, 13, 4710, 7378, 15934, 287, 847, 5109, 11, 358, 1744, 220, 17, 16, 13, 21, 3526, 374, 264, 6437, 16045, 369, 279, 3283, 6169, 13, 1416, 358, 1033, 42903, 11, 358, 2578, 6286, 279, 57406, 3082, 7042, 438, 1632, 11, 714, 2474, 279, 3405, 1101, 17064, 369, 279, 7042, 11, 279, 3283, 6169, 4977, 8311, 382, 641, 12126, 11, 12095, 374, 279, 6722, 11, 323, 1181, 7042, 374, 911, 220, 17, 16, 13, 21, 3526, 11, 14576, 5382, 2878, 279, 3283, 13388, 13, 576, 4718, 1265, 8708, 419, 1995, 9355, 323, 3529, 285, 974, 624, 151649, 271, 73594, 2236, 198, 515, 220, 330, 11141, 788, 330, 49000, 756, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 341, 262, 330, 48053, 3426, 788, 330, 17, 16, 13, 21, 3526, 756, 262, 330, 18286, 788, 330, 53371, 438, 315, 3213, 17530, 26, 63368, 57406, 3082, 11, 892, 374, 8131, 10040, 220, 456, 532, 73594, 151643], 'meta_info': {'id': '1a6501ac87d64921ace623789fb3fbd8', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 626, 'completion_tokens': 686, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.068231467157602, 'response_sent_to_client_ts': 1776003660.6156716}}</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.20s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.55s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.50s/it]


    2026-04-12 14:21:17,480 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-12 14:21:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:03,  3.23s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:03,  3.23s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:28,  1.59s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:28,  1.59s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:52,  1.05it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:52,  1.05it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:35,  1.54it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:35,  1.54it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:25,  2.12it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:25,  2.12it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.75it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.75it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:15,  3.40it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:15,  3.40it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:13,  3.59it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:13,  3.59it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:11,  4.21it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:11,  4.21it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:10,  4.49it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:10,  4.49it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:09,  4.84it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:09,  4.84it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:08,  5.24it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:08,  5.24it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:08,  5.24it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:06,  7.17it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:06,  7.17it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:06,  7.17it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04,  9.50it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04,  9.50it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04,  9.50it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:06<00:04,  9.50it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:02, 13.16it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:02, 13.16it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:02, 13.16it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:06<00:02, 13.16it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:06<00:02, 13.16it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:06<00:01, 18.70it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:06<00:01, 18.70it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:06<00:01, 18.70it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:06<00:01, 18.70it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:06<00:01, 18.70it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:06<00:01, 18.70it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:06<00:01, 25.64it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:06<00:01, 25.64it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:06<00:01, 25.64it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:06<00:01, 25.64it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:06<00:01, 25.64it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:06<00:01, 25.64it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:06<00:01, 25.64it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:06<00:00, 32.69it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:06<00:00, 32.69it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:06<00:00, 32.69it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:06<00:00, 32.69it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:06<00:00, 32.69it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:06<00:00, 32.69it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:06<00:00, 32.69it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:06<00:00, 38.86it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:06<00:00, 38.86it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:06<00:00, 38.86it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 38.86it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:06<00:00, 38.86it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:06<00:00, 38.86it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:06<00:00, 38.86it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 43.65it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 43.65it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 43.65it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 43.65it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:06<00:00, 43.65it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:06<00:00, 43.65it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:06<00:00, 43.65it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:06<00:00, 43.65it/s]

    Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 50.31it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 50.31it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 50.31it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:06<00:00, 50.31it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:06<00:00, 50.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=39.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=39.50 GB):   2%|▏         | 1/58 [00:00<00:17,  3.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=39.47 GB):   2%|▏         | 1/58 [00:00<00:17,  3.34it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=39.47 GB):   3%|▎         | 2/58 [00:00<00:16,  3.43it/s]Capturing num tokens (num_tokens=7168 avail_mem=39.45 GB):   3%|▎         | 2/58 [00:00<00:16,  3.43it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=39.45 GB):   5%|▌         | 3/58 [00:00<00:15,  3.61it/s]Capturing num tokens (num_tokens=6656 avail_mem=39.43 GB):   5%|▌         | 3/58 [00:00<00:15,  3.61it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=39.43 GB):   7%|▋         | 4/58 [00:01<00:14,  3.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=39.38 GB):   7%|▋         | 4/58 [00:01<00:14,  3.69it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=39.38 GB):   9%|▊         | 5/58 [00:01<00:13,  3.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=39.36 GB):   9%|▊         | 5/58 [00:01<00:13,  3.83it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=39.36 GB):  10%|█         | 6/58 [00:01<00:12,  4.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=39.36 GB):  10%|█         | 6/58 [00:01<00:12,  4.02it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=39.36 GB):  12%|█▏        | 7/58 [00:01<00:13,  3.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.86 GB):  12%|█▏        | 7/58 [00:01<00:13,  3.89it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=38.86 GB):  14%|█▍        | 8/58 [00:02<00:14,  3.43it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.87 GB):  14%|█▍        | 8/58 [00:02<00:14,  3.43it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.87 GB):  16%|█▌        | 9/58 [00:02<00:12,  4.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.87 GB):  16%|█▌        | 9/58 [00:02<00:12,  4.04it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=38.87 GB):  17%|█▋        | 10/58 [00:02<00:11,  4.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.12 GB):  17%|█▋        | 10/58 [00:02<00:11,  4.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.12 GB):  19%|█▉        | 11/58 [00:02<00:09,  4.94it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.12 GB):  19%|█▉        | 11/58 [00:02<00:09,  4.94it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=57.12 GB):  21%|██        | 12/58 [00:02<00:07,  5.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.12 GB):  21%|██        | 12/58 [00:02<00:07,  5.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.12 GB):  21%|██        | 12/58 [00:02<00:07,  5.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.12 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.12 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.40it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=57.12 GB):  24%|██▍       | 14/58 [00:03<00:05,  7.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.12 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.12 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.12 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.12 GB):  31%|███       | 18/58 [00:03<00:03, 10.80it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.12 GB):  31%|███       | 18/58 [00:03<00:03, 10.80it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=57.12 GB):  31%|███       | 18/58 [00:03<00:03, 10.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.12 GB):  31%|███       | 18/58 [00:03<00:03, 10.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.12 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.02it/s]Capturing num tokens (num_tokens=960 avail_mem=57.12 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.02it/s] Capturing num tokens (num_tokens=896 avail_mem=57.11 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.02it/s]Capturing num tokens (num_tokens=832 avail_mem=57.11 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.02it/s]Capturing num tokens (num_tokens=832 avail_mem=57.11 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.09it/s]Capturing num tokens (num_tokens=768 avail_mem=57.11 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.09it/s]

    Capturing num tokens (num_tokens=704 avail_mem=57.10 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.09it/s]Capturing num tokens (num_tokens=640 avail_mem=57.10 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.09it/s]Capturing num tokens (num_tokens=576 avail_mem=57.10 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.09it/s]Capturing num tokens (num_tokens=576 avail_mem=57.10 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.03it/s]Capturing num tokens (num_tokens=512 avail_mem=57.09 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.03it/s]Capturing num tokens (num_tokens=480 avail_mem=57.09 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.03it/s]Capturing num tokens (num_tokens=448 avail_mem=57.09 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.03it/s]

    Capturing num tokens (num_tokens=448 avail_mem=57.09 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.90it/s]Capturing num tokens (num_tokens=416 avail_mem=57.08 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.90it/s]Capturing num tokens (num_tokens=384 avail_mem=57.08 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.90it/s]Capturing num tokens (num_tokens=352 avail_mem=57.07 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.90it/s]Capturing num tokens (num_tokens=352 avail_mem=57.07 GB):  59%|█████▊    | 34/58 [00:03<00:01, 21.74it/s]Capturing num tokens (num_tokens=320 avail_mem=61.56 GB):  59%|█████▊    | 34/58 [00:03<00:01, 21.74it/s]Capturing num tokens (num_tokens=288 avail_mem=61.56 GB):  59%|█████▊    | 34/58 [00:03<00:01, 21.74it/s]

    Capturing num tokens (num_tokens=256 avail_mem=61.55 GB):  59%|█████▊    | 34/58 [00:03<00:01, 21.74it/s]Capturing num tokens (num_tokens=256 avail_mem=61.55 GB):  64%|██████▍   | 37/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=240 avail_mem=61.55 GB):  64%|██████▍   | 37/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=224 avail_mem=61.54 GB):  64%|██████▍   | 37/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=208 avail_mem=61.54 GB):  64%|██████▍   | 37/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=192 avail_mem=61.54 GB):  64%|██████▍   | 37/58 [00:04<00:00, 22.62it/s]Capturing num tokens (num_tokens=192 avail_mem=61.54 GB):  71%|███████   | 41/58 [00:04<00:00, 25.81it/s]Capturing num tokens (num_tokens=176 avail_mem=61.53 GB):  71%|███████   | 41/58 [00:04<00:00, 25.81it/s]Capturing num tokens (num_tokens=160 avail_mem=61.53 GB):  71%|███████   | 41/58 [00:04<00:00, 25.81it/s]

    Capturing num tokens (num_tokens=144 avail_mem=61.52 GB):  71%|███████   | 41/58 [00:04<00:00, 25.81it/s]Capturing num tokens (num_tokens=128 avail_mem=61.53 GB):  71%|███████   | 41/58 [00:04<00:00, 25.81it/s]Capturing num tokens (num_tokens=128 avail_mem=61.53 GB):  78%|███████▊  | 45/58 [00:04<00:00, 28.93it/s]Capturing num tokens (num_tokens=112 avail_mem=61.53 GB):  78%|███████▊  | 45/58 [00:04<00:00, 28.93it/s]Capturing num tokens (num_tokens=96 avail_mem=61.53 GB):  78%|███████▊  | 45/58 [00:04<00:00, 28.93it/s] Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  78%|███████▊  | 45/58 [00:04<00:00, 28.93it/s]Capturing num tokens (num_tokens=64 avail_mem=61.52 GB):  78%|███████▊  | 45/58 [00:04<00:00, 28.93it/s]Capturing num tokens (num_tokens=64 avail_mem=61.52 GB):  84%|████████▍ | 49/58 [00:04<00:00, 31.72it/s]Capturing num tokens (num_tokens=48 avail_mem=61.52 GB):  84%|████████▍ | 49/58 [00:04<00:00, 31.72it/s]Capturing num tokens (num_tokens=32 avail_mem=61.51 GB):  84%|████████▍ | 49/58 [00:04<00:00, 31.72it/s]

    Capturing num tokens (num_tokens=28 avail_mem=61.51 GB):  84%|████████▍ | 49/58 [00:04<00:00, 31.72it/s]Capturing num tokens (num_tokens=24 avail_mem=61.51 GB):  84%|████████▍ | 49/58 [00:04<00:00, 31.72it/s]Capturing num tokens (num_tokens=24 avail_mem=61.51 GB):  91%|█████████▏| 53/58 [00:04<00:00, 26.15it/s]Capturing num tokens (num_tokens=20 avail_mem=61.50 GB):  91%|█████████▏| 53/58 [00:04<00:00, 26.15it/s]Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  91%|█████████▏| 53/58 [00:04<00:00, 26.15it/s]

    Capturing num tokens (num_tokens=12 avail_mem=61.50 GB):  91%|█████████▏| 53/58 [00:04<00:00, 26.15it/s]Capturing num tokens (num_tokens=12 avail_mem=61.50 GB):  97%|█████████▋| 56/58 [00:04<00:00, 26.90it/s]Capturing num tokens (num_tokens=8 avail_mem=61.49 GB):  97%|█████████▋| 56/58 [00:04<00:00, 26.90it/s] Capturing num tokens (num_tokens=4 avail_mem=61.49 GB):  97%|█████████▋| 56/58 [00:04<00:00, 26.90it/s]Capturing num tokens (num_tokens=4 avail_mem=61.49 GB): 100%|██████████| 58/58 [00:04<00:00, 12.11it/s]


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
    Generated text: Rome is the capital of Italy
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
    Generated text: France
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
    
    Generated text: Okay, so I need to find the capital of France and provide its population in JSON format. Alright, I know that Paris is the capital of France, but I'm not exactly sure about the current population numbers. I remember reading somewhere that it's a big city with millions of people. Maybe around 20-something million? I think I should double-check that.
    
    Wait, I'm not a expert on population figures, so I should make sure I get this right. I believe the population figure is approximate, especially since it can change over time due to births, deaths, and migration. The figure I'm thinking of is roughly 21 million. But I'm not entirely certain if that's accurate for the latest data.
    
    So, I'll go with Paris as the capital and 21,624,000 as the population. Now, putting that into JSON format. I remember that JSON uses key-value pairs, so the keys would be "capital" and "population". The values would be the respective data points.
    
    Wait, should I include units? The population is a number, so maybe just the number is fine. I'll format it without commas for readability, so it would be 21624000 instead of 21,624,000. That's how it's represented in JSON, I think.
    
    Putting it all together, the JSON structure would have a single key-value pair where "capital" is "Paris" and "population" is 21624000. I'll make sure to format it correctly with proper syntax, using quotes around the keys and string values, and commas to separate the key-value pairs.
    
    I'm a bit concerned about the accuracy of the population number. Maybe I should mention that the figure is approximate. Oh, right, the user just asked for the information, so I don't need to overcomplicate it with explanations. Just providing the data accurately is key.
    
    Alright, I think I'm ready to present the JSON now. Let me double-check the syntax to avoid any errors. The keys should be in double quotes, and string values also in double quotes. The numbers don't need quotes. Yep, that looks correct.
    
    So, final check: "capital" is "Paris", "population" is 21624000. Yep, that should do it.
    </think>
    
    ```json
    {
      "capital": "Paris",
      "population": 21624000
    }
    ```



```python
llm.shutdown()
```
