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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-12 03:05:41] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.37s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.48s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.46s/it]


    2026-05-12 03:05:48,463 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 03:05:48] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:14,  5.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:14,  5.52s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:29,  2.66s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:29,  2.66s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:35,  1.74s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:35,  1.74s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:09,  1.29s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:09,  1.29s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:53,  1.01s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:53,  1.01s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:08<00:43,  1.19it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:08<00:43,  1.19it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:35,  1.42it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:35,  1.42it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:09<00:30,  1.65it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:09<00:30,  1.65it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:25,  1.91it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:25,  1.91it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:22,  2.18it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:22,  2.18it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:10<00:19,  2.42it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:10<00:19,  2.42it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:10<00:16,  2.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:10<00:16,  2.71it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:10<00:14,  3.00it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:10<00:14,  3.00it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:10<00:13,  3.31it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:10<00:13,  3.31it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:11<00:11,  3.67it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:11<00:11,  3.67it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:11<00:10,  4.08it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:11<00:10,  4.08it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:11<00:08,  4.57it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:11<00:08,  4.57it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:11<00:07,  5.07it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:11<00:07,  5.07it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:11<00:06,  5.64it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:11<00:06,  5.64it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:11<00:06,  6.25it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:11<00:06,  6.25it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:12<00:05,  6.98it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:12<00:05,  6.98it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:12<00:05,  6.98it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:12<00:03,  8.90it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:12<00:03,  8.90it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:12<00:03,  8.90it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:12<00:03, 10.42it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:12<00:03, 10.42it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:12<00:03, 10.42it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:12<00:02, 12.01it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:12<00:02, 12.01it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:12<00:02, 12.01it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:12<00:02, 13.81it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:12<00:02, 13.81it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:12<00:02, 13.81it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:12<00:02, 13.81it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:12<00:01, 16.19it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:12<00:01, 16.19it/s]

    Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:12<00:01, 16.19it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:12<00:01, 16.19it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:12<00:01, 18.57it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:12<00:01, 18.57it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:12<00:01, 18.57it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:12<00:01, 18.57it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:12<00:01, 18.57it/s]

    Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:12<00:00, 22.43it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:12<00:00, 22.43it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:12<00:00, 22.43it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:13<00:00, 22.43it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:13<00:00, 22.43it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:13<00:00, 25.02it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:13<00:00, 25.02it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:13<00:00, 25.02it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:13<00:00, 25.02it/s]

    Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:13<00:00, 25.02it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:13<00:00, 26.70it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:13<00:00, 26.70it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:13<00:00, 26.70it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:13<00:00, 26.70it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:13<00:00, 26.70it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:13<00:00, 28.87it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:13<00:00, 28.87it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:13<00:00, 28.87it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:13<00:00, 28.87it/s]

    Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:13<00:00, 28.87it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:13<00:00, 28.87it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:13<00:00, 28.87it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:13<00:00, 36.01it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:13<00:00, 36.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:13<00:00,  4.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.65 GB):   2%|▏         | 1/58 [00:00<00:52,  1.09it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.70 GB):   2%|▏         | 1/58 [00:00<00:52,  1.09it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.70 GB):   3%|▎         | 2/58 [00:01<00:48,  1.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   3%|▎         | 2/58 [00:01<00:48,  1.15it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   5%|▌         | 3/58 [00:02<00:46,  1.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.82 GB):   5%|▌         | 3/58 [00:02<00:46,  1.20it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.82 GB):   7%|▋         | 4/58 [00:03<00:41,  1.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.90 GB):   7%|▋         | 4/58 [00:03<00:41,  1.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.90 GB):   9%|▊         | 5/58 [00:03<00:38,  1.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=26.68 GB):   9%|▊         | 5/58 [00:03<00:38,  1.39it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=26.68 GB):  10%|█         | 6/58 [00:04<00:34,  1.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=26.68 GB):  10%|█         | 6/58 [00:04<00:34,  1.53it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=26.68 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=26.68 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.65it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=26.68 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.68 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.83it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.68 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.68 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.01it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=26.68 GB):  17%|█▋        | 10/58 [00:06<00:21,  2.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=26.67 GB):  17%|█▋        | 10/58 [00:06<00:21,  2.19it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=26.67 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=26.66 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.39it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=26.66 GB):  21%|██        | 12/58 [00:06<00:17,  2.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=26.66 GB):  21%|██        | 12/58 [00:06<00:17,  2.60it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=26.66 GB):  22%|██▏       | 13/58 [00:06<00:15,  2.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.32 GB):  22%|██▏       | 13/58 [00:06<00:15,  2.84it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.32 GB):  24%|██▍       | 14/58 [00:09<00:39,  1.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.02 GB):  24%|██▍       | 14/58 [00:09<00:39,  1.12it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.02 GB):  26%|██▌       | 15/58 [00:09<00:29,  1.44it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.08 GB):  26%|██▌       | 15/58 [00:09<00:29,  1.44it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=26.08 GB):  28%|██▊       | 16/58 [00:09<00:22,  1.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.14 GB):  28%|██▊       | 16/58 [00:09<00:22,  1.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.14 GB):  29%|██▉       | 17/58 [00:09<00:17,  2.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.40 GB):  29%|██▉       | 17/58 [00:09<00:17,  2.29it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=26.40 GB):  31%|███       | 18/58 [00:09<00:13,  2.86it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.40 GB):  31%|███       | 18/58 [00:09<00:13,  2.86it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.40 GB):  33%|███▎      | 19/58 [00:10<00:11,  3.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.42 GB):  33%|███▎      | 19/58 [00:10<00:11,  3.54it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=26.42 GB):  34%|███▍      | 20/58 [00:10<00:08,  4.34it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.57 GB):  34%|███▍      | 20/58 [00:10<00:08,  4.34it/s]Capturing num tokens (num_tokens=960 avail_mem=26.56 GB):  34%|███▍      | 20/58 [00:10<00:08,  4.34it/s] Capturing num tokens (num_tokens=960 avail_mem=26.56 GB):  38%|███▊      | 22/58 [00:10<00:06,  5.91it/s]Capturing num tokens (num_tokens=896 avail_mem=26.55 GB):  38%|███▊      | 22/58 [00:10<00:06,  5.91it/s]

    Capturing num tokens (num_tokens=832 avail_mem=26.55 GB):  38%|███▊      | 22/58 [00:10<00:06,  5.91it/s]Capturing num tokens (num_tokens=832 avail_mem=26.55 GB):  41%|████▏     | 24/58 [00:10<00:04,  7.30it/s]Capturing num tokens (num_tokens=768 avail_mem=26.53 GB):  41%|████▏     | 24/58 [00:10<00:04,  7.30it/s]Capturing num tokens (num_tokens=704 avail_mem=26.52 GB):  41%|████▏     | 24/58 [00:10<00:04,  7.30it/s]

    Capturing num tokens (num_tokens=704 avail_mem=26.52 GB):  45%|████▍     | 26/58 [00:10<00:03,  8.62it/s]Capturing num tokens (num_tokens=640 avail_mem=26.51 GB):  45%|████▍     | 26/58 [00:10<00:03,  8.62it/s]Capturing num tokens (num_tokens=576 avail_mem=26.50 GB):  45%|████▍     | 26/58 [00:10<00:03,  8.62it/s]Capturing num tokens (num_tokens=576 avail_mem=26.50 GB):  48%|████▊     | 28/58 [00:10<00:03,  9.97it/s]Capturing num tokens (num_tokens=512 avail_mem=26.48 GB):  48%|████▊     | 28/58 [00:10<00:03,  9.97it/s]

    Capturing num tokens (num_tokens=480 avail_mem=26.48 GB):  48%|████▊     | 28/58 [00:10<00:03,  9.97it/s]Capturing num tokens (num_tokens=480 avail_mem=26.48 GB):  52%|█████▏    | 30/58 [00:10<00:02, 11.18it/s]Capturing num tokens (num_tokens=448 avail_mem=26.47 GB):  52%|█████▏    | 30/58 [00:10<00:02, 11.18it/s]Capturing num tokens (num_tokens=416 avail_mem=26.45 GB):  52%|█████▏    | 30/58 [00:10<00:02, 11.18it/s]Capturing num tokens (num_tokens=416 avail_mem=26.45 GB):  55%|█████▌    | 32/58 [00:11<00:02, 12.27it/s]Capturing num tokens (num_tokens=384 avail_mem=26.44 GB):  55%|█████▌    | 32/58 [00:11<00:02, 12.27it/s]

    Capturing num tokens (num_tokens=352 avail_mem=26.44 GB):  55%|█████▌    | 32/58 [00:11<00:02, 12.27it/s]Capturing num tokens (num_tokens=352 avail_mem=26.44 GB):  59%|█████▊    | 34/58 [00:11<00:01, 13.52it/s]Capturing num tokens (num_tokens=320 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:11<00:01, 13.52it/s]Capturing num tokens (num_tokens=288 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:11<00:01, 13.52it/s]Capturing num tokens (num_tokens=288 avail_mem=26.43 GB):  62%|██████▏   | 36/58 [00:11<00:01, 14.83it/s]Capturing num tokens (num_tokens=256 avail_mem=26.42 GB):  62%|██████▏   | 36/58 [00:11<00:01, 14.83it/s]

    Capturing num tokens (num_tokens=240 avail_mem=26.41 GB):  62%|██████▏   | 36/58 [00:11<00:01, 14.83it/s]Capturing num tokens (num_tokens=240 avail_mem=26.41 GB):  66%|██████▌   | 38/58 [00:11<00:01, 16.05it/s]Capturing num tokens (num_tokens=224 avail_mem=26.40 GB):  66%|██████▌   | 38/58 [00:11<00:01, 16.05it/s]Capturing num tokens (num_tokens=208 avail_mem=26.38 GB):  66%|██████▌   | 38/58 [00:11<00:01, 16.05it/s]Capturing num tokens (num_tokens=208 avail_mem=26.38 GB):  69%|██████▉   | 40/58 [00:11<00:01, 16.83it/s]Capturing num tokens (num_tokens=192 avail_mem=26.37 GB):  69%|██████▉   | 40/58 [00:11<00:01, 16.83it/s]

    Capturing num tokens (num_tokens=176 avail_mem=26.36 GB):  69%|██████▉   | 40/58 [00:11<00:01, 16.83it/s]Capturing num tokens (num_tokens=160 avail_mem=26.36 GB):  69%|██████▉   | 40/58 [00:11<00:01, 16.83it/s]Capturing num tokens (num_tokens=160 avail_mem=26.36 GB):  74%|███████▍  | 43/58 [00:11<00:00, 18.11it/s]Capturing num tokens (num_tokens=144 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:11<00:00, 18.11it/s]Capturing num tokens (num_tokens=128 avail_mem=26.34 GB):  74%|███████▍  | 43/58 [00:11<00:00, 18.11it/s]Capturing num tokens (num_tokens=112 avail_mem=26.34 GB):  74%|███████▍  | 43/58 [00:11<00:00, 18.11it/s]

    Capturing num tokens (num_tokens=112 avail_mem=26.34 GB):  79%|███████▉  | 46/58 [00:11<00:00, 18.88it/s]Capturing num tokens (num_tokens=96 avail_mem=26.32 GB):  79%|███████▉  | 46/58 [00:11<00:00, 18.88it/s] Capturing num tokens (num_tokens=80 avail_mem=26.31 GB):  79%|███████▉  | 46/58 [00:11<00:00, 18.88it/s]Capturing num tokens (num_tokens=64 avail_mem=26.30 GB):  79%|███████▉  | 46/58 [00:11<00:00, 18.88it/s]Capturing num tokens (num_tokens=64 avail_mem=26.30 GB):  84%|████████▍ | 49/58 [00:11<00:00, 19.64it/s]Capturing num tokens (num_tokens=48 avail_mem=26.29 GB):  84%|████████▍ | 49/58 [00:11<00:00, 19.64it/s]Capturing num tokens (num_tokens=32 avail_mem=26.28 GB):  84%|████████▍ | 49/58 [00:11<00:00, 19.64it/s]

    Capturing num tokens (num_tokens=28 avail_mem=26.27 GB):  84%|████████▍ | 49/58 [00:12<00:00, 19.64it/s]Capturing num tokens (num_tokens=28 avail_mem=26.27 GB):  90%|████████▉ | 52/58 [00:12<00:00, 20.20it/s]Capturing num tokens (num_tokens=24 avail_mem=26.26 GB):  90%|████████▉ | 52/58 [00:12<00:00, 20.20it/s]Capturing num tokens (num_tokens=20 avail_mem=26.23 GB):  90%|████████▉ | 52/58 [00:12<00:00, 20.20it/s]Capturing num tokens (num_tokens=16 avail_mem=26.22 GB):  90%|████████▉ | 52/58 [00:12<00:00, 20.20it/s]Capturing num tokens (num_tokens=16 avail_mem=26.22 GB):  95%|█████████▍| 55/58 [00:12<00:00, 20.63it/s]Capturing num tokens (num_tokens=12 avail_mem=26.23 GB):  95%|█████████▍| 55/58 [00:12<00:00, 20.63it/s]

    Capturing num tokens (num_tokens=8 avail_mem=26.21 GB):  95%|█████████▍| 55/58 [00:12<00:00, 20.63it/s] Capturing num tokens (num_tokens=4 avail_mem=26.21 GB):  95%|█████████▍| 55/58 [00:12<00:00, 20.63it/s]Capturing num tokens (num_tokens=4 avail_mem=26.21 GB): 100%|██████████| 58/58 [00:12<00:00, 21.97it/s]Capturing num tokens (num_tokens=4 avail_mem=26.21 GB): 100%|██████████| 58/58 [00:12<00:00,  4.71it/s]


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think the population has been growing over the years. I recall reading somewhere that it's over 21 million now. Maybe around 21.6 million? I'm not sure if that's the exact number or just an estimate. I should look it up to confirm. Also, I should make sure that Paris is indeed the capital and not another city like Lyon or Marseille. I'm pretty sure Paris is the official capital, but I'm not 100% certain. Maybe I can think about the most well-known city in France and that's probably Paris.<br><br>So, putting it all together, the capital is Paris, and the population is approximately 21.6 million. I should present this information in JSON format as the user requested. I need to make sure the JSON is correctly formatted with the key "capital" and "population". I should also include the population as a number, not a string, so it's 21600000. Let me double-check the population number to ensure accuracy. Yeah, I think that's correct. So the final JSON should have the correct structure with the right values.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21600000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should probably check that.<br><br>Wait, I think the population has been growing over the years. I recall that in recent years, Paris has been attracting a lot of immigrants and tourists, so the population might be higher than I initially thought. Maybe it's over 21 million now? I'm not sure if it's exactly 21.3 million or something else. I should look up the latest data to confirm.<br><br>I also wonder if the population figure is accurate for the current year. Population numbers can change yearly due to births, deaths, and migration. So, it's important to get the most recent statistics to provide an accurate answer. I think the best way is to reference a reliable source, like a government website or a recent publication, to get the exact figure.<br><br>After checking, I find that as of 2023, the population of Paris is approximately 21.3 million people. That seems to align with what I thought. So, I can confidently say that the capital of France is Paris, and its population is about 21.3 million.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21300000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population. I think it's a big city, maybe around 3 million? But I'm not certain. I should probably check some reliable sources to confirm this. Maybe I can look up recent population data or news articles that mention Paris's population. I remember hearing that Paris is one of the most populous cities in the world, but I'm not sure if it's over 3 million or not. I should also consider factors like urbanization and migration that might affect the population numbers. Maybe the population has grown a bit since the last census. I'll try to recall if I've heard any recent statistics or if there are any upcoming censuses that might provide the latest data. I think the population figure is something like 3.5 million, but I'm not entirely sure. I should make sure to present this information in a clear and accurate way, perhaps referencing a recent source or official statistics to back it up.<br><br><br>content: Paris is the capital of France</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France. Hmm, I remember learning a bit about France in school, but I'm not 100% sure. Let me think. I know that Paris is a major city in France, and it's often referred to as the "City of Light" because of the famous Eiffel Tower. But is Paris the capital? I think it is, but I'm not entirely certain. <br><br>Wait, I also recall that there's another city called Lyon. Isn't Lyon the capital of a region or something? Maybe I'm mixing up the regions. I think the capital refers to the main city, so Paris might be the official capital. But I'm a bit confused because sometimes people talk about different capitals for regions or departments. For example, I think each department has a capital city, and maybe Paris is the capital of a department or something like that. <br><br>Let me try to remember. The Eiffel Tower is in Paris, and it's a symbol of France. Also, the Louvre Museum is in Paris, which is a world-renowned museum. So, if Paris is such a significant city with all these famous landmarks, it makes sense that it's the capital. But I'm still a little unsure because I think I heard somewhere that Lyon is the capital of France, but that might be incorrect. <br><br>I should probably double-check. I know that the capital is the seat of government, so maybe I can think about other capitals I know. For example, Germany's capital is Berlin, Italy's is Rome, Spain's is Madrid. So, following that pattern, France's capital should be Paris. Yeah, that seems right. I think I was confusing it with another city, maybe Lyon, but no, I'm pretty sure Paris is correct. <br><br>Also, I remember that Paris is the administrative center, where the government offices are located. So, that would make it the capital. Yeah, I'm pretty confident now that Paris is the capital of France.<br><br><br>content: Paris</strong>


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


<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the current date and time in New York and the weather there. I need to figure out how to get both pieces of information using the functions provided.<br><br>First, for the date and time, I should use the get_current_date function. The function requires a timezone parameter. Since the user is in New York, I'll set the timezone to 'America/New_York'. I don't need any other parameters for this function, so the parameters object will be empty.<br><br>Next, for the weather, I should use get_current_weather. This function needs a city and state. The user mentioned being in New York, which is NY, so the city is 'New York' and the state is 'NY'. The unit isn't specified, so I'll default to 'fahrenheit' as it's a common request.<br><br>I'll structure the message by first calling get_current_date with the timezone parameter, then get_current_weather with city and state, each followed by their respective parameters. I'll make sure to format each function call correctly, using the specified JSON structure within the tags.<br><br>Finally, I'll present both results in a clear and concise manner, ensuring that each function call is properly formatted and all required parameters are included.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that\'s the starting point.\n\nNext, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I\'m not exactly sure of the current number. I think it\'s around 2 million, but I should double-check that. Maybe I can recall that it\'s approximately 2,150,000 as of recent estimates.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.\n\nI should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they\'re strings, but population is a number, so it should be without quotes.\n\nPutting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.\n\nI wonder if the user needs more details, like the population figure\'s source or the exact year it was recorded. But since they only asked for the information, I\'ll stick to what\'s requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.\n\nAlso, considering the user\'s possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.\n\nIn summary, I\'ll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I\'ll keep it simple and straightforward since the user didn\'t ask for anything too complex.\n</think>{"name": "Paris", "population": 2150000}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 279, 6722, 315, 9625, 374, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 279, 5916, 1459, 382, 5847, 11, 358, 1184, 311, 1477, 279, 7042, 315, 12095, 13, 358, 6099, 429, 12095, 374, 264, 3598, 3283, 448, 264, 3460, 7042, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 1482, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 13, 10696, 358, 646, 19091, 429, 432, 594, 13187, 220, 17, 11, 16, 20, 15, 11, 15, 15, 15, 438, 315, 3213, 17530, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 1376, 330, 65063, 1, 448, 279, 897, 330, 59604, 1, 323, 2441, 1376, 330, 44441, 1, 448, 279, 1372, 358, 1101, 3381, 315, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 2938, 3363, 1667, 1990, 17194, 369, 6894, 323, 914, 2750, 11, 323, 76602, 34901, 1948, 1376, 19083, 13530, 13, 7281, 11, 279, 5109, 1265, 387, 304, 17194, 421, 807, 2299, 9069, 11, 714, 7042, 374, 264, 1372, 11, 773, 432, 1265, 387, 2041, 17194, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 1265, 1401, 1075, 419, 25, 5212, 65063, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 7810, 358, 1265, 3042, 419, 9355, 773, 279, 1196, 646, 6707, 3535, 323, 990, 279, 1995, 382, 40, 5775, 421, 279, 1196, 3880, 803, 3565, 11, 1075, 279, 7042, 7071, 594, 2530, 476, 279, 4734, 1042, 432, 572, 12433, 13, 1988, 2474, 807, 1172, 4588, 369, 279, 1995, 11, 358, 3278, 9214, 311, 1128, 594, 11223, 7241, 807, 2548, 369, 803, 13, 10696, 358, 1265, 6286, 429, 279, 7042, 7071, 374, 44868, 323, 646, 13289, 916, 882, 382, 13394, 11, 12831, 279, 1196, 594, 3204, 7385, 11, 807, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 264, 1895, 11, 476, 7196, 1101, 4586, 6540, 13, 80100, 13382, 323, 705, 4686, 18413, 1995, 374, 2989, 13, 358, 1265, 5978, 429, 279, 7042, 1372, 374, 3213, 3322, 311, 387, 9760, 382, 641, 12126, 11, 358, 3278, 5944, 279, 2033, 438, 264, 4718, 1633, 448, 279, 1378, 5189, 5043, 11, 3259, 2704, 279, 19482, 374, 4396, 323, 279, 821, 374, 13382, 13, 358, 3278, 2506, 432, 4285, 323, 30339, 2474, 279, 1196, 3207, 944, 2548, 369, 4113, 2238, 6351, 624, 151649, 4913, 606, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 20, 15, 15, 15, 15, 92, 151643], 'meta_info': {'id': 'b8b79c27718e4c0d855b32ba418cf51f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 454, 'completion_tokens': 473, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.05213098693639, 'response_sent_to_client_ts': 1778555214.71504}}



<strong style='color: #00008B;'>reasoing_content: Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what the capital of France is. I know that Paris is the capital, so that's the starting point.<br><br>Next, I need to find the population of Paris. I remember that Paris is a major city with a large population, but I'm not exactly sure of the current number. I think it's around 2 million, but I should double-check that. Maybe I can recall that it's approximately 2,150,000 as of recent estimates.<br><br>Now, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the key "capital" with the value "Paris" and another key "population" with the number I just thought of.<br><br>I should make sure the JSON syntax is correct. That means using double quotes for keys and string values, and commas appropriately between key-value pairs. Also, the numbers should be in quotes if they're strings, but population is a number, so it should be without quotes.<br><br>Putting it all together, the JSON object should look like this: {"capital": "Paris", "population": 2150000}. I should present this clearly so the user can easily understand and use the information.<br><br>I wonder if the user needs more details, like the population figure's source or the exact year it was recorded. But since they only asked for the information, I'll stick to what's requested unless they ask for more. Maybe I should mention that the population figure is approximate and can vary over time.<br><br>Also, considering the user's possible intent, they might be using this data for a project, a report, or maybe just general knowledge. Providing accurate and up-to-date information is important. I should ensure that the population number is recent enough to be relevant.<br><br>In summary, I'll structure the response as a JSON object with the two specified fields, making sure the syntax is correct and the data is accurate. I'll keep it simple and straightforward since the user didn't ask for anything too complex.<br><br><br>content: {"name": "Paris", "population": 2150000}</strong>


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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see, according to the 2020 census, Paris had a population of about 2,174,300. That seems accurate. I should make sure to include this number in the JSON.\n\nNext, I need to structure this information into a JSON format. The user wants a JSON, so I\'ll create an object with a "name" field for the city, "population" for the number, and "description" for a brief overview. The description should mention that Paris is the capital and its population figure.\n\nI should also consider the format. The JSON should be properly formatted with keys and values, and each key should be a string. The population number should be an integer since it\'s a count of people.\n\nPutting it all together, I\'ll write the JSON like this: a main object with "capital" containing the name, population, and description. I\'ll make sure the syntax is correct, with commas and brackets in the right places to avoid errors.\n\nFinally, I\'ll present the JSON to the user, keeping it simple and clear. I don\'t need to add extra information unless the user asks for it, so I\'ll stick to the basics they requested.\n</think>{\n\n"name": "Paris",\n"population": 217430000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 11, 4092, 311, 279, 220, 17, 15, 17, 15, 43602, 11, 12095, 1030, 264, 7042, 315, 911, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 13, 2938, 4977, 13382, 13, 358, 1265, 1281, 2704, 311, 2924, 419, 1372, 304, 279, 4718, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 576, 1196, 6801, 264, 4718, 11, 773, 358, 3278, 1855, 458, 1633, 448, 264, 330, 606, 1, 2070, 369, 279, 3283, 11, 330, 44441, 1, 369, 279, 1372, 11, 323, 330, 4684, 1, 369, 264, 9814, 23251, 13, 576, 4008, 1265, 6286, 429, 12095, 374, 279, 6722, 323, 1181, 7042, 7071, 382, 40, 1265, 1083, 2908, 279, 3561, 13, 576, 4718, 1265, 387, 10277, 23126, 448, 6894, 323, 2750, 11, 323, 1817, 1376, 1265, 387, 264, 914, 13, 576, 7042, 1372, 1265, 387, 458, 7546, 2474, 432, 594, 264, 1760, 315, 1251, 382, 97904, 432, 678, 3786, 11, 358, 3278, 3270, 279, 4718, 1075, 419, 25, 264, 1887, 1633, 448, 330, 65063, 1, 8482, 279, 829, 11, 7042, 11, 323, 4008, 13, 358, 3278, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 76602, 323, 38929, 304, 279, 1290, 7482, 311, 5648, 5975, 382, 23949, 11, 358, 3278, 3042, 279, 4718, 311, 279, 1196, 11, 10282, 432, 4285, 323, 2797, 13, 358, 1513, 944, 1184, 311, 912, 4960, 1995, 7241, 279, 1196, 17064, 369, 432, 11, 773, 358, 3278, 9214, 311, 279, 31774, 807, 11223, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': '7d01eb6e0ad74449a9a73789a506127a', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 363, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 20.16675909841433, 'response_sent_to_client_ts': 1778555234.891553}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'f30dac1e0fa4484488f75783192ec0ee', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14490794017910957, 'response_sent_to_client_ts': 1778555235.0820045}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '765209e7bbad4eeba27eda87137a276e', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1448392909951508, 'response_sent_to_client_ts': 1778555235.0820186}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': 'c1b94a80c84348de8e145461494d6848', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1448002546094358, 'response_sent_to_client_ts': 1778555235.082023}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': '629ffe30e7784fb489474ba247a84a1c', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 19.045597842894495, 'response_sent_to_client_ts': 1778555254.1349475}}


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


<strong style='color: #00008B;'>{'text': 'Alright, I just received a query from a user asking for the information and population of the capital of France in JSON format. Let me figure out how to approach this.\n\nFirst, I need to understand the user\'s request accurately. The user wants the population of the capital, which is Paris, in a JSON structure. They might be using this data for a project, report, or maybe just personal knowledge. Since it\'s JSON, they likely plan to use this data in another application or program.\n\nI should verify the latest population figure. Population data can change yearly, so it\'s important to provide the most recent and accurate numbers. As of 2023, I know the population is around 3.5 million in Paris and about 11 million for the Île-de-France region. I\'ll make sure to include these numbers accurately.\n\nNext, I should define the structure of the JSON. It should include the country, the city name, population figures, and any other relevant information. Maybe adding some notable landmarks or cultural facts could add depth to the JSON, making it more informative. Including an approximate date when the population estimate is current is also a good practice to ensure the data remains relevant.\n\nI also need to ensure that the JSON syntax is correct to prevent any errors. Proper indentation, commas, and brackets are essential for JSON to parse correctly. If I make a mistake in the formatting, it could cause the user\'s application to fail when trying to use the data.\n\nThinking about possible edge cases, the user might be looking for a more detailed breakdown, like population by age group or demographics. However, since they only asked for the overall population and provided a sample response, sticking to the basics might be sufficient unless they indicate a need for more detailed data.\n\nI should also consider the user\'s possible unspoken needs. They might be interested in comparing population sizes of capitals, so including the population of the country in JSON could be helpful. That way, they can include this data alongside other capitals if they need to.\n\nLastly, I\'ll present the JSON in a clear, easy-to-read format. Using pretty formatting will make it more readable, even though it\'s just a small dataset. This approach ensures the user gets exactly what they need in a clean and organized manner.\n</think>\n\nHere is the JSON information and population of the capital of France (Paris):\n\n```json\n{\n  "country": "France",\n  "capital": "Paris",\n  "population": {\n    "city": "3,500,000",\n    "region": "11,500,000"\n  },\n  " landmarks": {\n    "Notre-Dame Cathedral": "World heritage site",\n    "Eiffel Tower": " famous landmark",\n    "Louvre Museum": "World\'s largest art museum"\n  },\n  "people_rank": "2nd most populous capital in Europe"\n}\n```', 'output_ids': [71486, 11, 358, 1101, 3949, 264, 3239, 504, 264, 1196, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 7071, 700, 1246, 311, 5486, 419, 382, 5338, 11, 358, 1184, 311, 3535, 279, 1196, 594, 1681, 29257, 13, 576, 1196, 6801, 279, 7042, 315, 279, 6722, 11, 892, 374, 12095, 11, 304, 264, 4718, 5944, 13, 2379, 2578, 387, 1667, 419, 821, 369, 264, 2390, 11, 1895, 11, 476, 7196, 1101, 4345, 6540, 13, 8704, 432, 594, 4718, 11, 807, 4363, 3119, 311, 990, 419, 821, 304, 2441, 3766, 476, 2025, 382, 40, 1265, 10146, 279, 5535, 7042, 7071, 13, 39529, 821, 646, 2297, 44270, 11, 773, 432, 594, 2989, 311, 3410, 279, 1429, 3213, 323, 13382, 5109, 13, 1634, 315, 220, 17, 15, 17, 18, 11, 358, 1414, 279, 7042, 374, 2163, 220, 18, 13, 20, 3526, 304, 12095, 323, 911, 220, 16, 16, 3526, 369, 279, 59108, 273, 6810, 7276, 34106, 5537, 13, 358, 3278, 1281, 2704, 311, 2924, 1493, 5109, 29257, 382, 5847, 11, 358, 1265, 6979, 279, 5944, 315, 279, 4718, 13, 1084, 1265, 2924, 279, 3146, 11, 279, 3283, 829, 11, 7042, 12396, 11, 323, 894, 1008, 9760, 1995, 13, 10696, 7842, 1045, 27190, 59924, 476, 12752, 13064, 1410, 912, 7990, 311, 279, 4718, 11, 3259, 432, 803, 38219, 13, 55121, 458, 44868, 2400, 979, 279, 7042, 16045, 374, 1482, 374, 1083, 264, 1661, 6588, 311, 5978, 279, 821, 8458, 9760, 382, 40, 1083, 1184, 311, 5978, 429, 279, 4718, 19482, 374, 4396, 311, 5358, 894, 5975, 13, 64558, 69592, 11, 76602, 11, 323, 38929, 525, 7565, 369, 4718, 311, 4715, 12440, 13, 1416, 358, 1281, 264, 16523, 304, 279, 36566, 11, 432, 1410, 5240, 279, 1196, 594, 3766, 311, 3690, 979, 4460, 311, 990, 279, 821, 382, 93945, 911, 3204, 6821, 5048, 11, 279, 1196, 2578, 387, 3330, 369, 264, 803, 11682, 29985, 11, 1075, 7042, 553, 4231, 1874, 476, 62234, 13, 4354, 11, 2474, 807, 1172, 4588, 369, 279, 8084, 7042, 323, 3897, 264, 6077, 2033, 11, 36972, 311, 279, 31774, 2578, 387, 14016, 7241, 807, 13216, 264, 1184, 369, 803, 11682, 821, 382, 40, 1265, 1083, 2908, 279, 1196, 594, 3204, 650, 51758, 3880, 13, 2379, 2578, 387, 8014, 304, 26297, 7042, 12282, 315, 92999, 11, 773, 2670, 279, 7042, 315, 279, 3146, 304, 4718, 1410, 387, 10950, 13, 2938, 1616, 11, 807, 646, 2924, 419, 821, 16263, 1008, 92999, 421, 807, 1184, 311, 382, 80486, 11, 358, 3278, 3042, 279, 4718, 304, 264, 2797, 11, 4135, 4686, 28806, 3561, 13, 12091, 5020, 36566, 686, 1281, 432, 803, 33798, 11, 1496, 3498, 432, 594, 1101, 264, 2613, 10337, 13, 1096, 5486, 25351, 279, 1196, 5221, 6896, 1128, 807, 1184, 304, 264, 4240, 323, 16645, 11566, 624, 151649, 271, 8420, 374, 279, 4718, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 320, 59604, 7731, 73594, 2236, 198, 515, 220, 330, 11141, 788, 330, 49000, 756, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 341, 262, 330, 8926, 788, 330, 18, 11, 20, 15, 15, 11, 15, 15, 15, 756, 262, 330, 3943, 788, 330, 16, 16, 11, 20, 15, 15, 11, 15, 15, 15, 698, 220, 1153, 220, 330, 59924, 788, 341, 262, 330, 2623, 265, 9420, 373, 56729, 788, 330, 10134, 27848, 2747, 756, 262, 330, 36, 3092, 301, 21938, 788, 330, 11245, 37250, 756, 262, 330, 92806, 48506, 16328, 788, 330, 10134, 594, 7772, 1947, 23971, 698, 220, 1153, 220, 330, 16069, 20417, 788, 330, 17, 303, 1429, 94451, 6722, 304, 4505, 698, 532, 73594, 151643], 'meta_info': {'id': '316adaa4d4704afa8c4931deecd245d0', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 463, 'completion_tokens': 596, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 5.2353333458304405, 'response_sent_to_client_ts': 1778555259.3788822}}</strong>



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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.37s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.30s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.31s/it]


    2026-05-12 03:07:56,767 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 03:07:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:36,  5.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:36,  5.91s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:30,  2.69s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:30,  2.69s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:25,  1.55s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:25,  1.55s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:54,  1.01s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:54,  1.01s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:26,  1.93it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:26,  1.93it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:20,  2.55it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:20,  2.55it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:15,  3.26it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:15,  3.26it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:11,  4.09it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:11,  4.09it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:07<00:11,  4.09it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:08,  5.77it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:08,  5.77it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:08,  5.77it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.34it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.34it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.34it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  8.96it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  8.96it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  8.96it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:08<00:03, 10.84it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:08<00:03, 10.84it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:08<00:03, 10.84it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:08<00:03, 10.84it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:08<00:02, 14.03it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:08<00:02, 14.03it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:08<00:02, 14.03it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:08<00:02, 14.03it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:08<00:02, 14.03it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:08<00:01, 19.32it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:08<00:01, 19.32it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:08<00:01, 19.32it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:08<00:01, 19.32it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:08<00:01, 19.32it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:08<00:01, 19.32it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:08<00:01, 26.24it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:08<00:01, 26.24it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:08<00:01, 26.24it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:08<00:01, 26.24it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:08<00:01, 26.24it/s]

    Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:08<00:01, 26.24it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:08<00:00, 31.80it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:08<00:00, 31.80it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:08<00:00, 31.80it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:08<00:00, 31.80it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:08<00:00, 31.80it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:08<00:00, 31.80it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:08<00:00, 31.80it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:08<00:00, 37.75it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:08<00:00, 37.75it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:08<00:00, 37.75it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:08<00:00, 37.75it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:08<00:00, 37.75it/s]

    Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:08<00:00, 37.75it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:08<00:00, 37.75it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:08<00:00, 43.03it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:08<00:00, 43.03it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:08<00:00, 50.26it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:08<00:00, 50.26it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:08<00:00, 50.26it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:08<00:00, 50.26it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:08<00:00, 50.26it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:08<00:00, 50.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.23 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.23 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.20 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.20 GB):   3%|▎         | 2/58 [00:01<00:34,  1.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.20 GB):   3%|▎         | 2/58 [00:01<00:34,  1.61it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.20 GB):   5%|▌         | 3/58 [00:01<00:31,  1.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.20 GB):   5%|▌         | 3/58 [00:01<00:31,  1.75it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.20 GB):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.20 GB):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.20 GB):   9%|▊         | 5/58 [00:02<00:26,  2.01it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.20 GB):   9%|▊         | 5/58 [00:02<00:26,  2.01it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.20 GB):  10%|█         | 6/58 [00:03<00:23,  2.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.20 GB):  10%|█         | 6/58 [00:03<00:23,  2.19it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.20 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.20 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.34it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=25.20 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=25.20 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.60it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=25.20 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.20 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.89it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=25.20 GB):  17%|█▋        | 10/58 [00:04<00:14,  3.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=25.20 GB):  17%|█▋        | 10/58 [00:04<00:14,  3.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=25.20 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=25.19 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=25.19 GB):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=25.19 GB):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=25.19 GB):  22%|██▏       | 13/58 [00:04<00:09,  4.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=25.19 GB):  22%|██▏       | 13/58 [00:04<00:09,  4.74it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=25.19 GB):  24%|██▍       | 14/58 [00:04<00:08,  5.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=25.18 GB):  24%|██▍       | 14/58 [00:04<00:08,  5.29it/s]Capturing num tokens (num_tokens=2304 avail_mem=25.18 GB):  24%|██▍       | 14/58 [00:04<00:08,  5.29it/s]Capturing num tokens (num_tokens=2304 avail_mem=25.18 GB):  28%|██▊       | 16/58 [00:04<00:05,  7.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=25.18 GB):  28%|██▊       | 16/58 [00:04<00:05,  7.25it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=25.18 GB):  29%|██▉       | 17/58 [00:05<00:08,  5.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=25.18 GB):  29%|██▉       | 17/58 [00:05<00:08,  5.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=25.17 GB):  29%|██▉       | 17/58 [00:05<00:08,  5.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=25.17 GB):  33%|███▎      | 19/58 [00:05<00:05,  7.18it/s]Capturing num tokens (num_tokens=1280 avail_mem=25.17 GB):  33%|███▎      | 19/58 [00:05<00:05,  7.18it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=25.17 GB):  34%|███▍      | 20/58 [00:05<00:04,  7.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.03 GB):  34%|███▍      | 20/58 [00:05<00:04,  7.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.03 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.39it/s]Capturing num tokens (num_tokens=960 avail_mem=24.02 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.39it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=24.02 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.78it/s]Capturing num tokens (num_tokens=896 avail_mem=25.12 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.78it/s]Capturing num tokens (num_tokens=832 avail_mem=25.12 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.78it/s]

    Capturing num tokens (num_tokens=832 avail_mem=25.12 GB):  41%|████▏     | 24/58 [00:06<00:04,  8.26it/s]Capturing num tokens (num_tokens=768 avail_mem=24.13 GB):  41%|████▏     | 24/58 [00:06<00:04,  8.26it/s]Capturing num tokens (num_tokens=768 avail_mem=24.13 GB):  43%|████▎     | 25/58 [00:06<00:04,  8.10it/s]Capturing num tokens (num_tokens=704 avail_mem=24.12 GB):  43%|████▎     | 25/58 [00:06<00:04,  8.10it/s]

    Capturing num tokens (num_tokens=704 avail_mem=24.12 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.22it/s]Capturing num tokens (num_tokens=640 avail_mem=25.10 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.22it/s]Capturing num tokens (num_tokens=640 avail_mem=25.10 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.38it/s]Capturing num tokens (num_tokens=576 avail_mem=24.18 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.38it/s]

    Capturing num tokens (num_tokens=576 avail_mem=24.18 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.28it/s]Capturing num tokens (num_tokens=512 avail_mem=24.17 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.28it/s]Capturing num tokens (num_tokens=512 avail_mem=24.17 GB):  50%|█████     | 29/58 [00:06<00:03,  8.31it/s]Capturing num tokens (num_tokens=480 avail_mem=25.05 GB):  50%|█████     | 29/58 [00:06<00:03,  8.31it/s]

    Capturing num tokens (num_tokens=480 avail_mem=25.05 GB):  52%|█████▏    | 30/58 [00:06<00:03,  8.36it/s]Capturing num tokens (num_tokens=448 avail_mem=24.23 GB):  52%|█████▏    | 30/58 [00:06<00:03,  8.36it/s]Capturing num tokens (num_tokens=448 avail_mem=24.23 GB):  53%|█████▎    | 31/58 [00:06<00:03,  8.35it/s]Capturing num tokens (num_tokens=416 avail_mem=25.09 GB):  53%|█████▎    | 31/58 [00:06<00:03,  8.35it/s]

    Capturing num tokens (num_tokens=384 avail_mem=24.30 GB):  53%|█████▎    | 31/58 [00:06<00:03,  8.35it/s]Capturing num tokens (num_tokens=384 avail_mem=24.30 GB):  57%|█████▋    | 33/58 [00:07<00:02,  9.10it/s]Capturing num tokens (num_tokens=352 avail_mem=24.29 GB):  57%|█████▋    | 33/58 [00:07<00:02,  9.10it/s]

    Capturing num tokens (num_tokens=352 avail_mem=24.29 GB):  59%|█████▊    | 34/58 [00:07<00:02,  9.25it/s]Capturing num tokens (num_tokens=320 avail_mem=25.07 GB):  59%|█████▊    | 34/58 [00:07<00:02,  9.25it/s]Capturing num tokens (num_tokens=288 avail_mem=24.35 GB):  59%|█████▊    | 34/58 [00:07<00:02,  9.25it/s]Capturing num tokens (num_tokens=288 avail_mem=24.35 GB):  62%|██████▏   | 36/58 [00:07<00:02,  9.76it/s]Capturing num tokens (num_tokens=256 avail_mem=24.34 GB):  62%|██████▏   | 36/58 [00:07<00:02,  9.76it/s]

    Capturing num tokens (num_tokens=240 avail_mem=25.06 GB):  62%|██████▏   | 36/58 [00:07<00:02,  9.76it/s]Capturing num tokens (num_tokens=240 avail_mem=25.06 GB):  66%|██████▌   | 38/58 [00:07<00:01, 10.71it/s]Capturing num tokens (num_tokens=224 avail_mem=24.40 GB):  66%|██████▌   | 38/58 [00:07<00:01, 10.71it/s]Capturing num tokens (num_tokens=208 avail_mem=24.36 GB):  66%|██████▌   | 38/58 [00:07<00:01, 10.71it/s]

    Capturing num tokens (num_tokens=208 avail_mem=24.36 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.21it/s]Capturing num tokens (num_tokens=192 avail_mem=25.02 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.21it/s]Capturing num tokens (num_tokens=176 avail_mem=24.42 GB):  69%|██████▉   | 40/58 [00:07<00:01, 11.21it/s]Capturing num tokens (num_tokens=176 avail_mem=24.42 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.35it/s]Capturing num tokens (num_tokens=160 avail_mem=25.02 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.35it/s]

    Capturing num tokens (num_tokens=144 avail_mem=24.49 GB):  72%|███████▏  | 42/58 [00:07<00:01, 11.35it/s]Capturing num tokens (num_tokens=144 avail_mem=24.49 GB):  76%|███████▌  | 44/58 [00:08<00:01, 11.55it/s]Capturing num tokens (num_tokens=128 avail_mem=25.01 GB):  76%|███████▌  | 44/58 [00:08<00:01, 11.55it/s]Capturing num tokens (num_tokens=112 avail_mem=24.51 GB):  76%|███████▌  | 44/58 [00:08<00:01, 11.55it/s]

    Capturing num tokens (num_tokens=112 avail_mem=24.51 GB):  79%|███████▉  | 46/58 [00:08<00:01, 11.88it/s]Capturing num tokens (num_tokens=96 avail_mem=25.00 GB):  79%|███████▉  | 46/58 [00:08<00:01, 11.88it/s] Capturing num tokens (num_tokens=80 avail_mem=24.53 GB):  79%|███████▉  | 46/58 [00:08<00:01, 11.88it/s]Capturing num tokens (num_tokens=80 avail_mem=24.53 GB):  83%|████████▎ | 48/58 [00:08<00:00, 12.32it/s]Capturing num tokens (num_tokens=64 avail_mem=24.99 GB):  83%|████████▎ | 48/58 [00:08<00:00, 12.32it/s]

    Capturing num tokens (num_tokens=48 avail_mem=24.56 GB):  83%|████████▎ | 48/58 [00:08<00:00, 12.32it/s]Capturing num tokens (num_tokens=48 avail_mem=24.56 GB):  86%|████████▌ | 50/58 [00:08<00:00, 12.82it/s]Capturing num tokens (num_tokens=32 avail_mem=24.98 GB):  86%|████████▌ | 50/58 [00:08<00:00, 12.82it/s]Capturing num tokens (num_tokens=28 avail_mem=24.58 GB):  86%|████████▌ | 50/58 [00:08<00:00, 12.82it/s]

    Capturing num tokens (num_tokens=28 avail_mem=24.58 GB):  90%|████████▉ | 52/58 [00:08<00:00, 13.25it/s]Capturing num tokens (num_tokens=24 avail_mem=24.98 GB):  90%|████████▉ | 52/58 [00:08<00:00, 13.25it/s]Capturing num tokens (num_tokens=20 avail_mem=24.61 GB):  90%|████████▉ | 52/58 [00:08<00:00, 13.25it/s]Capturing num tokens (num_tokens=20 avail_mem=24.61 GB):  93%|█████████▎| 54/58 [00:08<00:00, 13.71it/s]Capturing num tokens (num_tokens=16 avail_mem=24.97 GB):  93%|█████████▎| 54/58 [00:08<00:00, 13.71it/s]

    Capturing num tokens (num_tokens=12 avail_mem=24.97 GB):  93%|█████████▎| 54/58 [00:08<00:00, 13.71it/s]Capturing num tokens (num_tokens=12 avail_mem=24.97 GB):  97%|█████████▋| 56/58 [00:08<00:00, 14.13it/s]Capturing num tokens (num_tokens=8 avail_mem=24.66 GB):  97%|█████████▋| 56/58 [00:08<00:00, 14.13it/s] Capturing num tokens (num_tokens=4 avail_mem=24.96 GB):  97%|█████████▋| 56/58 [00:08<00:00, 14.13it/s]Capturing num tokens (num_tokens=4 avail_mem=24.96 GB): 100%|██████████| 58/58 [00:09<00:00, 14.40it/s]Capturing num tokens (num_tokens=4 avail_mem=24.96 GB): 100%|██████████| 58/58 [00:09<00:00,  6.42it/s]


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
    
    Generated text: Alright, the user is asking for the information and population of the capital of France in JSON format. Hmm, first, I should figure out which city is the capital. I remember it's Paris. Okay, so I'll need to find the population. Wait, I'm pretty sure the population of Paris is around 2 million. But I should double-check to make sure I'm accurate. 
    
    Let me think about the current data. In recent years, Paris has been growing a bit, maybe a little over 2 million. But I should confirm that. Maybe the exact figure is 2,154,000 as of 2023? I think that's about right. 
    
    Now, structuring this into JSON. JSON format typically uses key-value pairs. So I'll need a main key, maybe "capital_info", which contains an object with "name", "population", and "coordinates". The population should be a number, and the coordinates are a tuple with longitude and latitude. 
    
    Wait, is the JSON structure clear? Let me make sure. The key "capital_info" has another object with "name", "population", and "coordinates". The population is 2154000, so as a number, that's fine. Coordinates for Paris are approximately 48.8566° N, 2.3522° E. 
    
    I should present this neatly, probably with proper indentation for readability. The user didn't specify any particular formatting beyond JSON, so this should be sufficient. 
    
    I wonder if the user is a developer working on a project that requires structured data. They might need this JSON for mapping services or integrating into an application. By providing the exact format, I'm helping them integrate seamlessly. 
    
    Also, maybe they're just curious about Paris's demographics and location. Either way, the JSON structure clearly presents the necessary information. 
    
    I should make sure to present the data accurately. If there's any uncertainty, it's better to be precise. Double-checking the population and coordinates to ensure they're up to date. 
    
    In summary, the response is straightforward: provide the JSON with the correct details about Paris's capital status, population, and coordinates. That should meet the user's needs effectively.
    </think>
    
    Here is the information and population of the capital of France in JSON format:
    
    ```json
    {
      "capital_info": {
        "name": "Paris",
        "population": 2154000,
        "coordinates": {
          "longitude": 2.3522,
          "latitude": 48.8566
        }
      }
    }
    ```



```python
llm.shutdown()
```
