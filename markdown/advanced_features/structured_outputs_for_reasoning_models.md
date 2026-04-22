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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(
    [2026-04-22 21:24:31] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-22 21:24:32] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-22 21:24:33] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-22 21:24:35] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-22 21:24:42] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-22 21:24:44] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False
    [2026-04-22 21:24:44] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.15s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.10s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.11s/it]


    2026-04-22 21:24:52,031 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 21:24:52] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:55,  3.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:55,  3.08s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:55,  2.06s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:55,  2.06s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:17,  1.40s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:17,  1.40s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:58,  1.07s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:58,  1.07s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:45,  1.17it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:45,  1.17it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:30,  1.66it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:30,  1.66it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:25,  1.94it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:25,  1.94it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:21,  2.28it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:21,  2.28it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:18,  2.63it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:18,  2.63it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:15,  2.98it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:15,  2.98it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:13,  3.32it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:13,  3.32it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:11,  3.82it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:11,  3.82it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:10,  4.33it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:10,  4.33it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:08,  4.90it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:08,  4.90it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:08,  5.01it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:08,  5.01it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:07,  5.21it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:07,  5.21it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:07,  5.54it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:07,  5.54it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:09<00:06,  6.04it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:09<00:06,  6.04it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:05,  6.84it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:05,  6.84it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:05,  6.84it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:09<00:04,  8.74it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:09<00:04,  8.74it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:09<00:04,  8.74it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:09<00:03,  9.65it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:09<00:03,  9.65it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:09<00:03,  9.65it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:09<00:02, 10.77it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:09<00:02, 10.77it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:09<00:02, 10.77it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:10<00:02, 11.81it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:10<00:02, 11.81it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:10<00:02, 11.81it/s]

    Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:10<00:02, 11.81it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:10<00:01, 14.28it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:10<00:01, 14.28it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:10<00:01, 14.28it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:10<00:01, 14.93it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:10<00:01, 14.93it/s]

    Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:10<00:01, 14.93it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:10<00:01, 14.93it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:10<00:01, 16.54it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:10<00:01, 16.54it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:10<00:01, 16.54it/s]

    Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:10<00:01, 16.54it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:10<00:01, 18.62it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:10<00:01, 18.62it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:10<00:01, 18.62it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:10<00:01, 18.62it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:10<00:00, 20.27it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:10<00:00, 20.27it/s]

    Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:10<00:00, 20.27it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:10<00:00, 20.27it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:10<00:00, 20.77it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:10<00:00, 20.77it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:10<00:00, 20.77it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:10<00:00, 20.77it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:10<00:00, 21.58it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:10<00:00, 21.58it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:11<00:00, 21.58it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:11<00:00, 21.58it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:11<00:00, 22.57it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:11<00:00, 22.57it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:11<00:00, 22.57it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:11<00:00, 22.57it/s]

    Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:11<00:00, 22.57it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:11<00:00, 24.57it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:11<00:00, 24.57it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:11<00:00, 24.57it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:11<00:00, 24.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  5.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.83 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.83 GB):   2%|▏         | 1/58 [00:00<00:49,  1.14it/s]Capturing num tokens (num_tokens=7680 avail_mem=26.02 GB):   2%|▏         | 1/58 [00:00<00:49,  1.14it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=26.02 GB):   3%|▎         | 2/58 [00:01<00:46,  1.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.63 GB):   3%|▎         | 2/58 [00:01<00:46,  1.21it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.63 GB):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.27 GB):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.27 GB):   7%|▋         | 4/58 [00:02<00:37,  1.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.41 GB):   7%|▋         | 4/58 [00:02<00:37,  1.43it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.41 GB):   9%|▊         | 5/58 [00:03<00:34,  1.53it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.84 GB):   9%|▊         | 5/58 [00:03<00:34,  1.53it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.84 GB):  10%|█         | 6/58 [00:04<00:30,  1.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.91 GB):  10%|█         | 6/58 [00:04<00:30,  1.68it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.91 GB):  12%|█▏        | 7/58 [00:04<00:27,  1.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.78 GB):  12%|█▏        | 7/58 [00:04<00:27,  1.86it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=25.78 GB):  14%|█▍        | 8/58 [00:04<00:23,  2.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.06 GB):  14%|█▍        | 8/58 [00:04<00:23,  2.09it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.06 GB):  16%|█▌        | 9/58 [00:05<00:21,  2.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.09 GB):  16%|█▌        | 9/58 [00:05<00:21,  2.32it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=26.09 GB):  17%|█▋        | 10/58 [00:05<00:18,  2.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=26.11 GB):  17%|█▋        | 10/58 [00:05<00:18,  2.53it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=26.11 GB):  19%|█▉        | 11/58 [00:05<00:16,  2.80it/s]Capturing num tokens (num_tokens=3328 avail_mem=26.04 GB):  19%|█▉        | 11/58 [00:05<00:16,  2.80it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=26.04 GB):  21%|██        | 12/58 [00:05<00:14,  3.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=26.16 GB):  21%|██        | 12/58 [00:05<00:14,  3.10it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=26.16 GB):  22%|██▏       | 13/58 [00:06<00:13,  3.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.19 GB):  22%|██▏       | 13/58 [00:06<00:13,  3.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.19 GB):  24%|██▍       | 14/58 [00:06<00:11,  3.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.17 GB):  24%|██▍       | 14/58 [00:06<00:11,  3.73it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.17 GB):  26%|██▌       | 15/58 [00:06<00:10,  4.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.25 GB):  26%|██▌       | 15/58 [00:06<00:10,  4.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.25 GB):  28%|██▊       | 16/58 [00:06<00:09,  4.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.23 GB):  28%|██▊       | 16/58 [00:06<00:09,  4.51it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=26.23 GB):  29%|██▉       | 17/58 [00:06<00:08,  5.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.55 GB):  29%|██▉       | 17/58 [00:06<00:08,  5.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.55 GB):  31%|███       | 18/58 [00:07<00:07,  5.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.50 GB):  31%|███       | 18/58 [00:07<00:07,  5.69it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=26.50 GB):  33%|███▎      | 19/58 [00:07<00:06,  6.35it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.52 GB):  33%|███▎      | 19/58 [00:07<00:06,  6.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.51 GB):  33%|███▎      | 19/58 [00:07<00:06,  6.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.51 GB):  36%|███▌      | 21/58 [00:07<00:04,  8.00it/s]Capturing num tokens (num_tokens=960 avail_mem=26.49 GB):  36%|███▌      | 21/58 [00:07<00:04,  8.00it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=26.46 GB):  36%|███▌      | 21/58 [00:07<00:04,  8.00it/s]Capturing num tokens (num_tokens=896 avail_mem=26.46 GB):  40%|███▉      | 23/58 [00:07<00:03,  9.24it/s]Capturing num tokens (num_tokens=832 avail_mem=26.48 GB):  40%|███▉      | 23/58 [00:07<00:03,  9.24it/s]Capturing num tokens (num_tokens=768 avail_mem=26.47 GB):  40%|███▉      | 23/58 [00:07<00:03,  9.24it/s]

    Capturing num tokens (num_tokens=768 avail_mem=26.47 GB):  43%|████▎     | 25/58 [00:07<00:03, 10.56it/s]Capturing num tokens (num_tokens=704 avail_mem=26.46 GB):  43%|████▎     | 25/58 [00:07<00:03, 10.56it/s]Capturing num tokens (num_tokens=640 avail_mem=26.45 GB):  43%|████▎     | 25/58 [00:07<00:03, 10.56it/s]Capturing num tokens (num_tokens=640 avail_mem=26.45 GB):  47%|████▋     | 27/58 [00:07<00:02, 11.77it/s]Capturing num tokens (num_tokens=576 avail_mem=26.44 GB):  47%|████▋     | 27/58 [00:07<00:02, 11.77it/s]Capturing num tokens (num_tokens=512 avail_mem=26.43 GB):  47%|████▋     | 27/58 [00:07<00:02, 11.77it/s]

    Capturing num tokens (num_tokens=512 avail_mem=26.43 GB):  50%|█████     | 29/58 [00:07<00:02, 13.14it/s]Capturing num tokens (num_tokens=480 avail_mem=26.42 GB):  50%|█████     | 29/58 [00:07<00:02, 13.14it/s]Capturing num tokens (num_tokens=448 avail_mem=26.41 GB):  50%|█████     | 29/58 [00:07<00:02, 13.14it/s]Capturing num tokens (num_tokens=448 avail_mem=26.41 GB):  53%|█████▎    | 31/58 [00:07<00:01, 14.08it/s]Capturing num tokens (num_tokens=416 avail_mem=26.37 GB):  53%|█████▎    | 31/58 [00:07<00:01, 14.08it/s]Capturing num tokens (num_tokens=384 avail_mem=26.38 GB):  53%|█████▎    | 31/58 [00:08<00:01, 14.08it/s]

    Capturing num tokens (num_tokens=384 avail_mem=26.38 GB):  57%|█████▋    | 33/58 [00:08<00:01, 15.22it/s]Capturing num tokens (num_tokens=352 avail_mem=26.35 GB):  57%|█████▋    | 33/58 [00:08<00:01, 15.22it/s]Capturing num tokens (num_tokens=320 avail_mem=26.34 GB):  57%|█████▋    | 33/58 [00:08<00:01, 15.22it/s]Capturing num tokens (num_tokens=288 avail_mem=26.35 GB):  57%|█████▋    | 33/58 [00:08<00:01, 15.22it/s]Capturing num tokens (num_tokens=288 avail_mem=26.35 GB):  62%|██████▏   | 36/58 [00:08<00:01, 17.02it/s]Capturing num tokens (num_tokens=256 avail_mem=26.34 GB):  62%|██████▏   | 36/58 [00:08<00:01, 17.02it/s]Capturing num tokens (num_tokens=240 avail_mem=26.33 GB):  62%|██████▏   | 36/58 [00:08<00:01, 17.02it/s]

    Capturing num tokens (num_tokens=224 avail_mem=26.32 GB):  62%|██████▏   | 36/58 [00:08<00:01, 17.02it/s]Capturing num tokens (num_tokens=224 avail_mem=26.32 GB):  67%|██████▋   | 39/58 [00:08<00:01, 18.74it/s]Capturing num tokens (num_tokens=208 avail_mem=26.31 GB):  67%|██████▋   | 39/58 [00:08<00:01, 18.74it/s]Capturing num tokens (num_tokens=192 avail_mem=26.30 GB):  67%|██████▋   | 39/58 [00:08<00:01, 18.74it/s]Capturing num tokens (num_tokens=176 avail_mem=26.29 GB):  67%|██████▋   | 39/58 [00:08<00:01, 18.74it/s]Capturing num tokens (num_tokens=176 avail_mem=26.29 GB):  72%|███████▏  | 42/58 [00:08<00:00, 21.00it/s]Capturing num tokens (num_tokens=160 avail_mem=26.29 GB):  72%|███████▏  | 42/58 [00:08<00:00, 21.00it/s]Capturing num tokens (num_tokens=144 avail_mem=26.29 GB):  72%|███████▏  | 42/58 [00:08<00:00, 21.00it/s]

    Capturing num tokens (num_tokens=128 avail_mem=26.30 GB):  72%|███████▏  | 42/58 [00:08<00:00, 21.00it/s]Capturing num tokens (num_tokens=128 avail_mem=26.30 GB):  78%|███████▊  | 45/58 [00:08<00:00, 22.86it/s]Capturing num tokens (num_tokens=112 avail_mem=26.29 GB):  78%|███████▊  | 45/58 [00:08<00:00, 22.86it/s]Capturing num tokens (num_tokens=96 avail_mem=26.28 GB):  78%|███████▊  | 45/58 [00:08<00:00, 22.86it/s] Capturing num tokens (num_tokens=80 avail_mem=26.27 GB):  78%|███████▊  | 45/58 [00:08<00:00, 22.86it/s]Capturing num tokens (num_tokens=80 avail_mem=26.27 GB):  83%|████████▎ | 48/58 [00:08<00:00, 24.21it/s]Capturing num tokens (num_tokens=64 avail_mem=26.27 GB):  83%|████████▎ | 48/58 [00:08<00:00, 24.21it/s]Capturing num tokens (num_tokens=48 avail_mem=26.26 GB):  83%|████████▎ | 48/58 [00:08<00:00, 24.21it/s]

    Capturing num tokens (num_tokens=32 avail_mem=26.25 GB):  83%|████████▎ | 48/58 [00:08<00:00, 24.21it/s]Capturing num tokens (num_tokens=32 avail_mem=26.25 GB):  88%|████████▊ | 51/58 [00:08<00:00, 25.52it/s]Capturing num tokens (num_tokens=28 avail_mem=26.25 GB):  88%|████████▊ | 51/58 [00:08<00:00, 25.52it/s]Capturing num tokens (num_tokens=24 avail_mem=26.24 GB):  88%|████████▊ | 51/58 [00:08<00:00, 25.52it/s]Capturing num tokens (num_tokens=20 avail_mem=26.23 GB):  88%|████████▊ | 51/58 [00:08<00:00, 25.52it/s]Capturing num tokens (num_tokens=20 avail_mem=26.23 GB):  93%|█████████▎| 54/58 [00:08<00:00, 26.54it/s]Capturing num tokens (num_tokens=16 avail_mem=26.23 GB):  93%|█████████▎| 54/58 [00:08<00:00, 26.54it/s]Capturing num tokens (num_tokens=12 avail_mem=26.22 GB):  93%|█████████▎| 54/58 [00:08<00:00, 26.54it/s]

    Capturing num tokens (num_tokens=8 avail_mem=26.21 GB):  93%|█████████▎| 54/58 [00:08<00:00, 26.54it/s] Capturing num tokens (num_tokens=8 avail_mem=26.21 GB):  98%|█████████▊| 57/58 [00:09<00:00, 26.35it/s]Capturing num tokens (num_tokens=4 avail_mem=26.21 GB):  98%|█████████▊| 57/58 [00:09<00:00, 26.35it/s]Capturing num tokens (num_tokens=4 avail_mem=26.21 GB): 100%|██████████| 58/58 [00:09<00:00,  6.42it/s]


    [2026-04-22 21:25:14] Tokenizer loaded as generic TokenizersBackend for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, retrying with use_fast=False


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


<strong style='color: #00008B;'>reasoing_content: Okay, so I need to figure out the capital of France and its population. I know that the capital of France is Paris, but I'm not exactly sure about the current population numbers. I remember that Paris is a very big city, but I think it's not the largest in the world. Maybe around 20 million? I'm not certain, though. I should check if that's correct.<br><br>Wait, I think the population might have changed a bit over the years. I recall reading somewhere that Paris has grown a lot, especially with the influx of people moving there for work. But I'm not sure if it's exactly 21 million or maybe a bit more. I should look up the latest data to confirm.<br><br>I also wonder if the population figure includes just the city proper or the entire metropolitan area. Sometimes, people talk about the metro area, which can be much larger. But the question specifically asks for the population of the capital, so I think it refers to the city limits. Still, I should make sure.<br><br>Another thing to consider is that population figures can vary depending on the source. Some might cite estimates from government agencies, while others might use more recent data from censuses or surveys. I should find a reliable source to get the most accurate number.<br><br>I think the population of Paris is around 21 million, but I'm not 100% sure. Maybe I should think about other major cities in France to compare. For example, Lyon is another big city, but it's much smaller. I believe its population is around 1.2 million. That gives me a sense that Paris is significantly larger.<br><br>Also, considering the economic activities in Paris, like the fashion industry and the entertainment sector, it makes sense that it's the capital and has a large population. The city hosts a lot of events, conventions, and businesses, which would attract a diverse population.<br><br>I should also think about the historical growth of Paris. It's been a major city for centuries, so its population has been increasing steadily. I think it's safe to say that it's over 20 million, but I'm still not certain about the exact number.<br><br>In summary, I'm pretty confident that the capital of France is Paris, and its population is around 21 million. However, to be precise, I should look up the latest statistics to confirm the exact figure.<br><br><br>content: {<br><br>"name": "Paris",<br>"population": 21538000<br>}</strong>


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


<strong style='color: #00008B;'>reasoing_content: Alright, the user is asking for the current date and time in New York and the weather there. I need to figure out how to get this information using the provided functions.<br><br>First, I should use the get_current_date function. It requires a timezone parameter. Since the user is in New York, the timezone is 'America/New_York'. I'll structure the function call with this parameter.<br><br>Next, for the weather, I'll use get_current_weather. The city is New York, the state is NY, and I'll default to Celsius since it's a common unit and the function allows it. I'll make sure to include all these parameters in the function call.<br><br>I should make sure each function call is separate and follows the required format. I'll write each function call on a new line, specifying the parameters correctly. Also, I'll include the sources where I got the function names from the provided data.<br><br>Double-checking the parameters to ensure they match the function descriptions. The timezone for date is correct, and the city, state, and unit for weather are all specified. I think that's all.<br><br><br>content: <function=get_current_date>{"timezone": "America/New_York"}</function>  <br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "celsius"}</function>  <br><br>Source:  <br>- Function descriptions provided in the instructions  <br>- get_current_date and get_current_weather function definitions</strong>


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

    {'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not exactly sure of the exact number. I think it\'s around 2 million, but I should double-check that.\n\nWait, maybe I should look up the latest statistics to be accurate. I recall that in recent years, Paris has seen a steady increase due to urban development and immigration. Let me see... I think the population is approximately 2,170,000 as of 2023. That seems about right.\n\nNow, the user wants this information in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to create a JSON object that includes the city name, its population, and maybe the country it\'s in. So, the keys could be "city", "population", and "country".\n\nPutting it all together, the JSON should look something like this: {"city": "Paris", "population": 2170000, "country": "France"}. I should make sure the syntax is correct, with proper commas and quotation marks.\n\nI should also consider if the user might need more details, like the metropolitan area population or other statistics, but since they specifically asked for the capital, I\'ll stick to that. Maybe mention that the data is up to date as of 2023 to provide context.\n\nAlright, I think that\'s all. I\'ll present the JSON and offer further assistance in case they need more information.\n</think>{\n  "name": "Paris",\n  "population": 2170000\n}', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 6896, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 1990, 15934, 429, 382, 14190, 11, 7196, 358, 1265, 1401, 705, 279, 5535, 13142, 311, 387, 13382, 13, 358, 19091, 429, 304, 3213, 1635, 11, 12095, 702, 3884, 264, 24020, 5263, 4152, 311, 15662, 4401, 323, 15093, 13, 6771, 752, 1490, 1112, 358, 1744, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 15, 11, 15, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 911, 1290, 382, 7039, 11, 279, 1196, 6801, 419, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 1616, 311, 5944, 821, 13, 358, 1184, 311, 1855, 264, 4718, 1633, 429, 5646, 279, 3283, 829, 11, 1181, 7042, 11, 323, 7196, 279, 3146, 432, 594, 304, 13, 2055, 11, 279, 6894, 1410, 387, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 11436, 97904, 432, 678, 3786, 11, 279, 4718, 1265, 1401, 2494, 1075, 419, 25, 5212, 8926, 788, 330, 59604, 497, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 11, 330, 11141, 788, 330, 49000, 1, 7810, 358, 1265, 1281, 2704, 279, 19482, 374, 4396, 11, 448, 6169, 76602, 323, 54231, 15423, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 803, 3565, 11, 1075, 279, 57406, 3082, 7042, 476, 1008, 13142, 11, 714, 2474, 807, 11689, 4588, 369, 279, 6722, 11, 358, 3278, 9214, 311, 429, 13, 10696, 6286, 429, 279, 821, 374, 705, 311, 2400, 438, 315, 220, 17, 15, 17, 18, 311, 3410, 2266, 382, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3042, 279, 4718, 323, 3010, 4623, 12994, 304, 1142, 807, 1184, 803, 1995, 624, 151649, 515, 220, 330, 606, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 17, 16, 22, 15, 15, 15, 15, 198, 92, 151643], 'meta_info': {'id': 'bc877e43e28a40bba540ed5b585fc56f', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 369, 'completion_tokens': 392, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 2.8291553324088454, 'response_sent_to_client_ts': 1776893146.2555177}}



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


<strong style='color: #00008B;'>{'text': 'Okay, so the user is asking for the information and population of the capital of France in JSON format. Let me break this down.\n\nFirst, I need to identify the capital of France. I know that Paris is the capital, so that\'s straightforward. Now, I should find the most recent population data. I remember that the population of Paris has been growing, but I\'m not sure of the exact number. I think it\'s around 2 million, but I should verify that.\n\nI\'ll check a reliable source, maybe the official Paris Municipality website or a recent census. Let me see... Yes, according to the latest data, the population is approximately 2,174,300 as of 2023. That seems accurate.\n\nNext, I need to structure this information into a JSON format. JSON requires key-value pairs, so I\'ll create an object with keys like "city", "population", and "country". The city is Paris, the population is the number I found, and the country is France.\n\nI should make sure the JSON syntax is correct. Each key should be in quotes, and the values as well. Also, the entire structure should be enclosed in curly braces. I\'ll format it properly to ensure there are no syntax errors.\n\nPutting it all together, the JSON object will have the city, population, and country. I\'ll present this to the user, making sure it\'s clear and easy to understand. I don\'t think the user needs anything else, but if they have more questions, I can provide additional information.\n\nI should also consider if the user might need the population figure in a different format, like a string or an integer, but since they asked for JSON, I\'ll stick with the number as an integer. \n\nAlright, I think that\'s all. I\'ll format the JSON correctly and present it to the user.\n</think>{\n\n"name": "Paris",\n"population": 2174300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'output_ids': [32313, 11, 773, 279, 1196, 374, 10161, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 382, 5338, 11, 358, 1184, 311, 10542, 279, 6722, 315, 9625, 13, 358, 1414, 429, 12095, 374, 279, 6722, 11, 773, 429, 594, 30339, 13, 4695, 11, 358, 1265, 1477, 279, 1429, 3213, 7042, 821, 13, 358, 6099, 429, 279, 7042, 315, 12095, 702, 1012, 7826, 11, 714, 358, 2776, 537, 2704, 315, 279, 4734, 1372, 13, 358, 1744, 432, 594, 2163, 220, 17, 3526, 11, 714, 358, 1265, 10146, 429, 382, 40, 3278, 1779, 264, 14720, 2530, 11, 7196, 279, 3946, 12095, 35703, 2719, 3910, 476, 264, 3213, 43602, 13, 6771, 752, 1490, 1112, 7414, 11, 4092, 311, 279, 5535, 821, 11, 279, 7042, 374, 13187, 220, 17, 11, 16, 22, 19, 11, 18, 15, 15, 438, 315, 220, 17, 15, 17, 18, 13, 2938, 4977, 13382, 382, 5847, 11, 358, 1184, 311, 5944, 419, 1995, 1119, 264, 4718, 3561, 13, 4718, 7460, 1376, 19083, 13530, 11, 773, 358, 3278, 1855, 458, 1633, 448, 6894, 1075, 330, 8926, 497, 330, 44441, 497, 323, 330, 11141, 3263, 576, 3283, 374, 12095, 11, 279, 7042, 374, 279, 1372, 358, 1730, 11, 323, 279, 3146, 374, 9625, 382, 40, 1265, 1281, 2704, 279, 4718, 19482, 374, 4396, 13, 8886, 1376, 1265, 387, 304, 17194, 11, 323, 279, 2750, 438, 1632, 13, 7281, 11, 279, 4453, 5944, 1265, 387, 43810, 304, 68103, 59191, 13, 358, 3278, 3561, 432, 10277, 311, 5978, 1052, 525, 902, 19482, 5975, 382, 97904, 432, 678, 3786, 11, 279, 4718, 1633, 686, 614, 279, 3283, 11, 7042, 11, 323, 3146, 13, 358, 3278, 3042, 419, 311, 279, 1196, 11, 3259, 2704, 432, 594, 2797, 323, 4135, 311, 3535, 13, 358, 1513, 944, 1744, 279, 1196, 3880, 4113, 770, 11, 714, 421, 807, 614, 803, 4755, 11, 358, 646, 3410, 5107, 1995, 382, 40, 1265, 1083, 2908, 421, 279, 1196, 2578, 1184, 279, 7042, 7071, 304, 264, 2155, 3561, 11, 1075, 264, 914, 476, 458, 7546, 11, 714, 2474, 807, 4588, 369, 4718, 11, 358, 3278, 9214, 448, 279, 1372, 438, 458, 7546, 13, 4710, 71486, 11, 358, 1744, 429, 594, 678, 13, 358, 3278, 3561, 279, 4718, 12440, 323, 3042, 432, 311, 279, 1196, 624, 151649, 4257, 1, 606, 788, 330, 59604, 756, 1, 44441, 788, 220, 17, 16, 22, 19, 18, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 'meta_info': {'id': 'ff7d7026556744abb6c1693c80f60e49', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 383, 'completion_tokens': 2048, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 15.239930912852287, 'response_sent_to_client_ts': 1776893161.5045474}}</strong>


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

    [{'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '61f6e1d6a8cb4beba01d966224b00349', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.08896859269589186, 'response_sent_to_client_ts': 1776893161.6160123}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '3dd052f1b7454819bd0458b1691cf791', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.08891701605170965, 'response_sent_to_client_ts': 1776893161.6160214}}, {'text': 'Berlin is the capital of France', 'output_ids': [3430, 81, 742, 77, 374, 279, 6722, 315, 9625, 151643], 'meta_info': {'id': '3d8c579d936446b3857dd39c26abdec7', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 11, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 10, 'cached_tokens': 10, 'cached_tokens_details': {'device': 10, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.0888677672483027, 'response_sent_to_client_ts': 1776893161.6160243}}]


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

    {'text': ' France, and the \n\\( n \\)  \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\( l \\) \\( m \\) \\( k \\) \\(', 'output_ids': [9625, 11, 323, 279, 220, 198, 44292, 308, 1124, 8, 220, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767, 326, 1124, 8, 17767, 296, 1124, 8, 17767, 595, 1124, 8, 17767], 'meta_info': {'id': 'a098de25b6624f7b824161c46b0fefda', 'finish_reason': {'type': 'length', 'length': 2048}, 'prompt_tokens': 6, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 2048, 'completion_tokens': 2048, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 21.123533317819238, 'response_sent_to_client_ts': 1776893182.7456002}}


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


<strong style='color: #00008B;'>{'text': 'Alright, the user has asked for the information and population of the capital of France in JSON format. Let me break this down. First, I need to identify what exactly they\'re looking for. The capital of France is Paris, so that\'s straightforward.\n\nI should include the population. I remember that the population of Paris is a bit under 4 million as of the last data I have, which is around 2023. But I should probably double-check that to make sure it\'s up to date. Let me verify the current population to ensure accuracy.\n\nNext, they requested the information in JSON format. JSON stands for JavaScript Object Notation, which is a lightweight data-interchange format that\'s easy for humans to read and write, and easy for machines to parse and generate. So, I\'ll need to structure the data accordingly. \n\nI think the user might need this for a quick data extraction or integration into another application, so including specific, accurate figures is crucial. Maybe they\'re compiling data for a report or a project, so precision matters.\n\nI should also make sure that the JSON is properly formatted to avoid any syntax errors. Let me recall the structure: key-value pairs within curly braces. The keys should be clear and relevant, like "capital" and "population". The population figure should be a number without any extra spaces to ensure it\'s a valid numeric value in JSON.\n\nIs there anything else the user might need? They might be looking for just the population number, but since they asked for information and population, including both aspects is better. Maybe they also need more details, but based on the question, two keys seem sufficient.\n\nI should also consider the possibility that the population data is approximate. Given that the user didn\'t specify a year, I\'ll go with the most recent, though I\'ll note that it\'s an estimate. Including a source isn\'t part of the query, so I\'ll stick to just providing the data in JSON without additional notes.\n\nIn summary, I\'ll create a JSON object with two keys: "capital" holding the string "Paris" and "population" with the number representing the approximate population. I\'ll ensure the formatting is correct and that the key-value pairs are clearly separated by commas if necessary.\n</think>\n\nHere is the information about the capital of France (Paris) in JSON format:\n\n```json\n{\n  "capital": "Paris",\n  "population": 1224123\n}\n```\n\nThis JSON object contains the following details:\n- `capital`: The name of the capital city of France.\n- `population`: An approximate population figure of the capital city.', 'output_ids': [71486, 11, 279, 1196, 702, 4588, 369, 279, 1995, 323, 7042, 315, 279, 6722, 315, 9625, 304, 4718, 3561, 13, 6771, 752, 1438, 419, 1495, 13, 5512, 11, 358, 1184, 311, 10542, 1128, 6896, 807, 2299, 3330, 369, 13, 576, 6722, 315, 9625, 374, 12095, 11, 773, 429, 594, 30339, 382, 40, 1265, 2924, 279, 7042, 13, 358, 6099, 429, 279, 7042, 315, 12095, 374, 264, 2699, 1212, 220, 19, 3526, 438, 315, 279, 1537, 821, 358, 614, 11, 892, 374, 2163, 220, 17, 15, 17, 18, 13, 1988, 358, 1265, 4658, 1990, 15934, 429, 311, 1281, 2704, 432, 594, 705, 311, 2400, 13, 6771, 752, 10146, 279, 1482, 7042, 311, 5978, 13403, 382, 5847, 11, 807, 11223, 279, 1995, 304, 4718, 3561, 13, 4718, 13352, 369, 12914, 3002, 2806, 367, 11, 892, 374, 264, 29144, 821, 44894, 3373, 3561, 429, 594, 4135, 369, 12677, 311, 1349, 323, 3270, 11, 323, 4135, 369, 12645, 311, 4715, 323, 6923, 13, 2055, 11, 358, 3278, 1184, 311, 5944, 279, 821, 27079, 13, 4710, 40, 1744, 279, 1196, 2578, 1184, 419, 369, 264, 3974, 821, 32189, 476, 17590, 1119, 2441, 3766, 11, 773, 2670, 3151, 11, 13382, 12396, 374, 16587, 13, 10696, 807, 2299, 54220, 821, 369, 264, 1895, 476, 264, 2390, 11, 773, 16052, 12850, 382, 40, 1265, 1083, 1281, 2704, 429, 279, 4718, 374, 10277, 23126, 311, 5648, 894, 19482, 5975, 13, 6771, 752, 19091, 279, 5944, 25, 1376, 19083, 13530, 2878, 68103, 59191, 13, 576, 6894, 1265, 387, 2797, 323, 9760, 11, 1075, 330, 65063, 1, 323, 330, 44441, 3263, 576, 7042, 7071, 1265, 387, 264, 1372, 2041, 894, 4960, 12621, 311, 5978, 432, 594, 264, 2697, 24064, 897, 304, 4718, 382, 3872, 1052, 4113, 770, 279, 1196, 2578, 1184, 30, 2379, 2578, 387, 3330, 369, 1101, 279, 7042, 1372, 11, 714, 2474, 807, 4588, 369, 1995, 323, 7042, 11, 2670, 2176, 13566, 374, 2664, 13, 10696, 807, 1083, 1184, 803, 3565, 11, 714, 3118, 389, 279, 3405, 11, 1378, 6894, 2803, 14016, 382, 40, 1265, 1083, 2908, 279, 13038, 429, 279, 7042, 821, 374, 44868, 13, 16246, 429, 279, 1196, 3207, 944, 13837, 264, 1042, 11, 358, 3278, 728, 448, 279, 1429, 3213, 11, 3498, 358, 3278, 5185, 429, 432, 594, 458, 16045, 13, 55121, 264, 2530, 4436, 944, 949, 315, 279, 3239, 11, 773, 358, 3278, 9214, 311, 1101, 8241, 279, 821, 304, 4718, 2041, 5107, 8388, 382, 641, 12126, 11, 358, 3278, 1855, 264, 4718, 1633, 448, 1378, 6894, 25, 330, 65063, 1, 9963, 279, 914, 330, 59604, 1, 323, 330, 44441, 1, 448, 279, 1372, 14064, 279, 44868, 7042, 13, 358, 3278, 5978, 279, 36566, 374, 4396, 323, 429, 279, 1376, 19083, 13530, 525, 9355, 18663, 553, 76602, 421, 5871, 624, 151649, 271, 8420, 374, 279, 1995, 911, 279, 6722, 315, 9625, 320, 59604, 8, 304, 4718, 3561, 1447, 73594, 2236, 198, 515, 220, 330, 65063, 788, 330, 59604, 756, 220, 330, 44441, 788, 220, 16, 17, 17, 19, 16, 17, 18, 198, 532, 13874, 19324, 1986, 4718, 1633, 5610, 279, 2701, 3565, 510, 12, 1565, 65063, 44622, 576, 829, 315, 279, 6722, 3283, 315, 9625, 624, 12, 1565, 44441, 44622, 1527, 44868, 7042, 7071, 315, 279, 6722, 3283, 13, 151643], 'meta_info': {'id': '449938f986ab44079189b4df69e42712', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 23, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 456, 'completion_tokens': 535, 'cached_tokens': 22, 'cached_tokens_details': {'device': 22, 'host': 0}, 'dp_rank': None, 'e2e_latency': 4.208292198833078, 'response_sent_to_client_ts': 1776893186.962875}}</strong>



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

    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-22 21:26:41] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.14s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.10s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.10s/it]


    2026-04-22 21:26:51,014 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 21:26:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:03,  3.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:03,  3.22s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:43,  1.85s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:43,  1.85s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:06,  1.20s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:06,  1.20s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:29,  1.79it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:29,  1.79it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:24,  2.12it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:24,  2.12it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:20,  2.45it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:20,  2.45it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:17,  2.85it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:17,  2.85it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:17,  2.85it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:10,  4.36it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:10,  4.36it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:10,  4.36it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:07,  5.90it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:07,  5.90it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:07,  5.90it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:05,  7.52it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:05,  7.52it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:05,  7.52it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:04,  9.41it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:04,  9.41it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:04,  9.41it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:04,  9.41it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:03, 12.58it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:03, 12.58it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:03, 12.58it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:03, 12.58it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:03, 12.58it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 17.61it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 17.61it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 17.61it/s]

    Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 17.61it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:07<00:01, 17.86it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:07<00:01, 17.86it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:07<00:01, 17.86it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:07<00:01, 17.86it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:01, 18.88it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:01, 18.88it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:01, 18.88it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:01, 18.88it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:07<00:01, 19.30it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:07<00:01, 19.30it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:07<00:01, 19.30it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:07<00:01, 19.30it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:01, 20.63it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:01, 20.63it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:01, 20.63it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:01, 20.63it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:07<00:00, 22.76it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:07<00:00, 22.76it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:07<00:00, 22.76it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:07<00:00, 22.76it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:08<00:00, 23.64it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:08<00:00, 23.64it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:08<00:00, 23.64it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:08<00:00, 23.64it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:08<00:00, 24.48it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:08<00:00, 24.48it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:08<00:00, 24.48it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:08<00:00, 24.48it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:08<00:00, 24.48it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 27.95it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 27.95it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 27.95it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 27.95it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 27.95it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:08<00:00, 27.95it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:08<00:00, 27.95it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:08<00:00, 27.95it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:08<00:00, 27.95it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:08<00:00, 27.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00, 44.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=60.95 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=60.95 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.92 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=60.92 GB):   3%|▎         | 2/58 [00:00<00:15,  3.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.92 GB):   3%|▎         | 2/58 [00:00<00:15,  3.57it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=60.92 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.92 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=60.92 GB):   7%|▋         | 4/58 [00:01<00:13,  4.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.93 GB):   7%|▋         | 4/58 [00:01<00:13,  4.08it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=60.93 GB):   9%|▊         | 5/58 [00:01<00:12,  4.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.93 GB):   9%|▊         | 5/58 [00:01<00:12,  4.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.93 GB):  10%|█         | 6/58 [00:01<00:10,  4.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.93 GB):  10%|█         | 6/58 [00:01<00:10,  4.73it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=60.93 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.94 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.94 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.94 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.60it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=60.94 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.94 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.94 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.94 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.51it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=60.94 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.93it/s]Capturing num tokens (num_tokens=3328 avail_mem=48.93 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.93it/s]Capturing num tokens (num_tokens=3328 avail_mem=48.93 GB):  21%|██        | 12/58 [00:02<00:06,  7.54it/s]Capturing num tokens (num_tokens=3072 avail_mem=45.92 GB):  21%|██        | 12/58 [00:02<00:06,  7.54it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=45.92 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.92 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.92 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.92 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.92 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.54it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=45.92 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=45.92 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.92 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.92 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.92 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.92 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.61it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=45.92 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.61it/s]Capturing num tokens (num_tokens=960 avail_mem=45.91 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.61it/s] Capturing num tokens (num_tokens=960 avail_mem=45.91 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.71it/s]Capturing num tokens (num_tokens=896 avail_mem=45.91 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.71it/s]Capturing num tokens (num_tokens=832 avail_mem=45.91 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.71it/s]Capturing num tokens (num_tokens=768 avail_mem=45.90 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.71it/s]Capturing num tokens (num_tokens=768 avail_mem=45.90 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.46it/s]Capturing num tokens (num_tokens=704 avail_mem=45.90 GB):  43%|████▎     | 25/58 [00:02<00:01, 18.46it/s]

    Capturing num tokens (num_tokens=640 avail_mem=45.90 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.46it/s]Capturing num tokens (num_tokens=576 avail_mem=45.89 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.46it/s]Capturing num tokens (num_tokens=576 avail_mem=45.89 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.87it/s]Capturing num tokens (num_tokens=512 avail_mem=45.89 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.87it/s]Capturing num tokens (num_tokens=480 avail_mem=45.89 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.87it/s]Capturing num tokens (num_tokens=448 avail_mem=45.88 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.87it/s]Capturing num tokens (num_tokens=416 avail_mem=45.88 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.87it/s]Capturing num tokens (num_tokens=416 avail_mem=45.88 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.22it/s]Capturing num tokens (num_tokens=384 avail_mem=45.88 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.22it/s]

    Capturing num tokens (num_tokens=352 avail_mem=45.87 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.22it/s]Capturing num tokens (num_tokens=320 avail_mem=45.87 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.22it/s]Capturing num tokens (num_tokens=288 avail_mem=45.86 GB):  55%|█████▌    | 32/58 [00:03<00:01, 24.22it/s]Capturing num tokens (num_tokens=288 avail_mem=45.86 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.06it/s]Capturing num tokens (num_tokens=256 avail_mem=45.86 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.06it/s]Capturing num tokens (num_tokens=240 avail_mem=45.85 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.06it/s]Capturing num tokens (num_tokens=224 avail_mem=45.85 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.06it/s]Capturing num tokens (num_tokens=208 avail_mem=45.85 GB):  62%|██████▏   | 36/58 [00:03<00:00, 27.06it/s]Capturing num tokens (num_tokens=208 avail_mem=45.85 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.72it/s]Capturing num tokens (num_tokens=192 avail_mem=45.84 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.72it/s]

    Capturing num tokens (num_tokens=176 avail_mem=45.84 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.72it/s]Capturing num tokens (num_tokens=160 avail_mem=45.84 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.72it/s]Capturing num tokens (num_tokens=144 avail_mem=45.83 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.72it/s]Capturing num tokens (num_tokens=144 avail_mem=45.83 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.75it/s]Capturing num tokens (num_tokens=128 avail_mem=45.84 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.75it/s]Capturing num tokens (num_tokens=112 avail_mem=45.84 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.75it/s]Capturing num tokens (num_tokens=96 avail_mem=45.83 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.75it/s] Capturing num tokens (num_tokens=80 avail_mem=45.83 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.75it/s]Capturing num tokens (num_tokens=80 avail_mem=45.83 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.24it/s]Capturing num tokens (num_tokens=64 avail_mem=45.83 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.24it/s]

    Capturing num tokens (num_tokens=48 avail_mem=45.83 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.24it/s]Capturing num tokens (num_tokens=32 avail_mem=45.82 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.24it/s]Capturing num tokens (num_tokens=28 avail_mem=45.82 GB):  83%|████████▎ | 48/58 [00:03<00:00, 33.24it/s]Capturing num tokens (num_tokens=28 avail_mem=45.82 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.16it/s]Capturing num tokens (num_tokens=24 avail_mem=45.81 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.16it/s]Capturing num tokens (num_tokens=20 avail_mem=45.81 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.16it/s]Capturing num tokens (num_tokens=16 avail_mem=45.81 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.16it/s]

    Capturing num tokens (num_tokens=12 avail_mem=45.80 GB):  90%|████████▉ | 52/58 [00:03<00:00, 34.16it/s]Capturing num tokens (num_tokens=12 avail_mem=45.80 GB):  97%|█████████▋| 56/58 [00:03<00:00, 30.19it/s]Capturing num tokens (num_tokens=8 avail_mem=45.80 GB):  97%|█████████▋| 56/58 [00:03<00:00, 30.19it/s] Capturing num tokens (num_tokens=4 avail_mem=45.80 GB):  97%|█████████▋| 56/58 [00:03<00:00, 30.19it/s]Capturing num tokens (num_tokens=4 avail_mem=45.80 GB): 100%|██████████| 58/58 [00:03<00:00, 14.52it/s]


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
    
    Generated text: Okay, so I need to figure out how to respond to this user's request. They asked for the information and population of the capital of France in JSON format. Let me break this down.
    
    First, I know that the capital of France is Paris. That's a fact I've heard before, so I'm pretty confident about that. Now, they want the population. I remember that Paris is a major city, so it's definitely a big population, but I'm not sure of the exact number. I think it's around 3 million, but I'm not certain. Maybe I should check if that's the most recent number or if it's slightly different now.
    
    Next, they specifically asked for the information and population in JSON format. JSON stands for JavaScript Object Notation, which is a way to structure data. I need to make sure that the response I give is in that format. So, I should create a JSON object with the key "capital" and a value that includes both the city name and the population.
    
    Wait, should I include both the name and the population in one value or separate them? I think it's better to have a single value that's an object itself, so that it's clear and organized. That way, if the user wants more details, they can access the population separately.
    
    So, putting it all together, the JSON structure would look like {"capital": {"city": "Paris", "population": 3000000}}. That seems straightforward. I should double-check if the population figure is accurate. Maybe I can recall that Paris is the most populous city in France, and its population has been increasing over the years. But I'm not sure if it's exactly 3 million or a bit more. For the sake of this response, I'll go with 3,000,000 unless I'm certain of a different number.
    
    I also need to make sure there are no typos in the JSON. JSON is case-sensitive, and missing a quote or bracket could cause issues. So I'll be careful to format it correctly. Once I have the JSON ready, I'll present it to the user. 
    
    I wonder if the user is a student working on a project, maybe for geography or a programming class. They might need this data for a report or a presentation. Alternatively, they could be developing an app that requires knowing the capital and population of France. Either way, providing accurate and well-structured data is important.
    
    I should also consider if there are any additional details they might find useful, but since they only asked for population, I'll stick to that. No need to include other cities or countries unless they specify. 
    
    To sum up, my response will be a JSON object with the key "capital" and a value that's another object containing the city name and population. I'll make sure the population number is accurate, and the formatting is correct. That should fulfill the user's request effectively.
    </think>
    
    ```json
    {
      "capital": {
        "city": "Paris",
        "population": 3000000
      }
    }
    ```



```python
llm.shutdown()
```
