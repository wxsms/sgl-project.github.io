# Offline Engine API

SGLang provides a direct inference engine without the need for an HTTP server, especially for use cases where additional HTTP server adds unnecessary complexity or overhead. Here are two general use cases:

- Offline Batch Inference
- Custom Server on Top of the Engine

This document focuses on the offline batch inference, demonstrating four different inference modes:

- Non-streaming synchronous generation
- Streaming synchronous generation
- Non-streaming asynchronous generation
- Streaming asynchronous generation

Additionally, you can easily build a custom server on top of the SGLang offline engine. A detailed example working in a python script can be found in [custom_server](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/custom_server.py).



## Nest Asyncio
Note that if you want to use **Offline Engine** in ipython or some other nested loop code, you need to add the following code:
```python
import nest_asyncio

nest_asyncio.apply()

```

## Advanced Usage

The engine supports [vlm inference](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference_vlm.py) as well as [extracting hidden states](https://github.com/sgl-project/sglang/blob/main/examples/runtime/hidden_states). 

Please see [the examples](https://github.com/sgl-project/sglang/tree/main/examples/runtime/engine) for further use cases.

## Offline Batch Inference

SGLang offline engine supports batch inference with efficient scheduling.


```python
# launch the offline engine
import asyncio

import sglang as sgl
import sglang.test.doc_patch  # noqa: F401
from sglang.utils import async_stream_and_merge, stream_and_merge

llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
```

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.16it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.16it/s]


    2026-04-06 07:17:35,392 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 07:17:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]

    Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:27,  1.99it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:27,  1.99it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:27,  1.99it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:15,  3.29it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:15,  3.29it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:15,  3.29it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:10,  4.75it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:10,  4.75it/s]

    Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:10,  4.75it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:07,  6.49it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:07,  6.49it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:07,  6.49it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:07,  6.49it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:04,  9.20it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:04,  9.20it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:04,  9.20it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:04,  9.20it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 12.03it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 12.03it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 12.03it/s]

    Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 12.03it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 14.39it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 14.39it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 14.39it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 14.39it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:02, 16.77it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:02, 16.77it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:02, 16.77it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 16.77it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:01, 19.29it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:01, 19.29it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:01, 19.29it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:01, 19.29it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 21.21it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 21.21it/s]

    Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 21.21it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 21.21it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 22.94it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 22.94it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 22.94it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 22.94it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 22.94it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:00, 26.46it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:00, 26.46it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:00, 26.46it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:00, 26.46it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:04<00:00, 26.46it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 29.26it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 29.26it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 29.26it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 29.26it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 29.26it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:04<00:00, 31.96it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:04<00:00, 31.96it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:04<00:00, 31.96it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:04<00:00, 31.96it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:04<00:00, 31.96it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 33.81it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 33.81it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 33.81it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 33.81it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 33.81it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 33.81it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 37.55it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 37.55it/s]

    Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 37.55it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 37.55it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 37.55it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 37.55it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 37.55it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 42.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=50.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=50.14 GB):   2%|▏         | 1/58 [00:00<00:06,  8.44it/s]Capturing num tokens (num_tokens=7680 avail_mem=50.11 GB):   2%|▏         | 1/58 [00:00<00:06,  8.44it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=50.11 GB):   3%|▎         | 2/58 [00:00<00:06,  8.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.11 GB):   3%|▎         | 2/58 [00:00<00:06,  8.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.11 GB):   5%|▌         | 3/58 [00:00<00:07,  7.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=49.54 GB):   5%|▌         | 3/58 [00:00<00:07,  7.19it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=49.54 GB):   7%|▋         | 4/58 [00:00<00:08,  6.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.08 GB):   7%|▋         | 4/58 [00:00<00:08,  6.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.08 GB):   9%|▊         | 5/58 [00:00<00:08,  6.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=50.07 GB):   9%|▊         | 5/58 [00:00<00:08,  6.05it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=50.07 GB):  10%|█         | 6/58 [00:00<00:08,  6.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=49.64 GB):  10%|█         | 6/58 [00:00<00:08,  6.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=49.64 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.07 GB):  12%|█▏        | 7/58 [00:01<00:08,  6.19it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=50.07 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=49.67 GB):  14%|█▍        | 8/58 [00:01<00:07,  6.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=49.67 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=50.07 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.68it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=50.07 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.67it/s]Capturing num tokens (num_tokens=3584 avail_mem=50.06 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.67it/s]Capturing num tokens (num_tokens=3584 avail_mem=50.06 GB):  19%|█▉        | 11/58 [00:01<00:06,  6.91it/s]Capturing num tokens (num_tokens=3328 avail_mem=49.72 GB):  19%|█▉        | 11/58 [00:01<00:06,  6.91it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=49.72 GB):  21%|██        | 12/58 [00:01<00:06,  7.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.05 GB):  21%|██        | 12/58 [00:01<00:06,  7.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.05 GB):  22%|██▏       | 13/58 [00:01<00:06,  7.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=50.05 GB):  22%|██▏       | 13/58 [00:01<00:06,  7.37it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=50.05 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=49.77 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=49.77 GB):  26%|██▌       | 15/58 [00:02<00:05,  7.98it/s]Capturing num tokens (num_tokens=2304 avail_mem=50.04 GB):  26%|██▌       | 15/58 [00:02<00:05,  7.98it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=50.04 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=50.04 GB):  28%|██▊       | 16/58 [00:02<00:05,  8.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=50.04 GB):  29%|██▉       | 17/58 [00:02<00:04,  8.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=49.80 GB):  29%|██▉       | 17/58 [00:02<00:04,  8.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=50.03 GB):  29%|██▉       | 17/58 [00:02<00:04,  8.41it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=50.03 GB):  33%|███▎      | 19/58 [00:02<00:04,  9.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=50.02 GB):  33%|███▎      | 19/58 [00:02<00:04,  9.30it/s]Capturing num tokens (num_tokens=1024 avail_mem=50.00 GB):  33%|███▎      | 19/58 [00:02<00:04,  9.30it/s]Capturing num tokens (num_tokens=1024 avail_mem=50.00 GB):  36%|███▌      | 21/58 [00:02<00:03, 10.13it/s]Capturing num tokens (num_tokens=960 avail_mem=49.83 GB):  36%|███▌      | 21/58 [00:02<00:03, 10.13it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=50.01 GB):  36%|███▌      | 21/58 [00:02<00:03, 10.13it/s]Capturing num tokens (num_tokens=896 avail_mem=50.01 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.73it/s]Capturing num tokens (num_tokens=832 avail_mem=50.00 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.73it/s]Capturing num tokens (num_tokens=768 avail_mem=50.00 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.73it/s]

    Capturing num tokens (num_tokens=768 avail_mem=50.00 GB):  43%|████▎     | 25/58 [00:03<00:02, 11.52it/s]Capturing num tokens (num_tokens=704 avail_mem=49.84 GB):  43%|████▎     | 25/58 [00:03<00:02, 11.52it/s]Capturing num tokens (num_tokens=640 avail_mem=49.98 GB):  43%|████▎     | 25/58 [00:03<00:02, 11.52it/s]Capturing num tokens (num_tokens=640 avail_mem=49.98 GB):  47%|████▋     | 27/58 [00:03<00:02, 11.86it/s]Capturing num tokens (num_tokens=576 avail_mem=49.98 GB):  47%|████▋     | 27/58 [00:03<00:02, 11.86it/s]

    Capturing num tokens (num_tokens=512 avail_mem=49.97 GB):  47%|████▋     | 27/58 [00:03<00:02, 11.86it/s]Capturing num tokens (num_tokens=512 avail_mem=49.97 GB):  50%|█████     | 29/58 [00:03<00:02, 12.23it/s]Capturing num tokens (num_tokens=480 avail_mem=49.98 GB):  50%|█████     | 29/58 [00:03<00:02, 12.23it/s]Capturing num tokens (num_tokens=448 avail_mem=49.97 GB):  50%|█████     | 29/58 [00:03<00:02, 12.23it/s]

    Capturing num tokens (num_tokens=448 avail_mem=49.97 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.18it/s]Capturing num tokens (num_tokens=416 avail_mem=49.94 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.18it/s]Capturing num tokens (num_tokens=384 avail_mem=49.94 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.18it/s]Capturing num tokens (num_tokens=384 avail_mem=49.94 GB):  57%|█████▋    | 33/58 [00:03<00:02, 11.37it/s]Capturing num tokens (num_tokens=352 avail_mem=49.94 GB):  57%|█████▋    | 33/58 [00:03<00:02, 11.37it/s]

    Capturing num tokens (num_tokens=320 avail_mem=49.95 GB):  57%|█████▋    | 33/58 [00:03<00:02, 11.37it/s]Capturing num tokens (num_tokens=320 avail_mem=49.95 GB):  60%|██████    | 35/58 [00:03<00:01, 12.46it/s]Capturing num tokens (num_tokens=288 avail_mem=49.90 GB):  60%|██████    | 35/58 [00:03<00:01, 12.46it/s]Capturing num tokens (num_tokens=256 avail_mem=49.93 GB):  60%|██████    | 35/58 [00:03<00:01, 12.46it/s]Capturing num tokens (num_tokens=256 avail_mem=49.93 GB):  64%|██████▍   | 37/58 [00:03<00:01, 13.11it/s]Capturing num tokens (num_tokens=240 avail_mem=49.93 GB):  64%|██████▍   | 37/58 [00:03<00:01, 13.11it/s]

    Capturing num tokens (num_tokens=224 avail_mem=49.94 GB):  64%|██████▍   | 37/58 [00:04<00:01, 13.11it/s]Capturing num tokens (num_tokens=224 avail_mem=49.94 GB):  67%|██████▋   | 39/58 [00:04<00:01, 13.98it/s]Capturing num tokens (num_tokens=208 avail_mem=49.92 GB):  67%|██████▋   | 39/58 [00:04<00:01, 13.98it/s]Capturing num tokens (num_tokens=192 avail_mem=49.91 GB):  67%|██████▋   | 39/58 [00:04<00:01, 13.98it/s]Capturing num tokens (num_tokens=192 avail_mem=49.91 GB):  71%|███████   | 41/58 [00:04<00:01, 14.25it/s]Capturing num tokens (num_tokens=176 avail_mem=49.91 GB):  71%|███████   | 41/58 [00:04<00:01, 14.25it/s]

    Capturing num tokens (num_tokens=160 avail_mem=49.90 GB):  71%|███████   | 41/58 [00:04<00:01, 14.25it/s]Capturing num tokens (num_tokens=160 avail_mem=49.90 GB):  74%|███████▍  | 43/58 [00:04<00:01, 14.93it/s]Capturing num tokens (num_tokens=144 avail_mem=49.90 GB):  74%|███████▍  | 43/58 [00:04<00:01, 14.93it/s]Capturing num tokens (num_tokens=128 avail_mem=49.89 GB):  74%|███████▍  | 43/58 [00:04<00:01, 14.93it/s]Capturing num tokens (num_tokens=128 avail_mem=49.89 GB):  78%|███████▊  | 45/58 [00:04<00:00, 14.72it/s]Capturing num tokens (num_tokens=112 avail_mem=49.88 GB):  78%|███████▊  | 45/58 [00:04<00:00, 14.72it/s]

    Capturing num tokens (num_tokens=96 avail_mem=49.87 GB):  78%|███████▊  | 45/58 [00:04<00:00, 14.72it/s] Capturing num tokens (num_tokens=96 avail_mem=49.87 GB):  81%|████████  | 47/58 [00:04<00:00, 15.10it/s]Capturing num tokens (num_tokens=80 avail_mem=49.87 GB):  81%|████████  | 47/58 [00:04<00:00, 15.10it/s]Capturing num tokens (num_tokens=64 avail_mem=49.87 GB):  81%|████████  | 47/58 [00:04<00:00, 15.10it/s]Capturing num tokens (num_tokens=64 avail_mem=49.87 GB):  84%|████████▍ | 49/58 [00:04<00:00, 15.50it/s]Capturing num tokens (num_tokens=48 avail_mem=49.87 GB):  84%|████████▍ | 49/58 [00:04<00:00, 15.50it/s]

    Capturing num tokens (num_tokens=32 avail_mem=49.86 GB):  84%|████████▍ | 49/58 [00:04<00:00, 15.50it/s]Capturing num tokens (num_tokens=32 avail_mem=49.86 GB):  88%|████████▊ | 51/58 [00:04<00:00, 15.94it/s]Capturing num tokens (num_tokens=28 avail_mem=49.85 GB):  88%|████████▊ | 51/58 [00:04<00:00, 15.94it/s]Capturing num tokens (num_tokens=24 avail_mem=49.85 GB):  88%|████████▊ | 51/58 [00:04<00:00, 15.94it/s]Capturing num tokens (num_tokens=24 avail_mem=49.85 GB):  91%|█████████▏| 53/58 [00:04<00:00, 16.56it/s]Capturing num tokens (num_tokens=20 avail_mem=49.84 GB):  91%|█████████▏| 53/58 [00:04<00:00, 16.56it/s]

    Capturing num tokens (num_tokens=16 avail_mem=49.84 GB):  91%|█████████▏| 53/58 [00:05<00:00, 16.56it/s]Capturing num tokens (num_tokens=16 avail_mem=49.84 GB):  95%|█████████▍| 55/58 [00:05<00:00, 17.13it/s]Capturing num tokens (num_tokens=12 avail_mem=49.83 GB):  95%|█████████▍| 55/58 [00:05<00:00, 17.13it/s]Capturing num tokens (num_tokens=8 avail_mem=49.82 GB):  95%|█████████▍| 55/58 [00:05<00:00, 17.13it/s] Capturing num tokens (num_tokens=8 avail_mem=49.82 GB):  98%|█████████▊| 57/58 [00:05<00:00, 17.32it/s]Capturing num tokens (num_tokens=4 avail_mem=49.82 GB):  98%|█████████▊| 57/58 [00:05<00:00, 17.32it/s]

    Capturing num tokens (num_tokens=4 avail_mem=49.82 GB): 100%|██████████| 58/58 [00:05<00:00, 11.06it/s]


### Non-streaming Synchronous Generation


```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

    ===============================
    Prompt: Hello, my name is
    Generated text:  Peter and I'm the CEO of Small World. I specialize in building and managing scalable, intelligent, and secure solutions for complex problems.
    My background spans over 20 years in leadership roles in the product development, innovation, and business development domains. This includes my work with product management, marketing, business development, sales, and sales. I’ve been involved in building some of the world’s most successful and popular product lines, including Cellbec, PPD, and USB Station.
    I’m a fan of tech, travel, photography, and music, and I’m passionate about sharing my passion for these things. I have a master’s
    ===============================
    Prompt: The president of the United States is
    Generated text:  a(n) ______. A. president of the United States B. head of government C. cabinet D. premier.
    Answer: B
    
    Which of the following groups of words has all correct pronunciations for the highlighted characters?
    A. Dizzy (xuàn) Limp (pín) Prayer (qí)
    B. Shackle (gù) Sudden death (cù) Heat and cold (zhà)
    C. Restrained (shè) Vast (pù) Spring water (qiàn)
    D. Submerged (hùn) Despise (zè) Cloudy (c
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Lyon
    C. Lille
    D. Marseille
    Answer:
    A
    
    [Multiple Choice Question] Which of the following statements about the management of official documents is correct? 
    A. It is the specific embodiment of the management activities carried out by the State Council in its daily work. 
    B. The implementation of the State Council's document management work is carried out by the State Council itself. 
    C. The State Council is responsible for the management of the country's official documents. 
    D. The State Council is responsible for the management of the country's financial budgetary documents. 
    Answer:
    
    ===============================
    Prompt: The future of AI is
    Generated text:  highly complex, involving multiple layers and directions. One of the more exciting and potentially revolutionary areas of AI is called "deep learning," which involves the use of large neural networks to perform tasks that are currently handled by traditional machine learning methods. Deep learning has the potential to revolutionize a wide range of industries, from healthcare to finance to transportation. However, the process of building and training deep learning models can be complex and challenging, and there is a high risk of overfitting and model instability.
    One of the key challenges in building and training deep learning models is the need for a large and diverse dataset. This can be difficult to obtain,


### Streaming Synchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {
    "temperature": 0.2,
    "top_p": 0.9,
}

print("\n=== Testing synchronous streaming generation with overlap removal ===\n")

for prompt in prompts:
    print(f"Prompt: {prompt}")
    merged_output = stream_and_merge(llm, prompt, sampling_params)
    print("Generated text:", merged_output)
    print()
```

    
    === Testing synchronous streaming generation with overlap removal ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your profession or role]. I enjoy [insert a brief description of your hobbies or interests]. What's your favorite [insert a short description of your hobby or interest]. I'm always looking for new experiences and learning opportunities. What's your favorite [insert a short description of your hobby or interest]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite [insert a short description of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its cuisine, fashion, and art scene. Paris is a cultural and economic hub of France and the world, and it is a popular tourist destination. The city is home to many famous landmarks and attractions, including the Louvre, the Eiffel Tower, and the Champs-Élysées. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could emerge in the coming years:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and preferences.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI
    


### Non-streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous batch generation ===")


async def main():
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")


asyncio.run(main())
```

    
    === Testing asynchronous batch generation ===


    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text:  [Name], and I'm a [job title] at [company name]. My [job title] has been with [company name] since [start date]. I've always been a [specific skill or trait] in my field, and I love [job title]. I enjoy [job title] and I'm always striving to [job title] in my work. I'm always eager to learn new things and improve my skills. I'm always looking for the next step in my career and I'm always willing to pursue it. My goals are to [goal 1], [goal 2], and [goal 3].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The capital of France is Paris. Paris is the largest and most populous city in France, and it is located on the Seine River in the northern suburbs of the capital city of Paris, the city of Paris. Paris is known for its rich history, beautiful architecture, and vibrant culture, and it is the heart of the French capital. It is also the birthplace of many famous people, including Émile Zola and Victor Hugo. Paris is home to the Eiffel Tower and the Louvre Museum, among other iconic landmarks. The city is known for its annual celebrations, including the Eiffel Tower Parcours
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a wide range of trends and developments that could shape the direction of this field. Some possible future trends include:
    
    1. Increased automation: AI is increasingly being used to automate repetitive tasks, allowing humans to focus on more complex and creative work.
    
    2. Enhanced cognitive abilities: AI is expected to continue to develop and improve its ability to think and learn, making it more capable of performing tasks that were previously thought to be too complex for machines to handle.
    
    3. Integration with other technologies: AI is expected to become more integrated with other technologies, such as the Internet of Things (IoT), enabling more intelligent and connected devices.
    
    


### Streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous streaming generation (no repeats) ===")


async def main():
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated text: ", end="", flush=True)

        # Replace direct calls to async_generate with our custom overlap-aware version
        async for cleaned_chunk in async_stream_and_merge(llm, prompt, sampling_params):
            print(cleaned_chunk, end="", flush=True)

        print()  # New line after each prompt


asyncio.run(main())
```

    
    === Testing asynchronous streaming generation (no repeats) ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text: 

     [

    Character

    's

     name

    ],

     and

     I

    'm

     here

     to

     meet

     you

    .

     I

    'm

     from

     [

    birth

    place

    ],

     and

     I

    've

     been

     [

    number

     of

     years

    ]

     years

     of

     age

     in

     this

     city

    .

     I

    'm

     a

     [

    occupation

    ]

     with

     a

     strong

     work

     ethic

     and

     dedication

     to

     [

    something

     specific

    ].

     If

     you

     have

     any

     questions

     or

     would

     like

     to

     chat

     about

     something

    ,

     please

     don

    't

     hesitate

     to

     reach

     out

    .

     I

     look

     forward

     to

     meeting

     you

    .

     [

    Character

    's

     name

    ].

     

    📞

    ✨

    ✨

    🌟

    📖

    ✨

    📖

    ✨

    ✨

    ✨

    ✨

    🌟

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    📖

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     North

    -East

     of

     the

     country

    .

     The

     city

     is

     a

     major

     cultural

     and

     economic

     hub

    ,

     with

     a

     rich

     history

     dating

     back

     to

     ancient

     times

    .

     Paris

     is

     known

     for

     its

     iconic

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     home

     to

     many

     famous

     artists

    ,

     such

     as

     Picasso

     and

     Da

     Vinci

    ,

     and

     its

     cuisine

     is

     a

     major

     factor

     in

     the

     country

    's

     culinary

     tradition

    .

     Paris

     has

     a

     diverse

     and

     multicultural

     population

    ,

     with

     many

     people

     coming

     from

     all

     over

     the

     world

    .

     It

     is

     the

     most

     visited

     city

     in

     the

     world

     and

     is

     considered

     to

     be

     one

     of

     the

     world

    's

     most

     famous

     cities

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     increasing

     sophistication

    ,

     integration

     with

     other

     technologies

    ,

     and

     more

     natural

     and

     human

    -like

     interactions

     with

     humans

    .

     Here

     are

     some

     potential

     trends

    :
    


    1

    .

     Improved

     accuracy

     and

     reliability

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     will

     be

     able

     to

     perform

     more

     complex

     tasks

     and

     generate

     more

     accurate

     results

    .

     This

     will

     lead

     to

     more

     reliable

     and

     trustworthy

     AI

     applications

    .
    


    2

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     will

     be

     increasingly

     integrated

     with

     other

     technologies

     such

     as

     sensors

    ,

     cameras

    ,

     and

     IoT

     devices

     to

     create

     more

     sophisticated

     and

     intelligent

     systems

    .
    


    3

    .

     Personal

    ization

     and

     contextual

     awareness

    :

     AI

     systems

     will

     be

     able

     to

     learn

     from

     user

     data

     and

     context

     to

     provide

     more

     personalized

     and

     context

    



```python
llm.shutdown()
```
