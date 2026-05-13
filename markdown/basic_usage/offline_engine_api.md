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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.44it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.44it/s]


    2026-05-13 01:17:20,036 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 01:17:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.61it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.61it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.61it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.61it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  7.00it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  7.00it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  7.00it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  7.00it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  7.00it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.76it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.76it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.76it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.76it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.76it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.56it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.56it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.56it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.56it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.56it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 14.56it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 19.62it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 19.62it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 19.62it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 19.62it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 19.62it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 23.13it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 23.13it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 23.13it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 23.13it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 23.13it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 23.13it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 32.99it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 32.99it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 32.99it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 32.99it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 32.99it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 32.99it/s]

    Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 32.99it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 38.50it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 38.50it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 38.50it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 38.50it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 38.50it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 38.50it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 38.50it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 42.57it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 42.57it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 42.57it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 42.57it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 42.57it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:06<00:00, 42.57it/s] 

    Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:06<00:00, 42.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.45 GB):   2%|▏         | 1/58 [00:00<00:07,  7.65it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.65it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   3%|▎         | 2/58 [00:00<00:07,  7.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.62it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   5%|▌         | 3/58 [00:00<00:07,  7.62it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   7%|▋         | 4/58 [00:00<00:07,  7.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.42 GB):   9%|▊         | 5/58 [00:00<00:06,  7.94it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):   9%|▊         | 5/58 [00:00<00:06,  7.94it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.41 GB):  10%|█         | 6/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  10%|█         | 6/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.55it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.55it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.40 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.41it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.31it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.31it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.17it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 11.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  26%|██▌       | 15/58 [00:01<00:03, 11.81it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.35it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  29%|██▉       | 17/58 [00:01<00:03, 12.35it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  33%|███▎      | 19/58 [00:01<00:02, 13.67it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=58.34 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.14it/s]Capturing num tokens (num_tokens=960 avail_mem=58.36 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.14it/s] Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.14it/s]Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.83it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.83it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  40%|███▉      | 23/58 [00:02<00:02, 14.83it/s]

    Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:02<00:02, 14.88it/s]Capturing num tokens (num_tokens=704 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 14.88it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  43%|████▎     | 25/58 [00:02<00:02, 14.88it/s]Capturing num tokens (num_tokens=640 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:01, 15.88it/s]Capturing num tokens (num_tokens=576 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:02<00:01, 15.88it/s]Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  47%|████▋     | 27/58 [00:02<00:01, 15.88it/s]

    Capturing num tokens (num_tokens=512 avail_mem=58.33 GB):  50%|█████     | 29/58 [00:02<00:01, 15.11it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:02<00:01, 15.11it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:02<00:01, 15.11it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 16.13it/s]Capturing num tokens (num_tokens=416 avail_mem=58.34 GB):  53%|█████▎    | 31/58 [00:02<00:01, 16.13it/s]Capturing num tokens (num_tokens=384 avail_mem=58.33 GB):  53%|█████▎    | 31/58 [00:02<00:01, 16.13it/s]

    Capturing num tokens (num_tokens=384 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:02<00:01, 17.04it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:02<00:01, 17.04it/s]Capturing num tokens (num_tokens=320 avail_mem=58.32 GB):  57%|█████▋    | 33/58 [00:02<00:01, 17.04it/s]Capturing num tokens (num_tokens=320 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 17.81it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 17.81it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  60%|██████    | 35/58 [00:02<00:01, 17.81it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:01, 18.27it/s]Capturing num tokens (num_tokens=240 avail_mem=58.32 GB):  64%|██████▍   | 37/58 [00:02<00:01, 18.27it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:02<00:01, 18.27it/s]Capturing num tokens (num_tokens=224 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:02<00:01, 18.60it/s]Capturing num tokens (num_tokens=208 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:02<00:01, 18.60it/s]Capturing num tokens (num_tokens=192 avail_mem=58.31 GB):  67%|██████▋   | 39/58 [00:02<00:01, 18.60it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  67%|██████▋   | 39/58 [00:03<00:01, 18.60it/s]

    Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.71it/s]Capturing num tokens (num_tokens=160 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.71it/s]Capturing num tokens (num_tokens=144 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.71it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.71it/s]Capturing num tokens (num_tokens=128 avail_mem=58.30 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.44it/s]Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.44it/s]Capturing num tokens (num_tokens=96 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.44it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  78%|███████▊  | 45/58 [00:03<00:00, 20.44it/s]Capturing num tokens (num_tokens=80 avail_mem=58.29 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.88it/s]Capturing num tokens (num_tokens=64 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.88it/s]Capturing num tokens (num_tokens=48 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.88it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  83%|████████▎ | 48/58 [00:03<00:00, 20.88it/s]Capturing num tokens (num_tokens=32 avail_mem=58.28 GB):  88%|████████▊ | 51/58 [00:03<00:00, 21.13it/s]Capturing num tokens (num_tokens=28 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:03<00:00, 21.13it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.27 GB):  88%|████████▊ | 51/58 [00:03<00:00, 21.13it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  88%|████████▊ | 51/58 [00:03<00:00, 21.13it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.88it/s]Capturing num tokens (num_tokens=16 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.88it/s]Capturing num tokens (num_tokens=12 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.88it/s]Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  93%|█████████▎| 54/58 [00:03<00:00, 21.88it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=58.26 GB):  98%|█████████▊| 57/58 [00:03<00:00, 22.38it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB):  98%|█████████▊| 57/58 [00:03<00:00, 22.38it/s]Capturing num tokens (num_tokens=4 avail_mem=58.25 GB): 100%|██████████| 58/58 [00:03<00:00, 15.33it/s]


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
    Generated text:  David. I’m a 28 year old student from Loma Linda, California, United States. I graduated from San Jose State University with a major in Mechanical Engineering and I am now taking a part time job to make money. My first job is a sales job with a company called Alcatel. I can get a 40 hour work week. The pay is about $20 an hour. My second job is a part time job at a small non-profit organization that is providing a full time job for people who have lost their jobs. I earn about $25 an hour. I am interested in all the career
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the government of the country. He is in charge of making decisions and taking care of the country. He is also in charge of the country's money. He is the president of the United States.
    What kind of job is the president of the United States?
    The president of the United States is a very important person in the government of the country. He is in charge of making decisions and taking care of the country. He is also in charge of the country's money. He is the head of state and the head of government of the United States. In short, the president of the United States is the most
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the seat of the French government, and also of the city’s most important universities, music, art, theater, and ballet.
    The city is the capital of a region known as Île-de-France, which also includes the Île-de-France metropolitan area and, together with the rest of France, is part of the Euro-Atlantic area. The Île-de-France is separated from the rest of France by the Seine river, which flows between the Seine and the Garonne river to the south.
    A major thoroughfare, the Seine, has been a part of Paris since ancient times.
    ===============================
    Prompt: The future of AI is
    Generated text:  still in the works. While the technology has been around for decades, the rise of big data and the proliferation of sensors and mobile devices has made it a powerful tool for the real-world. AI has a great deal of promise, but it also has a lot of risks. One of the biggest is the risk of cyberattacks. In the past, companies have found it necessary to implement anti-virus software to prevent attacks. Now, however, AI has the potential to do the same thing. The problem is, the more sophisticated an AI, the more difficult it is to defend against the attacks. In addition, the more data the AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic, or neutral statement about yourself]. I enjoy [insert a short, positive, enthusiastic, or neutral statement about your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short, positive, enthusiastic, or neutral statement about your favorite activity]. I'm always looking for new ways to challenge myself and expand
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also home to many famous museums, including the Musée d'Orsay, the Musée Rodin, and the Musée d'Orsay. Paris is a cultural and economic center of France and a major tourist destination. It is also known for its rich history, including the French Revolution and the French Revolution. The city is home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more sophisticated, there will be a greater emphasis on ethical considerations and responsible use of AI. This will include issues such as bias, transparency, and accountability.
    
    2. Greater integration with other technologies: AI is already being integrated into a wide range of other technologies, such
    


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
    Generated text:  [Name] and I'm a [job title], [Job Title]. I've always loved [specific field or profession], and I'm passionate about [specific activity or aspect of the profession]. I'm always looking for new opportunities to [specific achievement or skill], and I'm eager to learn and grow in [specific area of the profession]. I'm a [type of person], and I'm comfortable in [specific setting or environment], and I'm always up for [specific challenge or adventure]. What can you tell me about yourself?
    [Name], my enthusiastic personality, creative mind, and dedication to my profession make me an ideal candidate
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the most populous city in the European Union and the largest city in Europe. The city has a rich history and culture, with landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also known as the city of love and a great place to visit for tourists. Paris is a center for art, fashion, music, and literature, and is home to many museums, theaters, and parks. The city is also known for its fashion and food scene, with Parisian cuisine being considered one of the best in the world. Paris is a historic city with a long history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be a combination of several trends that will drive innovation and advancement in the field. Some of the potential future trends in AI include:
    
    1. Increased focus on ethical AI: With the increasing concern about AI's potential impact on society, there is a growing push towards developing AI that is more ethical and accountable. This could involve developing systems that can be programmed to identify and mitigate bias in AI algorithms.
    
    2. More integration with human intelligence: AI is increasingly being integrated with human intelligence to improve its performance and capabilities. For example, chatbots and virtual assistants can be trained to understand natural language and respond in a way that is more human


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

     Sarah

    .

     I

    'm

     a

     college

     student

     major

    ing

     in

     business

     and

     psychology

    .

     I

     love

     spending

     my

     free

     time

     playing

     sports

    ,

     especially

     basketball

     and

     soccer

    ,

     and

     I

     enjoy

     hiking

     and

     exploring

     new

     places

    .

     I

     also

     enjoy

     reading

     and

     learning

     new

     things

    ,

     especially

     in

     fields

     like

     science

     and

     technology

    .

     I

    'm

     passionate

     about

     helping

     people

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     and

     grow

    .

     What

     else

     can

     I

     say

    ?

     Welcome

    ,

     Sarah

    !

     You

    're

     a

     well

    -rounded

     and

     intelligent

     person

     with

     a

     love

     for

     various

     activities

    ,

     which

     makes

     you

     a

     versatile

     character

    .

     Let

    's

     see

     how

     you

     can

     make

     your

     self

    -int

    roduction

     more

     neutral

     and

     impactful

    .

     Let

    's

     brainstorm

     some

     more

     creative

     ways

     to

     introduce

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    .

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     on

     the

     North

     Bank

     of

     the

     Se

    ine

     River

     in

     the

     Paris

     Region

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     population

    ,

     with

     an

     estimated

     population

     of

     over

     

    2

    .

    2

     million

     people

     as

     of

     

    2

    0

    2

    1

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     vibrant

     cultural

     scene

    .

     Paris

     is

     a

     major

     international

     hub

     for

     art

    ,

     fashion

    ,

     and

     entertainment

    ,

     with

     numerous

     museums

    ,

     theaters

    ,

     and

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Lou

    vre

     Museum

    .

     It

     is

     also

     home

     to

     the

     French

     Parliament

    ,

     the

     É

    lys

    ée

     Palace

    ,

     and

     the

     Lou

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     advancements

     in

     areas

     such

     as

     machine

     learning

    ,

     computer

     vision

    ,

     natural

     language

     processing

    ,

     and

     robotics

    .

     Some

     possible

     future

     trends

     include

    :
    


    1

    .

     Increased

     efficiency

     and

     accuracy

     in

     AI

    -based

     systems

    .

     As

     AI

     technology

     continues

     to

     evolve

    ,

     we

     can

     expect

     to

     see

     even

     more

     advanced

     algorithms

     that

     can

     process

     and

     analyze

     vast

     amounts

     of

     data

     more

     quickly

     and

     accurately

     than

     ever

     before

    .
    


    2

    .

     AI

    -driven

     decision

    -making

     and

     automation

    .

     AI

     is

     increasingly

     being

     used

     to

     automate

     routine

     and

     repetitive

     tasks

    ,

     from

     customer

     service

     to

     manufacturing

     to

     transportation

    .

     This

     could

     lead

     to

     the

     automation

     of

     jobs

     and

     the

     creation

     of

     new

     industries

     that

     require

     AI

     expertise

    .
    


    3

    .

     Increased

     integration

     with

     human

    



```python
llm.shutdown()
```
