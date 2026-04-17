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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-17 22:40:10] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]


    2026-04-17 22:40:15,412 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 22:40:15] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:23,  2.27it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.52it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.52it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.52it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.51it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.51it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.51it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.51it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.51it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.51it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.51it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.51it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.51it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.51it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.30it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.16it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.72it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.40 GB):   3%|▎         | 2/58 [00:00<00:03, 17.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.72it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.60it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.60it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 29.54it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 29.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 29.54it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.37it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.13it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.13it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.13it/s]

    Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  36%|███▌      | 21/58 [00:14<00:01, 34.13it/s]

    Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  40%|███▉      | 23/58 [00:17<00:59,  1.69s/it]Capturing num tokens (num_tokens=832 avail_mem=120.28 GB):  40%|███▉      | 23/58 [00:17<00:59,  1.69s/it]

    Capturing num tokens (num_tokens=832 avail_mem=120.28 GB):  41%|████▏     | 24/58 [00:17<00:51,  1.51s/it]Capturing num tokens (num_tokens=768 avail_mem=120.08 GB):  41%|████▏     | 24/58 [00:17<00:51,  1.51s/it]Capturing num tokens (num_tokens=704 avail_mem=119.42 GB):  41%|████▏     | 24/58 [00:17<00:51,  1.51s/it]Capturing num tokens (num_tokens=640 avail_mem=119.05 GB):  41%|████▏     | 24/58 [00:17<00:51,  1.51s/it]Capturing num tokens (num_tokens=640 avail_mem=119.05 GB):  47%|████▋     | 27/58 [00:17<00:31,  1.00s/it]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:17<00:31,  1.00s/it]

    Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  47%|████▋     | 27/58 [00:17<00:31,  1.00s/it]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:17<00:31,  1.00s/it]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  52%|█████▏    | 30/58 [00:17<00:19,  1.47it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  52%|█████▏    | 30/58 [00:17<00:19,  1.47it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  52%|█████▏    | 30/58 [00:18<00:19,  1.47it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  52%|█████▏    | 30/58 [00:18<00:19,  1.47it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  57%|█████▋    | 33/58 [00:18<00:11,  2.11it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:18<00:11,  2.11it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:18<00:11,  2.11it/s]

    Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:18<00:11,  2.11it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  57%|█████▋    | 33/58 [00:18<00:11,  2.11it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  64%|██████▍   | 37/58 [00:18<00:06,  3.28it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  64%|██████▍   | 37/58 [00:18<00:06,  3.28it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  64%|██████▍   | 37/58 [00:18<00:06,  3.28it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:18<00:06,  3.28it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:18<00:06,  3.28it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:18<00:06,  3.28it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  72%|███████▏  | 42/58 [00:18<00:03,  5.24it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  72%|███████▏  | 42/58 [00:18<00:03,  5.24it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:18<00:03,  5.24it/s]

    Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:18<00:03,  5.24it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:18<00:03,  5.24it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  72%|███████▏  | 42/58 [00:18<00:03,  5.24it/s] Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:18<00:01,  7.70it/s]Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:18<00:01,  7.70it/s]Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:18<00:01,  7.70it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:18<00:01,  7.70it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:18<00:01,  7.70it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  81%|████████  | 47/58 [00:18<00:01,  7.70it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:18<00:00, 10.72it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:18<00:00, 10.72it/s]

    Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:18<00:00, 10.72it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:18<00:00, 10.72it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:18<00:00, 10.72it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:18<00:00, 10.72it/s] Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  98%|█████████▊| 57/58 [00:18<00:00, 14.30it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  98%|█████████▊| 57/58 [00:18<00:00, 14.30it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:18<00:00,  3.11it/s]


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
    Generated text:  Xiao Jun, and I'm a junior college student. Recently, I have been learning a lot of information from the internet. I'm worried about the future. I think the internet will become the new era for everyone. Everyone will be online. Everyone's experience will change. This will change all the ways of life. I'm thinking of how I can make a change. I'm worried that I'm not ready for this. I think I should take action. I'll take action by learning things in class. I think I should also try online courses. What do you think? I'll share my thoughts with you.
    Answer the following
    ===============================
    Prompt: The president of the United States is
    Generated text:  paid 20% more than the secretary, and the secretary is paid 10% more than the assistant treasurer. If the assistant treasurer earns $6000 per month, how much does the president earn per month?
    
    To determine how much the president earns per month, we need to follow the given relationships step by step.
    
    1. Let's denote the amount the secretary earns per month as \( S \).
    2. According to the problem, the secretary earns 10% more than the assistant treasurer. Therefore, we can write the relationship for the secretary's earnings as:
       \[
       S = 60
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the second largest city in the world. Paris is the capital of France. The city is located in the center of the country and has been the capital since the 16th century. It is in the Île de la Cité, which is a large island, surrounded by the Seine river. Paris is famous for its world-famous monuments, such as the Eiffel Tower, the Louvre Museum, and Notre Dame Cathedral. It is also a cultural and historical center, known for its museums, theaters, and theaters. The city has a population of over 2.1 million people. It is an
    ===============================
    Prompt: The future of AI is
    Generated text:  here. AI has the potential to transform the world in some ways that are incredibly profound, such as revolutionizing healthcare, education, and transportation. However, as with any new technology, there are some challenges and concerns that need to be addressed before we can fully harness the potential of AI. One of the most significant challenges is how to ensure that AI systems are fair and unbiased, so that they do not perpetuate harmful biases and discrimination.
    
    One approach to this challenge is to implement transparency and accountability measures, such as data privacy laws and algorithmic accountability frameworks, to ensure that AI systems are transparent and explainable, allowing users to understand how


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I enjoy [reason for interest in the industry]. I'm always looking for new opportunities to [reason for interest in the industry]. I'm a [reason for interest in the industry] and I'm always eager to learn and grow. I'm a [reason for interest in the industry] and I'm always eager to learn and grow. I'm a [reason for interest in the industry] and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art scene, and cuisine, and is a major tourist destination. Paris is a city of contrasts, with its modern architecture and cultural attractions blending seamlessly with its historical landmarks. It is a city that has played a significant role in French history and continues to be a major cultural and economic center in the country. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that we interact with technology and the world around us. Here are some possible future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to improve, we are likely to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and productive machines that can perform tasks that were previously done by humans.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need for privacy and security. This will likely lead to the development of new technologies that can
    


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
    Generated text:  [name] and I am an [age] year old [gender] who specializes in [main skill or interest]. I bring a unique blend of [relevant skills, knowledge, or experiences] to the table, making me an asset to any team or organization. I am [career goal or project] and am always striving to [any accomplishment or achievement]. I have a [range of interests, hobbies, or passions], and I believe that [reason why you're passionate about your work]. My approach to work is [ideological or strategic]. I always aim to [positive or proactive] and work towards [future goal or accomplishment].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The statement is factual. Paris is the capital city of France, located on the North Bank of the Seine in the Île de France. It is the largest and most populous city in France and one of the largest in the world. It is known for its rich history, art, music, cuisine, fashion, and tourism. Paris is a cultural and political center that plays a vital role in French society and plays an important role in the country's economy. The city is home to many world-renowned landmarks such as the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and the Palace of Versailles
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some of the potential trends that are likely to shape it:
    
    1. Increased Integration with Human Intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence. This could lead to more advanced AI that can better understand and adapt to human behavior.
    
    2. Greater Use of AI for Military and Strategic Purposes: AI could be used to develop new military strategies, improve surveillance and reconnaissance, and improve decision-making in combat. This could lead to significant changes in military operations and the way we perceive the world.
    
    3. AI in Healthcare: AI could be used to develop more accurate and personalized


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

    Name

    ],

     and

     I

     am

     [

    Age

    ].

     I

     am

     an

     [

    occupation

    /

    field

    ]

     with

     a

     [

    technical

     skill

     or

     special

     ability

    ]

     that

     sets

     me

     apart

     from

     the

     rest

    .

     I

     am

     fluent

     in

     [

    language

     or

     dialect

    ]

     and

     [

    language

     or

     dialect

    ].

     I

     have

     been

     learning

     about

     [

    topic

    ]

     and

     am

     always

     looking

     for

     ways

     to

     [

    describe

     how

     I

     can

     make

     a

     difference

     in

     the world

    ].

     I

     am

     excited

     to

     [

    what

     you

     can

     do

     for

     the

     company

    ].

     I

     look

     forward

     to

     [

    starting

    ]

     and

     am

     always

     eager

     to

     learn

    .

     


    Your

     job

     is

     to

     write

     an

     introduction

     that

     is

     both

     short

     and

     neutral

    .

     It

     should

     start

     with

     your

     name

     and

     age

    ,

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     as

     the

     "

    City

     of

     Light

    "

     for

     its

     vibrant

     culture

     and

     arts

     scene

    .

     
    


    A

     Conc

    ise

     Fact

     Statement

    :

     Paris

    ,

     the

     capital

     of

     France

    ,

     is

     renowned

     for

     its

     diverse

     cultural

     and

     artistic

     activities

    ,

     with

     Paris

     Museum

     of

     Modern

     Art

     (

    Pal

    ais

     Garn

    ier

    )

     and

     the

     Lou

    vre

     Museum

     being

     among

     the

     most

     famous

     landmarks

    .

     
    


    This

     brief

     statement

     encaps

    ulates

     the

     essence

     of

     Paris

    ,

     including

     its

     role

     as

     the

     nation

    's

     capital

    ,

     its

     cultural

     prominence

    , and

     its

     iconic

     landmarks

    .

     It

    's

     a

     concise

     yet

     comprehensive

     summary

     of

     Paris

    's

     key

     attributes

    .

     
    


    Please

     note

     that

     while

     the

     statement

     is

     concise

    ,

     it

     does

     not

     capture

     all

     of

     Paris

    's

     aspects

     or

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     with

     many

     potential

     trends

     that

     could

     shape

     the

     way

     we

     live

    ,

     work

    ,

     and

     interact

     with

     technology

    .

     Here

     are

     some

     of

     the

     most

     likely

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Autonomous

     vehicles

    :

     With

     the

     increase

     in

     autonomous

     vehicle

     technology

    ,

     it

     is

     expected

     that

     many

     cities

     will

     see

     a

     reduction

     in

     traffic

     congestion

     and

     pollution

    .

     However

    ,

     this

     technology

     may

     also

     create

     new

     challenges

    ,

     such

     as

     the

     need

     for

     a

     more

     efficient

     and

     reliable

     way

     to

     deliver

     goods

     and

     services

    .
    


    2

    .

     Emotional

     intelligence

    :

     AI

     technology

     is

     becoming

     more

     capable

     of

     understanding

     and

     processing

     human

     emotions

    ,

     allowing

     it

     to

     become

     more

     intelligent

     and

     empath

    etic

    .

     This

     could

     lead

     to

     more

     effective

     and

     compassionate

     decision

    



```python
llm.shutdown()
```
