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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


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
    [2026-04-24 09:52:00] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.59it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.57it/s]


    2026-04-24 09:52:04,615 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 09:52:04] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.54it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.54it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.50it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.50it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.50it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.50it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.50it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.50it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.50it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.50it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.50it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.50it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.18it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.18it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.18it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.18it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.18it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.18it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.18it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.18it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.18it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.18it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 38.98it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.47it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=136.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=136.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=136.72 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=136.71 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=136.71 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=136.71 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=136.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=136.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=136.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=136.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.17it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=136.69 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=136.69 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=136.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=136.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=136.66 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=3328 avail_mem=136.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=135.12 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=135.12 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.67it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=134.95 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=134.95 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=134.95 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.00it/s]Capturing num tokens (num_tokens=2048 avail_mem=134.95 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=134.94 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.26 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=123.27 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=119.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.44it/s]Capturing num tokens (num_tokens=960 avail_mem=119.90 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.44it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=112.83 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.44it/s]Capturing num tokens (num_tokens=896 avail_mem=112.83 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=832 avail_mem=106.11 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=768 avail_mem=105.26 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=704 avail_mem=105.26 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=640 avail_mem=105.26 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=640 avail_mem=105.26 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.94it/s]Capturing num tokens (num_tokens=576 avail_mem=105.26 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.94it/s]Capturing num tokens (num_tokens=512 avail_mem=105.25 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.94it/s]Capturing num tokens (num_tokens=480 avail_mem=105.26 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.94it/s]

    Capturing num tokens (num_tokens=448 avail_mem=105.26 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.94it/s]Capturing num tokens (num_tokens=448 avail_mem=105.26 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=416 avail_mem=105.26 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=384 avail_mem=105.26 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=352 avail_mem=105.25 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=320 avail_mem=105.25 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=320 avail_mem=105.25 GB):  60%|██████    | 35/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=288 avail_mem=105.24 GB):  60%|██████    | 35/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=256 avail_mem=105.24 GB):  60%|██████    | 35/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=240 avail_mem=105.24 GB):  60%|██████    | 35/58 [00:01<00:00, 33.23it/s]

    Capturing num tokens (num_tokens=224 avail_mem=105.23 GB):  60%|██████    | 35/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=224 avail_mem=105.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=208 avail_mem=105.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=192 avail_mem=105.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=176 avail_mem=105.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=160 avail_mem=105.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=160 avail_mem=105.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.46it/s]Capturing num tokens (num_tokens=144 avail_mem=105.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.46it/s]Capturing num tokens (num_tokens=128 avail_mem=105.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.46it/s]Capturing num tokens (num_tokens=112 avail_mem=105.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.46it/s]

    Capturing num tokens (num_tokens=96 avail_mem=105.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.46it/s] Capturing num tokens (num_tokens=96 avail_mem=105.21 GB):  81%|████████  | 47/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=80 avail_mem=105.21 GB):  81%|████████  | 47/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=64 avail_mem=105.20 GB):  81%|████████  | 47/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=48 avail_mem=105.20 GB):  81%|████████  | 47/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=32 avail_mem=105.19 GB):  81%|████████  | 47/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=32 avail_mem=105.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.13it/s]Capturing num tokens (num_tokens=28 avail_mem=105.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.13it/s]Capturing num tokens (num_tokens=24 avail_mem=105.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.13it/s]Capturing num tokens (num_tokens=20 avail_mem=105.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.13it/s]

    Capturing num tokens (num_tokens=16 avail_mem=105.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.13it/s]Capturing num tokens (num_tokens=12 avail_mem=105.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.13it/s]Capturing num tokens (num_tokens=12 avail_mem=105.18 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=8 avail_mem=105.18 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.12it/s] Capturing num tokens (num_tokens=4 avail_mem=105.17 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=4 avail_mem=105.17 GB): 100%|██████████| 58/58 [00:01<00:00, 30.97it/s]


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
    Generated text:  Lin. I'm a 15-year-old girl. I like taking pictures. Here are some photos from my recent pictures. This is my friend. She is five years old. She likes red. And she wants to wear her red dress. This is my cousin. She is my second cousin. She is thirteen. She likes pink. She wants to wear her pink dress. My parents buy some beautiful things for me. I love them. What do you like? I like reading. I love to draw too. But I don't like to watch TV. I have a lot of books and magazines. And I love to play
    ===============================
    Prompt: The president of the United States is
    Generated text:  a 43rd member of the U.S. House of Representatives. After his election, the president is sworn in as a member of the U.S. House of Representatives. The president of the United States can be elected to two terms of office.
    
    What is the first term of office for the president of the United States? The first term of office for the president of the United States is two years. 
    
    To break it down further:
    
    1. The president serves a two-year term.
    2. Two terms of office are typically completed each Congress.
    3. The president's term of office is not perpetual but ends upon the conclusion
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Rome
    D. Madrid
    Answer: A
    
    The capital of France is:
    A. Paris
    B. London
    C. Rome
    D. Madrid
    Answer: A
    
    The capital of France is:
    A. Paris
    B. London
    C. Rome
    D. Madrid
    
    Answer: A
    
    What is the capital of New Zealand?
    A. Wellington
    B. Auckland
    C. Hamilton
    D. Taupo
    Answer: A
    
    The capital of New Zealand is:
    A. Wellington
    B. Auckland
    C. Hamilton
    D. Taupo
    Answer:
    ===============================
    Prompt: The future of AI is
    Generated text:  looking a lot like the future of the internet and the internet of things. However, it will be a very different future. AI (Artificial Intelligence) will be everywhere, and in all areas of life. Think about a project you've been working on, or a project that you've done that you would like to see become a real-world technology. What can you do to make that project a success? Consider the impact of AI on the future, and how it can impact your project.
    AI is a rapidly growing field that is making a significant impact in many areas, including healthcare, transportation, finance, and more. However, it


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title] at [company name]. I'm always looking for ways to [job title] and [job title] at [company name], and I'm always eager to learn and grow. What's your favorite hobby or activity? I'm always looking for new experiences and adventures, and I'm always eager to try new things. What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its rich history, beautiful architecture, and vibrant culture. It is the largest city in France and is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its food, fashion, and music scene, making it a popular destination for tourists and locals alike. The city is also home to many important institutions such as the French Academy of Sciences and the French National Library. Overall, Paris is a city that is a true reflection of France's rich history and culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more and more AI systems become involved in decision-making processes, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the potential for AI to be used for malicious purposes.
    
    2. Integration of AI with other technologies: AI is likely to become more integrated with other technologies, such as blockchain, IoT, and the Internet of Things (IoT). This will enable AI to be
    


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
    Generated text:  [Name], and I'm a [Job Title/Role] at [Company Name]. I'm passionate about [Your Passion for the Role], and I'm always looking to learn and grow. I'm committed to [Your Commitment to the Role], and I'm always willing to put in the extra effort to achieve the best results. What makes you unique and what do you look forward to the most in your work? [Your Unique Feature or Passion/Commitment]. What's next on your to-do list? [Your Next Step]. What's your favorite way to relax after a long day at work? [Your Favorite Relaxation
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its towering Eiffel Tower, iconic landmarks such as the Louvre Museum, and a rich cultural heritage.
    
    That's a great summary! Could you provide some additional information about Paris's history and significance? Sure! Paris has a rich history dating back thousands of years, with evidence of prehistoric human habitation found in nearby sites like the Pit of Pithom and the Egyptian pyramids. The city has been an important center for trade and culture for centuries, and has played a crucial role in shaping French history and identity. Today, Paris remains a major center for politics, culture, and entertainment, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to evolve, incorporating new technologies and applications that will revolutionize the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Integration of AI with other technologies: AI will continue to be integrated with other technologies, such as IoT (Internet of Things), machine learning, and blockchain, to create more complex and powerful AI systems.
    
    2. Enhanced cognitive capabilities: AI will continue to gain new capabilities, such as better self-awareness, empathy, and emotional intelligence, to improve our ability to interact with people and machines.
    
    3. Personalized AI: As AI systems become more sophisticated


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

    ]

     and

     I

    'm

     a

     [

    Age

    ]

     [

    Gender

    ]

     who

     has

     [

    职

    ]

     in

     [

    Position

    ].

     [

    Name

    ]

     is

     a

     [

    job

     title

    ]

     at

     [

    Company

     name

    ]

     [

    Name

    ]

     is

     known

     for

     [

    summary

     of

     what

     you

     do

     at

     work

    ].

     What

     kind

     of

     personality

     traits

     do

     you

     possess

     that

     make

     you

     stand

     out

     to

     your

     coworkers

     and

     clients

    ?

     In

     your

     personal

     life

    ,

     what

     are

     you

     most

     passionate

     about

    ,

     and

     what

     are

     your

     hobbies

    ?

     Lastly

    ,

     what

     motiv

    ates

     you

     to

     make

     a

     difference

     in

     the

     world

    ?


    [

    Name

    ]

     (

    Name

    )

     is

     a

     [

    Name

    ]

     at

     [

    Company

     name

    ]

     [

    Name

    ]

     is

     known

     for

     [

    summary

     of

     what

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     an

     important

     city

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     vibrant

     culture

    .

     It

     is

     the

     largest

     city

     in

     the

     country

     and

     hosts

     numerous

     famous

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     known

     for

     its

     extensive

     food

     scene

    ,

     with

     a

     wide

     variety

     of

     French

     cuisine

     and

     restaurants

     serving

     international

     dishes

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     art

     galleries

    ,

     and

     theaters

    ,

     making

     it

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     Overall

    ,

     Paris

     is

     a

     city

     of

     contrasts

    ,

     offering

     a

     unique

     experience

     for

     visitors

     from

     around

     the

     world

    .

     
    


    This

     statement

     encaps

    ulates

     Paris

    's

     significance

     as

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     poised

     to

     be

     one

     of

     the

     most

     transformative

     and

     exciting

     in

     recent

     history

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     AI

     ethics

    :

     The

     AI

     industry

     is

     increasingly

     recognized

     for

     its

     potential

     to

     address

     ethical

     issues

     related

     to

     AI

    ,

     such

     as

     bias

    ,

     privacy

    ,

     and

     transparency

    .

     As

     more

     ethical

     guidelines

     are

     developed

    ,

     AI

     systems

     will

     become

     more

     transparent

     and

     accountable

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     becoming

     more

     common

     and

     widespread

    ,

     with

     many

     companies

     developing

     AI

    -powered

     vehicles

     for

     private

     and

     public

     use

    .

     This

     will

     likely

     result

     in

     the

     creation

     of

     a

     new

     industry

     focused

     on

     the

     development

     and

     deployment

     of

     AI

    -powered

     vehicles

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     is

    



```python
llm.shutdown()
```
