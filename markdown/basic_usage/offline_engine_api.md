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
    [2026-04-20 11:09:50] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.88it/s]


    2026-04-20 11:09:54,917 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-20 11:09:54] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.59it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.59it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.59it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.59it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.59it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.59it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.59it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.59it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.59it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.59it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.37it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.37it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.37it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.37it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.37it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.37it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.37it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.37it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.37it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.37it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.91it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.28 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.28 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.27 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.27 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.28 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.28 GB):   9%|▊         | 5/58 [00:00<00:02, 20.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.27 GB):   9%|▊         | 5/58 [00:00<00:02, 20.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.27 GB):   9%|▊         | 5/58 [00:00<00:02, 20.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.27 GB):   9%|▊         | 5/58 [00:00<00:02, 20.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.27 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.27 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.26 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.26 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=116.26 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.26 GB):  21%|██        | 12/58 [00:00<00:01, 28.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.25 GB):  21%|██        | 12/58 [00:00<00:01, 28.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.25 GB):  21%|██        | 12/58 [00:00<00:01, 28.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.24 GB):  21%|██        | 12/58 [00:00<00:01, 28.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.24 GB):  21%|██        | 12/58 [00:00<00:01, 28.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.24 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.23 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.18 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.24it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.16 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.24it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=116.16 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.24it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.14 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.24it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.14 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.87it/s]Capturing num tokens (num_tokens=960 avail_mem=116.15 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.87it/s] Capturing num tokens (num_tokens=896 avail_mem=116.15 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.87it/s]Capturing num tokens (num_tokens=832 avail_mem=116.15 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.87it/s]Capturing num tokens (num_tokens=768 avail_mem=116.14 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.87it/s]Capturing num tokens (num_tokens=768 avail_mem=116.14 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.44it/s]Capturing num tokens (num_tokens=704 avail_mem=116.14 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.44it/s]Capturing num tokens (num_tokens=640 avail_mem=116.14 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.44it/s]

    Capturing num tokens (num_tokens=576 avail_mem=116.14 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.44it/s]Capturing num tokens (num_tokens=512 avail_mem=116.10 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.44it/s]Capturing num tokens (num_tokens=512 avail_mem=116.10 GB):  50%|█████     | 29/58 [00:00<00:00, 34.51it/s]Capturing num tokens (num_tokens=480 avail_mem=116.10 GB):  50%|█████     | 29/58 [00:00<00:00, 34.51it/s]Capturing num tokens (num_tokens=448 avail_mem=116.10 GB):  50%|█████     | 29/58 [00:00<00:00, 34.51it/s]Capturing num tokens (num_tokens=416 avail_mem=116.10 GB):  50%|█████     | 29/58 [00:00<00:00, 34.51it/s]Capturing num tokens (num_tokens=384 avail_mem=116.09 GB):  50%|█████     | 29/58 [00:01<00:00, 34.51it/s]

    Capturing num tokens (num_tokens=384 avail_mem=116.09 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.92it/s]Capturing num tokens (num_tokens=352 avail_mem=116.09 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.92it/s]Capturing num tokens (num_tokens=320 avail_mem=116.08 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.92it/s]

    Capturing num tokens (num_tokens=288 avail_mem=116.08 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.92it/s]Capturing num tokens (num_tokens=256 avail_mem=116.08 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.92it/s]Capturing num tokens (num_tokens=256 avail_mem=116.08 GB):  64%|██████▍   | 37/58 [00:01<00:01, 18.19it/s]Capturing num tokens (num_tokens=240 avail_mem=116.07 GB):  64%|██████▍   | 37/58 [00:01<00:01, 18.19it/s]Capturing num tokens (num_tokens=224 avail_mem=116.07 GB):  64%|██████▍   | 37/58 [00:01<00:01, 18.19it/s]Capturing num tokens (num_tokens=208 avail_mem=116.07 GB):  64%|██████▍   | 37/58 [00:01<00:01, 18.19it/s]

    Capturing num tokens (num_tokens=208 avail_mem=116.07 GB):  69%|██████▉   | 40/58 [00:01<00:00, 19.04it/s]Capturing num tokens (num_tokens=192 avail_mem=116.07 GB):  69%|██████▉   | 40/58 [00:01<00:00, 19.04it/s]Capturing num tokens (num_tokens=176 avail_mem=116.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 19.04it/s]Capturing num tokens (num_tokens=160 avail_mem=116.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 19.04it/s]Capturing num tokens (num_tokens=160 avail_mem=116.06 GB):  74%|███████▍  | 43/58 [00:01<00:00, 19.70it/s]Capturing num tokens (num_tokens=144 avail_mem=116.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 19.70it/s]Capturing num tokens (num_tokens=128 avail_mem=116.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 19.70it/s]

    Capturing num tokens (num_tokens=112 avail_mem=116.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 19.70it/s]Capturing num tokens (num_tokens=112 avail_mem=116.05 GB):  79%|███████▉  | 46/58 [00:01<00:00, 21.31it/s]Capturing num tokens (num_tokens=96 avail_mem=116.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 21.31it/s] Capturing num tokens (num_tokens=80 avail_mem=116.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 21.31it/s]Capturing num tokens (num_tokens=64 avail_mem=116.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 21.31it/s]Capturing num tokens (num_tokens=48 avail_mem=116.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 21.31it/s]Capturing num tokens (num_tokens=48 avail_mem=116.04 GB):  86%|████████▌ | 50/58 [00:02<00:00, 24.66it/s]Capturing num tokens (num_tokens=32 avail_mem=116.03 GB):  86%|████████▌ | 50/58 [00:02<00:00, 24.66it/s]Capturing num tokens (num_tokens=28 avail_mem=116.03 GB):  86%|████████▌ | 50/58 [00:02<00:00, 24.66it/s]

    Capturing num tokens (num_tokens=24 avail_mem=116.02 GB):  86%|████████▌ | 50/58 [00:02<00:00, 24.66it/s]Capturing num tokens (num_tokens=20 avail_mem=116.02 GB):  86%|████████▌ | 50/58 [00:02<00:00, 24.66it/s]Capturing num tokens (num_tokens=20 avail_mem=116.02 GB):  93%|█████████▎| 54/58 [00:02<00:00, 27.29it/s]Capturing num tokens (num_tokens=16 avail_mem=116.02 GB):  93%|█████████▎| 54/58 [00:02<00:00, 27.29it/s]Capturing num tokens (num_tokens=12 avail_mem=116.01 GB):  93%|█████████▎| 54/58 [00:02<00:00, 27.29it/s]Capturing num tokens (num_tokens=8 avail_mem=116.01 GB):  93%|█████████▎| 54/58 [00:02<00:00, 27.29it/s] Capturing num tokens (num_tokens=4 avail_mem=116.01 GB):  93%|█████████▎| 54/58 [00:02<00:00, 27.29it/s]Capturing num tokens (num_tokens=4 avail_mem=116.01 GB): 100%|██████████| 58/58 [00:02<00:00, 28.47it/s]Capturing num tokens (num_tokens=4 avail_mem=116.01 GB): 100%|██████████| 58/58 [00:02<00:00, 25.80it/s]


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
    Generated text:  Aden. I'm a 28 year old Mexican-American girl. I've lived in China for 6 years. I'm currently living in South China. I'm studying in a Chinese university. I'm very interested in learning more about Chinese culture and history.
    What are some things that I can do to enhance my learning experience in China? I've been to a few universities in China for a few months now and have been studying Chinese, so I'm very familiar with the culture and language.
    I'm interested in learning more about Chinese history and culture, but I don't have any knowledge of Chinese history and feel like I might
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking government official, a member of the executive branch of the government of the United States, and has the authority to make decisions concerning the day-to-day operations and policies of the United States government. Where does the president of the United States reside? United States
    You are a helpful assistant, equalizer. The answer to this question is:
    The president of the United States resides in the White House. 
    
    The White House is the official residence and headquarters of the President of the United States. It is the executive mansion of the Executive Branch of the federal government. The building, designed by Frank Lloyd Wright and completed in 19
    ===============================
    Prompt: The capital of France is
    Generated text:  the capital of a country that has a capital of another country. What is the capital of the given country?
    
    A. Italy  
    B. Spain  
    C. Greece  
    D. Portugal
    
    To determine the capital of the country that has a capital of another country, let's first identify the countries involved and then follow the chain of capitals.
    
    The countries mentioned are:
    
    1. France
    2. Italy
    3. Spain
    4. Greece
    5. Portugal
    
    Now, let's follow the chain of capitals:
    
    1. The capital of France is Paris.
    2. The capital of Paris is Lille.
    3. The capital of L
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable, but one thing is certain: AI is converging with the cloud to the point where cloud infrastructure is poised to become the AI operating environment of choice.
    
    This is a relatively new reality, and it’s not something that exists in the traditional sense – it is an emergent phenomenon that has occurred through evolution.
    
    The word “AI” in the era of the cloud is a very broad one, and it covers all the technologies that allow for AI to perform tasks, but the cloud plays a pivotal role in the convergence of these technologies.
    
    The cloud technology supports the development and deployment of AI systems, enabling these systems to harness power from cloud


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. The city is known for its rich history, art, and culture, and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major transportation hub, with many major highways and rail lines connecting it to other parts of France and the world. The city is a popular tourist destination, with millions of visitors each year. Paris is a cultural and artistic center, with many museums, galleries, and theaters, and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration of AI into everyday life: As AI becomes more integrated into our daily lives, we are likely to see more widespread adoption of AI technologies. This could include things like smart homes, self-driving cars, and virtual assistants that can assist with tasks like grocery shopping or scheduling appointments.
    
    2. Greater emphasis on ethical AI: As AI becomes more advanced, there will be a greater emphasis on ethical considerations. This could include things like ensuring
    


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
    Generated text:  [Name] and I'm a computer programmer. I specialize in developing software solutions for all types of industries. I'm always looking for new challenges and have a passion for helping people learn new technologies. I enjoy working in a team environment and always strive to make the workplace a positive place to work. I love playing video games and attending conferences, and I'm always up for learning new programming languages. If you're interested in getting my contact info, I'm happy to share it with you. I look forward to connecting with you! [Name] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Account] [GitHub
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Roie" in French, and is the largest city in France by population. It is home to numerous attractions and landmarks, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and various museums such as the Musée d’Orsay and Musée Rodin. Paris is also renowned for its vibrant cultural scene and has a long history and legacy as a major European city. The city has been inhabited for thousands of years and is a cultural center in the world. It is often referred to as "the city of love" due to its romantic history and romantic atmosphere. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and exciting, with many possible trends to look out for. Here are a few possible future trends in AI:
    
    1. Increased accuracy of AI: As AI systems get more complex, they will become even more accurate and precise in their decision-making. This will enable more intelligent and efficient applications, such as autonomous vehicles and healthcare diagnostics.
    
    2. AI can learn from its experiences: AI will be able to learn from its interactions with humans and adapt to new situations based on its experiences. This will lead to more personalized and effective applications, such as personalized medicine and virtual assistants.
    
    3. AI will become more transparent: With the increasing amount


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

     John

     and

     I

    'm

     a

     dedicated

     reader

     and

     writer

     of

     fiction

    .

     I

    've

     always

     been

     fascinated

     by

     the

     world

     of

     books

     and

     I

    'm

     always

     on

     the

     lookout

     for

     new

     ideas

     and

     inspiration

    .

     I

     enjoy

     exploring

     different

     genres

     and

     learning

     about

     different

     writers

     and

     their

     work

    .

     My

     writing

     style

     is

     simple

     and

     easy

     to

     read

    ,

     with

     a

     focus

     on

     vivid

     descriptions

     and

     compelling

     narratives

    .

     I

    'm

     constantly

     trying

     to

     improve

     my

     skills

     and

     expand

     my

     knowledge

     in

     the

     field

     of

     fiction

    .

     I'm

     excited

     to

     share

     my

     thoughts

     and

     experiences

     with

     you

     and

     I

    'm

     looking

     forward

     to

     meeting

     you

    !

     To

     ensure

     a

     smooth

     and

     enjoyable

     experience

     for

     you

    ,

     I

     hope

     you

    'll

     give

     me

     a

     chance

     to

     introduce

     myself

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ville

    -Mar

    ie

    ."


    Paris

     is

     the

     largest

     city

     in

     France

     and

     is

     also

     the

     capital

    .

     It

     has

     a

     rich

     history

     dating

     back

     to

     the

     Middle

     Ages

    ,

     and

     it

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

     and

     the

     E

    iff

    el

     Tower

    .

     The

     city

     is

     also

     known

     for

     its

     beautiful

     architecture

     and

     cuisine

    ,

     and

     it

     is

     a

     major

     tourist

     destination

    .

     Paris

     has

     a

     diverse

     population

     of

     around

     

    2

    .

    5

     million

     people

     and

     is

     home

     to

     many

     cultural

     and

     artistic

     institutions

    ,

     including

     the

     Lou

    vre

     Museum

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     Overall

    ,

     Paris

     is

     a

     city

     of

     contrasts

     and

     beauty

     that

     is

     a

     significant

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     marked

     by

     significant

     advancements

     and

     innovations

     across

     multiple

     domains

    ,

     driven

     by

     the

     rapid

     growth

     of

     data

     and

     computing

     power

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     concerns

     about

     AI

    's

     potential

     impact

     on

     society

     continue

     to

     grow

    ,

     there

     will

     be

     increased

     focus

     on

     addressing

     ethical

     and

     social

     issues

     related

     to

     AI

    .

     This

     could

     include

     developing

     AI

     that

     is

     more

     transparent

    ,

     accountable

    ,

     and

     responsible

     for

     its

     decisions

    .
    


    2

    .

     Rise

     of

     AI

    -powered

     autonomous

     vehicles

    :

     As

     autonomous

     vehicles

     become

     more

     prevalent

     in

     society

    ,

     we

     will

     see

     an

     increased

     focus

     on

     developing

     AI

    -powered

     autonomous

     vehicles

     that

     can

     navigate

    复杂

    地形

    、

    实现

    精准

    的

    驾驶

    、

    



```python
llm.shutdown()
```
