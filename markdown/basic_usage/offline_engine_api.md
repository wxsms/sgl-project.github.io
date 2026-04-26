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
    [2026-04-26 13:43:03] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.98it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.96it/s]


    2026-04-26 13:43:07,864 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 13:43:07] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=5632):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=3328):  10%|█         | 6/58 [00:04<00:31,  1.68it/s]

    Compiling num tokens (num_tokens=3072):  10%|█         | 6/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=2816):  10%|█         | 6/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:09,  4.82it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:04<00:09,  4.82it/s] Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:05<00:09,  4.82it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03,  9.42it/s]

    Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:05<00:03,  9.42it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:05<00:01, 15.20it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.02it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.02it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.02it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.02it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.02it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.02it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.02it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.02it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.02it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 22.02it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 29.62it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.86 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.83 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.83 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.82 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.82 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.80 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.80 GB):   9%|▊         | 5/58 [00:00<00:02, 20.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.80 GB):   9%|▊         | 5/58 [00:00<00:02, 20.85it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.85it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:02, 20.85it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.90it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.27it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.63it/s]Capturing num tokens (num_tokens=960 avail_mem=118.72 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.63it/s] Capturing num tokens (num_tokens=896 avail_mem=118.72 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.63it/s]

    Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.63it/s]Capturing num tokens (num_tokens=768 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.63it/s]Capturing num tokens (num_tokens=768 avail_mem=118.71 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.82it/s]Capturing num tokens (num_tokens=704 avail_mem=118.71 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.82it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.82it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.82it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.82it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  50%|█████     | 29/58 [00:01<00:00, 31.20it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 31.20it/s]

    Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 31.20it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 31.20it/s]Capturing num tokens (num_tokens=384 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 31.20it/s]Capturing num tokens (num_tokens=384 avail_mem=118.70 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.35it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.35it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.35it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.35it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.35it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.35it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.99it/s]Capturing num tokens (num_tokens=224 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.99it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.99it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.99it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.99it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.51it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.51it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.51it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.51it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.51it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.60it/s]Capturing num tokens (num_tokens=32 avail_mem=118.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.60it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.60it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.60it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.60it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  93%|█████████▎| 54/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=16 avail_mem=118.63 GB):  93%|█████████▎| 54/58 [00:01<00:00, 35.85it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 35.85it/s] Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:01<00:00, 31.11it/s]


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
    Generated text:  Steven James. I am the founder of the website. I am a programmer, and I work for a firm in the South Florida area. We provide product, service, and support for many web applications. I have a deep understanding of the technology behind websites, and I have been involved in all areas of web development. I have a background in web design, and I have a lot of experience with how to develop a responsive website. In addition, I have experience with HTML5, CSS3, and JavaScript. I am very experienced with the development of mobile sites, and I know how to develop a website that is responsive and mobile-first
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected official. He is the leader of the country, and he has a huge amount of power. He is called the "Chief Executive Officer," and he is also called the "President of the United States." The president is the first person in the United States to be elected by the people. The President has the power to veto bills. The President also has the power to declare a national emergency. The President has the power to appoint and reappoint judges. The President has the power to appoint and reappoint ambassadors, and to appoint and reappoint Supreme Court justices. The president has the power to appoint and reappoint federal judges.
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A．Paris B．Paris
    B．Paris
    
    The capital of France is Paris. 
    
    Paris is the capital of France, and it is a city located on the left bank of the River Seine in the north of the country. It is the largest city in France, with a population of over 2.5 million people. Paris is a historic and cultural center with many landmarks such as the Eiffel Tower, the Louvre Museum, the Champs-Élysées, the Arc de Triomphe, and the Panthéon. It is also a major financial and business center of France, and is
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain as we encounter new challenges such as the use of data to combat climate change, which is causing a global climate change, or how to create a new tool to better understand the effects of depression. However, as we move forward, we will have an enhanced ability to automate processes, reduce the cost of operations, and create new opportunities for businesses to grow. And with a growing number of businesses adopting AI, we can expect to see new applications of AI in the future.
    
    However, it is not yet clear how the technology will be used. Some experts believe that AI will be used to improve the efficiency and speed of processes, while others


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title] work. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title] work. I'm always looking for new challenges and opportunities to grow and learn.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter. Paris is a bustling metropolis with a rich cultural heritage and is a popular tourist destination. It is the largest city in France and the second-largest city in the world by population. The city is known for its diverse cuisine, art, and fashion, and is a major center for business, politics, and culture. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence. This could lead to more sophisticated forms of AI, such as those that can understand and adapt to human emotions and behaviors.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in even more areas, including diagnosis, treatment, and
    


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
    Generated text:  [Name] and I am a software engineer who works at [Company Name]. I have a passion for [Job Title] and enjoy helping others, whether it's through code, data analysis, or problem-solving. I am currently [Current Position] and I am always looking for opportunities to learn and grow in my field. I am a team player and enjoy working with others to achieve our goals. I believe that technology has the potential to change the world and I am excited to be a part of this exciting new journey. Thank you for asking to meet me, and I look forward to the opportunity to discuss how I can contribute to your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the Seine-Saint-Denis region of the Loire Valley. 
    
    A. Correct
    B. Incorrect
    
    B. Incorrect
    
    The capital city of France, Paris, is located in the Seine-Saint-Denis region of the Loire Valley. However, the region is not entirely surrounded by the Seine River, and there is also a smaller part of the city that is near the River Orne. The correct statement would be that Paris is located in the Seine-Saint-Denis region of the Loire Valley. 
    
    Among the given options, only option A ("Correct") is incorrect. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by an increase in the ability of AI systems to understand and reason about complex human-like behavior and emotions. This is known as "emotional intelligence," and it is likely to be a key feature in many AI systems in the future. 
    
    AI systems will also become more capable of learning from data and adapting to new situations, leading to more efficient and effective decision-making. This is likely to result in AI systems that can handle a wider range of tasks and applications than currently possible.
    
    In addition, AI systems will become more integrated with natural language processing and machine learning algorithms, enabling them to better understand and respond to human language


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

    Your

     Name

    ].

     I

     am

     a

     [

    insert

     your

     profession

    ,

     such

     as

     "

    program

    mer

    ",

     "

    writer

    ",

     "

    business

    woman

    ",

     etc

    .

    ].

     I

     work

     hard

     to

     bring

     my

     ideas

     to

     life

     and

     am

     always

     up

     for

     a

     challenge

    .

     I

     enjoy

     problem

    -solving

     and

     constantly

     strive

     to

     learn

     new

     skills

     and

     expand

     my

     knowledge

     base

    .

     I

     am

     a

     self

    -st

    arter

     who

     thr

    ives

     in

     fast

    -paced

     environments

     and

     is

     always

     willing

     to

     take

     risks

     for

     growth

     and

     development

    .

     What

    's

     your

     favorite

     hobby

     or

     activity

    ?

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     hobbies

     or

     activities

    ,

     but

     I

     can

     provide

     information

     about

     them

    !

     Can

     you

     tell

     me

     more

     about

     your

     favorite

     hobby

     or

     activity

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     one

     of

     the

     oldest

     cities

     in

     the

     world

    .

     Paris

     is

     renowned

     for

     its

     historical

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     Lou

    vre

     Museum

    ,

     and

     for

     its

     fashion

    ,

     music

    ,

     and

     gastr

    onomy

    .

     It

     is

     also

     a

     major

     economic

     center

     and

     home

     to

     many

     influential

     institutions

     and

     organizations

    ,

     including

     the

     French

     Academy

     of

     Sciences

     and

     the

     National

     Library

     of

     France

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     a

     major

     tourist

     destination

    .

     It

     is

     also

     known

     for

     its

     wine

     production

     and

     its

     rich

     culture

     and

     history

    .

     The

     city

     has

     a

     population

     of

     over

     

    2

     million

     people

     and

     is

     one

     of

     the

     largest

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     diverse

     and

     constantly

     evolving

    ,

     with

     potential

     areas

     of

     development

     including

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

    :

     Future

     AI

     will

     likely

     continue

     to

     advance

     in

     the

     field

     of

     machine

     learning

    ,

     leading

     to

     more

     sophisticated

     algorithms

     and

     models

     that

     can

     more

     accurately

     analyze

     and

     predict

     human

     behavior

    .
    


    2

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

     greater

     emphasis

     on

     ethical

     AI

    ,

     ensuring

     that

     AI

     systems

     are

     designed

     and

     operated

     in

     ways

     that

     promote

     fairness

    ,

     transparency

    ,

     and

     accountability

    .
    


    3

    .

     Integration

     with

     other

     fields

    :

     AI

     will

     likely

     continue

     to

     be

     integrated

     with

     other

     fields

     such

     as

     healthcare

    ,

     finance

    ,

     and

     energy

    ,

     leading

     to

    



```python
llm.shutdown()
```
