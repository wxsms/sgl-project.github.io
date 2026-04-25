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
    [2026-04-25 22:36:16] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.65it/s]


    2026-04-25 22:36:20,840 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-25 22:36:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.40it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.97it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.97it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.97it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.97it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.97it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:02<00:06,  6.97it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:02<00:06,  6.97it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:02<00:06,  6.97it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:02<00:06,  6.97it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:02<00:06,  6.97it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:02<00:02, 14.13it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:02<00:02, 14.13it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:02<00:02, 14.13it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:02<00:02, 14.13it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 14.13it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 14.13it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 14.13it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 14.13it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 14.13it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 21.36it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 21.36it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 21.36it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 21.36it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 21.36it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 21.36it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 21.36it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 21.36it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 21.36it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 21.36it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 30.19it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 30.19it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 30.19it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 30.19it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 30.19it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 30.19it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 30.19it/s]

    Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 30.19it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 30.19it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 30.19it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 39.00it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 39.00it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 39.00it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 39.00it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 39.00it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 39.00it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 39.00it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 39.00it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 39.00it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 39.00it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 47.71it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 47.71it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 47.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.05 GB):   3%|▎         | 2/58 [00:00<00:04, 12.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=117.03 GB):   3%|▎         | 2/58 [00:00<00:04, 12.11it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=117.03 GB):   3%|▎         | 2/58 [00:00<00:04, 12.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=117.03 GB):   7%|▋         | 4/58 [00:00<00:03, 14.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.45 GB):   7%|▋         | 4/58 [00:00<00:03, 14.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.44 GB):   7%|▋         | 4/58 [00:00<00:03, 14.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.43 GB):   7%|▋         | 4/58 [00:00<00:03, 14.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.43 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.44 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.44 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.11it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=116.44 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.43 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.43 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.43 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.42 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.42 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.15it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.42 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.15it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.41 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.15it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.41 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.41 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.74it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=116.41 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.40 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.40 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.38 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.38 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.16it/s]Capturing num tokens (num_tokens=960 avail_mem=116.40 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.16it/s] Capturing num tokens (num_tokens=896 avail_mem=116.39 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.16it/s]Capturing num tokens (num_tokens=832 avail_mem=116.38 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.16it/s]Capturing num tokens (num_tokens=768 avail_mem=116.38 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.16it/s]Capturing num tokens (num_tokens=704 avail_mem=116.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.16it/s]

    Capturing num tokens (num_tokens=704 avail_mem=116.35 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=640 avail_mem=116.35 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=576 avail_mem=116.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=512 avail_mem=116.33 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=480 avail_mem=116.35 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=448 avail_mem=116.35 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=448 avail_mem=116.35 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=416 avail_mem=116.35 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=384 avail_mem=116.34 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=352 avail_mem=116.34 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=320 avail_mem=116.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.87it/s]

    Capturing num tokens (num_tokens=288 avail_mem=116.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=288 avail_mem=116.33 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.86it/s]Capturing num tokens (num_tokens=256 avail_mem=116.33 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.86it/s]Capturing num tokens (num_tokens=240 avail_mem=116.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.86it/s]Capturing num tokens (num_tokens=224 avail_mem=116.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.86it/s]Capturing num tokens (num_tokens=208 avail_mem=116.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.86it/s]Capturing num tokens (num_tokens=192 avail_mem=116.29 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.86it/s]Capturing num tokens (num_tokens=192 avail_mem=116.29 GB):  71%|███████   | 41/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=176 avail_mem=116.28 GB):  71%|███████   | 41/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=160 avail_mem=116.28 GB):  71%|███████   | 41/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=144 avail_mem=116.28 GB):  71%|███████   | 41/58 [00:01<00:00, 39.69it/s]

    Capturing num tokens (num_tokens=128 avail_mem=116.27 GB):  71%|███████   | 41/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=112 avail_mem=116.27 GB):  71%|███████   | 41/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=112 avail_mem=116.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=96 avail_mem=116.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.01it/s] Capturing num tokens (num_tokens=80 avail_mem=116.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=64 avail_mem=116.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=48 avail_mem=116.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=32 avail_mem=116.25 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=32 avail_mem=116.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=28 avail_mem=116.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=24 avail_mem=116.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.85it/s]

    Capturing num tokens (num_tokens=20 avail_mem=116.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=16 avail_mem=116.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=16 avail_mem=116.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=12 avail_mem=116.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=8 avail_mem=116.23 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.75it/s] Capturing num tokens (num_tokens=4 avail_mem=116.23 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=4 avail_mem=116.23 GB): 100%|██████████| 58/58 [00:01<00:00, 34.51it/s]


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
    Generated text:  Alex and I am a 33 year old American who is fluent in many languages. I have always had a passion for writing and I enjoy writing about something that interests me. I am curious about how the language world is changing in the last few years, and I would like to contribute to this conversation by sharing some thoughts on the current state of the language world.
    Certainly! As an AI language model, I am not a native speaker of any language, but I can provide some insights based on current trends and developments in the language world. Language evolution is a fascinating topic, and there are several key trends that are shaping the language landscape
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. The president of the United States was born in 1946. When is the president of the United States likely to be born?
    To determine when the president of the United States is likely to be born, we need to follow these steps:
    
    1. Identify the current year.
    2. Subtract the birth year of the president from the current year.
    
    First, let's determine the current year. The current year is 2023, which is an odd number.
    
    Next, we need to find out what year the president was born. According to the problem, the president was born in 1946
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Does it follow that "Paris is the capital of France."?
    OPTIONS:
    [I] yes
    [II] it is not possible to tell
    [III] no
    
    [I]
    ===============================
    Prompt: The future of AI is
    Generated text:  in the AI
    
    The Future of AI is in the AI
    
    
      1. 2050 is the year of the smart machines, which will play a crucial role in the future, but as they do, they will disappear. “We do not know how the future will be,” said Claude, the artificial intelligence created by Google.
    
      2. The last time I was impressed by a machine was when I saw it making a connection. I was surprised by its intelligence. It was even more intelligent than I was.
    
      3. AI is the future of the technology industry, but it also poses a threat to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your job or experience here]. I enjoy [insert a short description of your hobbies or interests here]. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests here]. What's your favorite hobby? I love [insert a short description of your favorite hobby here]. What's your favorite book or movie? I love [insert a short description of your favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is home to iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as a vibrant arts and culture scene. Paris is a popular tourist destination and a major economic hub, with a strong economy and a thriving food and fashion industry. The city is also known for its fashion and art scene, with many famous artists and designers based in the area. Overall, Paris is a vibrant and dynamic city that is a must
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more robust and transparent AI systems that are designed to minimize harm and maximize safety
    


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
    Generated text:  [Your Name]. I am a [Your Profession] and I have been working in [Your Company] since [Your Company’s Founded Year] and I am currently a [Your Role in the Company].
    
    Please provide me with a brief summary of your work experience in your current role. What is your most rewarding project or experience that you have had the pleasure of collaborating on?
    
    Please also provide me with a brief summary of your long-term career goals in your field and how you plan to achieve them. What are the key skills that you believe are most important to a [Your Profession]? What industry trends and innovations are you most excited about and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the most populous city in France, with a population of over 2.3 million people. The city is known for its stunning architecture, vibrant cultural scene, and important historical landmarks. The French government occupies the Old Town quarter, while the nearby Eiffel Tower is a famous landmark. Other major attractions include the Louvre, Notre-Dame Cathedral, and the Seine River. Paris has played a significant role in the development and expansion of modern France, and continues to attract millions of visitors from around the world. Its status as the capital city is recognized by its status as a city of democracy and a UNESCO World
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very exciting and varied. Here are some possible trends that we can expect to see in the years ahead:
    
    1. Increased AI autonomy: With the ability to make decisions on the fly, AI will become more autonomous and able to make more personal decisions, such as choosing what content to browse or when to respond to a user's request.
    
    2. AI will be integrated into more areas of daily life: AI will be integrated into more areas of daily life such as healthcare, transportation, and education. This will lead to a more connected and interconnected world.
    
    3. AI will become more focused on creating more efficient and effective systems: AI will become


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

    ],

     and

     I

     am

     a

     [brief

    ly

     describe

     your

     profession

    ,

     role

    ,

     or

     current

     occupation

    ].

     I

     have

     always

     been

     fascinated

     by

     the

     idea

     of

     creating

     and

     improving

     things

    ,

     and

     I

     enjoy

     sharing

     my

     knowledge

     and

     expertise

     with

     others

    .

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     learn

     and

     grow

    .

     If

     you

     have

     any

     questions

     or

     need

     help

     with

     anything

    ,

     feel

     free

     to

     reach

     out

    .

     I

    'm

     always

     here

     to

     assist

     and

     inspire

     you

     in

     your

     journey

     to

     success

    !

     [

    Your

     Name

    ].

     [

    Your

     Name

    's

     profession

    ,

     role

    ,

     or

     current

     occupation

    ].

     [

    Your

     Name

    ]

     is

     a

     [

    brief

    ly

     describe

     your

     profession

    ,

     role

    ,

     or

     current

     occupation

    ].

     As

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     and

     most

     important

     city

     in

     the

     country

     and

     is

     the

     seat

     of

     government

    ,

     administration

    ,

     and

     culture

    .

     Paris

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

    ,

     and

     is

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    .

     It

     is

     the

     birth

    place

     of

     many

     famous

     artists

    ,

     writers

    ,

     and

     thinkers

    ,

     and

     is

     home

     to

     many

     of

     the

     world

    's

     major

     museums

    ,

     theaters

    ,

     and

     galleries

    .

     Paris

     is

     a

     major

     financial

     hub

     and

     plays

     a

     vital

     role

     in

     the

     country

    's

     economy

    ,

     with

     the

     city

     of

     Paris

     accounting

     for

     

    1

    8

    %

     of

     GDP

     and

     employing

     over

     

    1

    0

     million

     people

    .

     With

     its

     beautiful

     architecture

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

    ,

     but

     it

     is

     also

     highly

     uncertain

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     integration

     with

     natural

     language

     processing

    :

     As

     AI

     becomes

     more

     adept

     at

     understanding

     and

     interpreting

     natural

     language

    ,

     we

     can

     expect

     to

     see

     more

     AI

     systems

     that

     can

     better

     understand

     and

     respond

     to

     human

     language

     in

     new

     and

     creative

     ways

    .
    


    2

    .

     Adv

    ancements

     in

     machine

     learning

     and

     deep

     learning

    :

     With

     the

     continued

     development

     of

     machine

     learning

     and

     deep

     learning

     algorithms

    ,

     AI

     systems

     will

     become

     increasingly

     capable

     of

     learning

     and

     improving

     on

     their

     own

    ,

     leading

     to

     even

     greater

     levels

     of

     automation

     and

     efficiency

    .
    


    3

    .

     Improved

     cybersecurity

    :

     As

     AI

     systems

     become

     more

     complex

     and

     rely

     on

     vast

     amounts

     of

     data

    ,

    



```python
llm.shutdown()
```
