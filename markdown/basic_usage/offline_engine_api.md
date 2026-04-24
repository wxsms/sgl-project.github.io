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
    [2026-04-24 03:21:56] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.58it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.57it/s]


    2026-04-24 03:22:01,532 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 03:22:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.70it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.70it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.70it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.70it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.69it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.69it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.69it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.69it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.69it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.69it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.69it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.69it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.69it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 20.78it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 20.78it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 20.78it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 20.78it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 20.78it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 20.78it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 20.78it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 20.78it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 20.78it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 20.78it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 29.72it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 29.72it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 29.72it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 48.13it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 48.13it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 48.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.08it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  31%|███       | 18/58 [00:00<00:01, 34.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 34.79it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 34.79it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  31%|███       | 18/58 [00:00<00:01, 34.79it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  31%|███       | 18/58 [00:00<00:01, 34.79it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  31%|███       | 18/58 [00:00<00:01, 34.79it/s]Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.37it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.37it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.37it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.37it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.37it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.37it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.95it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.95it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.95it/s]

    Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.95it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.95it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.95it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.20it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.20it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.20it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.20it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.92it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.92it/s]

    Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.92it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.92it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.92it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.92it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.63it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.63it/s]

    Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.01it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 42.68it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 38.07it/s]


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
    Generated text:  Karen, I am 25 years old. I have been attending a group therapy class to help me deal with my anxiety. I have noticed a few things recently: my heart feels heavy, my legs feel heavy, I have a painful feeling in my jaw, and I also have a burning sensation in my throat, which is causing me to feel like I can't breathe. 
    
    I have also noticed that my chest feels tight, and I often have a racing heart. I have also experienced a decrease in my ability to focus and I am having difficulty sleeping. I often find it difficult to connect with others.
    
    When I am in a
    ===============================
    Prompt: The president of the United States is
    Generated text:  a public office. Candidates of every major political party run for this office, seeking to govern the United States. However, some political scientists argue that the role of the president is not just that of a political office-holder, but also that of a political leader who helps shape the political system. Candidates should be evaluated by a political scientist, and the role of the president should be reshaped to better serve the public interest and the democratic process. 
    
    Based on this passage, what is the role of the president in the American political system?
    
    OPTIONS:
    A). The president should be evaluated by a political scientist to ensure that the president is a good
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Brussels C. London D. Moscow
    Answer:
    A
    
    In the early days of the People's Republic of China, which of the following statements about the People's Heroes Monument is correct?
    A. It was built in 1950.
    B. It was commissioned by the Communist Party of China.
    C. It was located on the southern shore of the Yellow Sea.
    D. The granite material for the monument was mainly sourced from the sandstone of the Ordos Basin.
    Answer:
    D
    
    [Multiple Choice Question] After a short period of storage, which of the following would be the most
    ===============================
    Prompt: The future of AI is
    Generated text:  in the process of discovery, in the study of how these technologies work, and the issues that arise from their use and development. AI is increasingly important in many different sectors, including healthcare, finance, transportation, and security.
    The future of AI is also at risk due to the proliferation of cyber attacks and the lack of security measures. The use of AI can also lead to job displacement as computers and machines replace human workers, which could lead to social and economic problems.
    As AI continues to evolve and develop, it is important for people to stay informed about its potential impact on society and to take steps to ensure that it is used in a


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about your personality or skills]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do for fun? I enjoy [insert a short, positive, enthusiastic statement about your hobbies or interests]. I'm always looking for new experiences and learning opportunities to grow and develop as a person. What's your favorite hobby or activity? I love [insert a short, positive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and a diverse population of over 2 million people. It is a popular tourist destination and a major economic center in Europe. The city is known for its cuisine, fashion, and art, and is home to many famous museums, theaters, and landmarks. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of innovation and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation and efficiency: AI is expected to continue to automate a wide range of tasks, from manufacturing and transportation to customer service and healthcare. This will lead to increased efficiency and productivity, as machines can perform tasks that would previously require human intervention.
    
    2. Enhanced human-computer interaction: AI is likely to become more integrated with human-computer interaction, allowing for more natural and intuitive interactions between humans and machines. This could lead to more personalized and effective interactions, as well as a greater understanding of human emotions and motivations.
    
    3. Greater reliance on AI for decision-making: AI is likely
    


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
    Generated text:  [Name] and I am a [Occupation]. I'm [Job Title] at [Company/Location]. I've been [Years in Position] at [Company/Location], and I'm here to share my experiences with you. Thank you. (End of self-introduction) How can I assist you today? [Name] [Occupation]
    
    Hello, my name is [Name] and I am a [Occupation]. I'm [Job Title] at [Company/Location]. I've been [Years in Position] at [Company/Location], and I'm here to share my experiences with you. Thank you.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the second-largest city in Europe, with an estimated population of around 10 million people. The city is known for its medieval architecture, rich culture, and lively nightlife. It is also known as the "City of Love" for its iconic Eiffel Tower, and as a symbol of France's historical and political strength. Paris is home to numerous museums, theaters, and landmarks, including the Louvre, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also a hub of the French economy and a major transportation hub, with numerous airports and railroads
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and exciting, with many possibilities and potential applications. Here are some possible trends in AI that could shape the future:
    
    1. Increased AI integration with human decision-making: AI is becoming more integrated with human decision-making systems, allowing for more complex and informed decisions to be made. This could lead to more effective and ethical use of AI in various fields.
    
    2. AI-driven automation: AI is being used to automate various tasks, from production processes to customer service to healthcare. While this can lead to increased efficiency, it also raises concerns about job loss and the potential for automation to lead to inefficiency.
    
    3. AI-driven privacy concerns:


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

    ].

     I

    'm

     a

     [

    age

    ]

     year

     old

     [

    gender

    ]

     (

    gender

    )

     who

     has

     always

     been

     passionate

     about

     [

    topic

     of

     interest

    ]

     and

     strive

     to

     make

     a

     difference

    .

     I

     thrive

     in

     a

     fast

    -paced

     environment

     and

     am

     a

     team

     player

    .

     I

     love

     to

     learn

     new

     things

    ,

     and

     I

    'm

     always

     willing

     to

     adapt

     to

     new

     challenges

    .

     My

     goal

     is

     to

     inspire

     others

     to

     do

     the

     same

    ,

     and

     I

     believe

     that

     being

     a

     leader

     is

     essential

     to

     achieving

     my

     vision

     for

     the

     future

    .

     What

    's

     your

     name

    ?

     How

     old

     are

     you

    ?

     What

    's

     your

     gender

    ?

     What

    's

     your

     profession

    ?

     What

    's

     your

     area

     of

     interest

    ?

     What

    's

     your

     goal

    ?

     When

     did

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     modern

     and

     cosm

    opolitan

     city

     renowned

     for

     its

     historical

     architecture

    ,

     art

     museums

    ,

     and

     culinary

     delights

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

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

     autonomous

     vehicles

    ,

     natural

     language

     processing

    ,

     and

     predictive

     analytics

    .

     Additionally

    ,

     AI

     is

     expected

     to

     become

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     blockchain

    ,

     to

     create

     new

     possibilities

     and

     applications

    .
    


    One

     area

     of

     focus

     is

     the

     development

     of

     AI

     that

     can

     better

     understand

     and

     adapt

     to

     the

     nuances

     of

     human

     behavior

    .

     This

     could

     include

     developing

     more

     sophisticated

     algorithms

     that

     can

     recognize

     patterns

     in

     human

     behavior

     and

     respond

     appropriately

     to

     different

     situations

    .
    


    Another

     area

     of

     focus

     is

     the

     development

     of

     AI

     that

     can

     perform

     tasks

     that

     were

     previously

     considered

     too

     difficult

     or

     dangerous

     for

     humans

     to

     do

    .

     For

     example

    ,

     AI

     could

     be

     used

     to

     develop

     safer

    ,

     more

     efficient

    



```python
llm.shutdown()
```
