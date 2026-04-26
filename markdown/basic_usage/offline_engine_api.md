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
    [2026-04-26 17:55:00] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.21it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.19it/s]


    2026-04-26 17:55:04,785 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 17:55:04] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.70it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.70it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.70it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.76it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.76it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.76it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.76it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.76it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.76it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.76it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.76it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.76it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.76it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.55it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.55it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.55it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.55it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.55it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.55it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.55it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.55it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.55it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.55it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 36.58it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 36.58it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 36.58it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.65 GB):   3%|▎         | 2/58 [00:00<00:03, 16.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.64 GB):   3%|▎         | 2/58 [00:00<00:03, 16.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.64 GB):   3%|▎         | 2/58 [00:00<00:03, 16.27it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=121.64 GB):   3%|▎         | 2/58 [00:00<00:03, 16.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.64 GB):   9%|▊         | 5/58 [00:00<00:02, 19.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.63 GB):   9%|▊         | 5/58 [00:00<00:02, 19.95it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.62 GB):   9%|▊         | 5/58 [00:00<00:02, 19.95it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.62 GB):   9%|▊         | 5/58 [00:00<00:02, 19.95it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=121.61 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=121.61 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]Capturing num tokens (num_tokens=3328 avail_mem=121.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=121.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=121.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.58it/s]Capturing num tokens (num_tokens=2560 avail_mem=121.60 GB):  21%|██        | 12/58 [00:00<00:01, 28.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=121.60 GB):  21%|██        | 12/58 [00:00<00:01, 28.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=121.59 GB):  21%|██        | 12/58 [00:00<00:01, 28.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=121.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=121.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=121.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.71it/s]Capturing num tokens (num_tokens=1280 avail_mem=121.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.71it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=121.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=121.57 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.89it/s]Capturing num tokens (num_tokens=960 avail_mem=121.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.89it/s] Capturing num tokens (num_tokens=896 avail_mem=121.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.89it/s]Capturing num tokens (num_tokens=832 avail_mem=121.57 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.89it/s]Capturing num tokens (num_tokens=768 avail_mem=121.57 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.89it/s]Capturing num tokens (num_tokens=704 avail_mem=121.57 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.89it/s]Capturing num tokens (num_tokens=704 avail_mem=121.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=640 avail_mem=121.56 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=576 avail_mem=121.56 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=512 avail_mem=121.55 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.90it/s]

    Capturing num tokens (num_tokens=480 avail_mem=121.56 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=448 avail_mem=121.56 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=448 avail_mem=121.56 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.05it/s]Capturing num tokens (num_tokens=416 avail_mem=121.56 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.05it/s]Capturing num tokens (num_tokens=384 avail_mem=121.56 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.05it/s]Capturing num tokens (num_tokens=352 avail_mem=121.55 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.05it/s]Capturing num tokens (num_tokens=320 avail_mem=121.54 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.05it/s]Capturing num tokens (num_tokens=288 avail_mem=121.54 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.05it/s]Capturing num tokens (num_tokens=288 avail_mem=121.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=256 avail_mem=121.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=240 avail_mem=121.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.63it/s]

    Capturing num tokens (num_tokens=224 avail_mem=121.53 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=208 avail_mem=121.53 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=208 avail_mem=121.53 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=192 avail_mem=121.53 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=176 avail_mem=121.53 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=160 avail_mem=121.52 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=144 avail_mem=121.52 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=128 avail_mem=121.52 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=128 avail_mem=121.52 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=112 avail_mem=121.52 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.63it/s]

    Capturing num tokens (num_tokens=96 avail_mem=121.51 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.63it/s] Capturing num tokens (num_tokens=80 avail_mem=121.51 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=64 avail_mem=121.50 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=64 avail_mem=121.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.29it/s]Capturing num tokens (num_tokens=48 avail_mem=121.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.29it/s]Capturing num tokens (num_tokens=32 avail_mem=121.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.29it/s]Capturing num tokens (num_tokens=28 avail_mem=121.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.29it/s]Capturing num tokens (num_tokens=24 avail_mem=121.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.29it/s]Capturing num tokens (num_tokens=24 avail_mem=121.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=20 avail_mem=121.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.39it/s]

    Capturing num tokens (num_tokens=16 avail_mem=121.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=12 avail_mem=121.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=8 avail_mem=121.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.39it/s] Capturing num tokens (num_tokens=8 avail_mem=121.48 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=4 avail_mem=121.47 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=4 avail_mem=121.47 GB): 100%|██████████| 58/58 [00:01<00:00, 35.67it/s]


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
    Generated text:  Fatma. I am 33 years old. I'm a teacher. My hobbies are reading and playing the violin. What can I say about myself? How would you describe me? Fatma
    You described yourself well! Your personality is active, passionate, and enthusiastic. You seem to love exploring and learning new things, which is admirable. Your passion for reading and your love for playing the violin are intriguing and demonstrate a strong interest in your life and work. 
    
    You also show good time management, which is very important. This can be seen in how you have a busy schedule with classes and work, but you manage it well
    ===============================
    Prompt: The president of the United States is
    Generated text:  considered to be the most powerful man in the world, and he is also the president of the United Nations. Based on the above statements, the president of the United States and the president of the United Nations are ____. 
    A. The same person
    B. Not necessarily the same person
    C. Different people
    D. Different organizations
    
    To determine the relationship between the president of the United States and the president of the United Nations based on the given statements, let's analyze each statement step by step.
    
    1. The president of the United States is considered to be the most powerful man in the world.
       - This statement suggests that
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Rome
    D. Athens
    
    The capital of France is Paris. Therefore, the correct answer is:
    
    A. Paris
    
    Rome, Athens, and London are all capitals of different countries, not France. London is the capital of the United Kingdom, Athens is the capital of Greece, and Rome is the capital of Italy. Paris is the capital of France.
    ===============================
    Prompt: The future of AI is
    Generated text:  set to shift from just collecting and organizing data to actually using that data to solve real-world problems. In fact, in the first quarter of 2016, Google announced that it was developing a technology that could predict a patient’s disease with 90% accuracy. This is a significant advancement in the field of AI, as it shows that AI can be used to solve complex problems that are beyond the scope of human expertise.
    The development of AI has not only improved the efficiency of healthcare, but it has also led to the creation of new medical technologies. For example, the use of AI in medical imaging has made it possible


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. What can you tell me about yourself? I'm a [insert a short, interesting fact about yourself]. I enjoy [insert a short, interesting fact about yourself]. I'm always looking for new experiences and learning opportunities. What are some of your favorite hobbies or interests? I'm always looking for new challenges and opportunities to grow as a person. What's your favorite book or movie? I love [insert a short, interesting fact about your favorite book or movie]. I'm always looking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its fashion industry, art, and cuisine. Paris is a major tourist destination and a cultural hub, with many museums, theaters, and restaurants. It is also home to the French Parliament and the French National Library. The city is known for its vibrant nightlife and is a popular destination for tourists and locals
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation and robotics: As AI technology continues to improve, we are likely to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and cost-effective solutions, but it could also lead to job displacement for some workers.
    
    2. Enhanced privacy and security: As AI technology becomes more advanced, there will be an increased need for privacy and security measures to protect sensitive information. This could lead to
    


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
    Generated text:  [Name] and I'm a [Job/Position] with over [Number] years of experience. I've always been passionate about helping others and my goal is to create meaningful connections with people. I'm always up for trying new things and learning new skills. I have a great sense of humor and I enjoy sharing my knowledge and experiences with others. I'm always looking for new challenges and opportunities to grow and learn. Thank you. [Name]. Welcome! I'm [Name] from [Company]. I'm excited to meet you and learn from you. [Name]. Hello, my name is [Name] and I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France, located on the north bank of the Seine River. It is the largest city in Europe and the third-largest city in the world. The city is known for its rich history, architectural marvels, and vibrant culture, including the annual Eiffel Tower parade, the Louvre Museum, and the Marais district. Paris is a cosmopolitan metropolis with a diverse population, including many international residents and tourists. It is also a global hub for finance, business, and media, and is home to many of the world's famous landmarks and museums.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and varied, with numerous potential areas of advancement and development. Here are some of the most promising trends to watch out for:
    
    1. Increased AI integration with other technologies: In the coming years, we are likely to see more AI integrated with other technologies, such as IoT, blockchain, and edge computing. This will create a more interconnected and seamless system, where AI can work in harmony with other systems to solve complex problems.
    
    2. Greater focus on ethical and responsible AI: As we become more aware of the potential risks of AI, there will be a greater focus on ethical and responsible AI. This will mean that AI systems will need


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

     Name

    ]

     and

     I

     am

     a

     [

    character

    's

     title

    ]

     at

     [

    company

     name

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     help

     you

     understand

     the

     role

     and

     mission

     of

     your

     company

    .

     How

     can

     I

     assist

     you

     today

    ?

     [

    Character

     Name

    ]

     is

     a

     [

    character

    's

     title

    ]

     at

     [

    company

     name

    ],

     dedicated

     to

     [

    mission

     statement

    ].

     I

    'm

     looking

     forward

     to

     our

     conversation

     and

     looking

     forward

     to

     learning

     more

     about

     your

     company

     and

     its

     goals

    .

     Can

     you

     provide

     me

     with

     any

     additional

     information

    ?

     [

    Character

     Name

    ]

     is

     a

     [

    character

    's

     title

    ]

     at

     [

    company

     name

    ],

     dedicated

     to

     [

    mission

     statement

    ].

     I

    'm

     looking

     forward

     to

     our

     conversation

     and

     looking

     forward

     to

     learning

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    City

    :

     France

    


    Country

    :

     France

    


    Population

    :

     

    6

    .

    7

     million

    


    Leader

    :

     President

    


    Founded

    :

     

    1

    3

    4

    8

    


    Area

    :

     

    1

    0

    0

    1

     km

    ²

    


    Country

     code

    :

     France

     -

     

    3

    3

    


    Capital

    :

     Paris

    


    Currency

    :

     Euro

     (

    EUR

    )
    


    Keywords

    :

     France

    ,

     capital

     city

    ,

     population

    ,

     leader

    ,

     founded

    ,

     area

    ,

     country

     code

    ,

     population

    ,

     leader

    ,

     founded

    ,

     area

    ,

     country

     code

    ,

     capital

    ,

     currency

    ,

     European

     Union

    ,

     city

    ,

     leader

    ,

     population

    ,

     leader

    ,

     founded

    ,

     area

    ,

     country

     code

    ,

     capital

    ,

     euro

    


    City

    :

     France

    


    Country

    :

     France

    


    Population

    :

     

    6

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

    ,

     but

     some

     possible

     trends

     that

     are

     currently

     being

     explored

     include

    :
    


    1

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

     AI

     systems

     are

     becoming

     more

     powerful

     and

     capable

     of

     learning

     from

     data

    .

     Future

     AI

     could

     see

     even

     more

     complex

     learning

     algorithms

     that

     can

     recognize

     patterns

     and

     make

     decisions

     without

     being

     explicitly

     programmed

    .
    


    2

    .

     Em

    phasis

     on

     ethical

     AI

    :

     With

     the

     increasing

     amount

     of

     data

     and

     personal

     information

     being

     processed

    ,

     there

     is

     a

     growing

     emphasis

     on

     ethical

     AI

    .

     This

     includes

     issues

     such

     as

     privacy

    ,

     bias

    ,

     and

     transparency

    .
    


    3

    .

     Increased

     reliance

     on

     AI

     for

     decision

    -making

    :

     As

     AI

     becomes

     more

     capable

     of

     making

     decisions

    ,

     there

     will

     be

     an

     increasing

     focus

     on

     its

     use

     in

    



```python
llm.shutdown()
```
