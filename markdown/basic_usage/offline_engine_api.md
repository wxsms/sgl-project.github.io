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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.25it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.24it/s]


    2026-04-12 08:12:58,760 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-12 08:12:58] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.79it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.51it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.69it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.69it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.69it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.69it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.69it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.69it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.69it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.69it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.25it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.25it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.25it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.25it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.25it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.25it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.25it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.25it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.25it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.25it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.25it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.25it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.25it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.25it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.61it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.61it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.61it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.61it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.61it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.61it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.61it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.98it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.98it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.98it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.98it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.98it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.98it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.98it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.98it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.08 GB):   2%|▏         | 1/58 [00:00<00:10,  5.42it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=119.05 GB):   2%|▏         | 1/58 [00:00<00:10,  5.42it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.05 GB):   3%|▎         | 2/58 [00:00<00:09,  5.71it/s]Capturing num tokens (num_tokens=7168 avail_mem=119.04 GB):   3%|▎         | 2/58 [00:00<00:09,  5.71it/s]Capturing num tokens (num_tokens=6656 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:09,  5.71it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=119.02 GB):   7%|▋         | 4/58 [00:00<00:05,  9.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=119.02 GB):   7%|▋         | 4/58 [00:00<00:05,  9.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.01 GB):   7%|▋         | 4/58 [00:00<00:05,  9.95it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.02 GB):   7%|▋         | 4/58 [00:00<00:05,  9.95it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.02 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.01 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.01 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.01 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.00 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.71it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=119.00 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.00 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=3072 avail_mem=119.00 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.00 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.96 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.96 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.96 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.94 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.01it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=118.92 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.92 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=960 avail_mem=118.94 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.77it/s] Capturing num tokens (num_tokens=896 avail_mem=118.93 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=832 avail_mem=118.93 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=768 avail_mem=118.93 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=704 avail_mem=118.92 GB):  36%|███▌      | 21/58 [00:01<00:01, 33.77it/s]Capturing num tokens (num_tokens=704 avail_mem=118.92 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=640 avail_mem=118.92 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=576 avail_mem=118.92 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=512 avail_mem=118.91 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.18it/s]

    Capturing num tokens (num_tokens=480 avail_mem=118.92 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=448 avail_mem=118.92 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=448 avail_mem=118.92 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.38it/s]Capturing num tokens (num_tokens=416 avail_mem=118.92 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.38it/s]Capturing num tokens (num_tokens=384 avail_mem=118.92 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.38it/s]Capturing num tokens (num_tokens=352 avail_mem=118.91 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.38it/s]Capturing num tokens (num_tokens=320 avail_mem=118.91 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.38it/s]Capturing num tokens (num_tokens=288 avail_mem=118.90 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.38it/s]Capturing num tokens (num_tokens=288 avail_mem=118.90 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=256 avail_mem=118.90 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=240 avail_mem=118.90 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.83it/s]

    Capturing num tokens (num_tokens=224 avail_mem=118.90 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=208 avail_mem=118.89 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=192 avail_mem=118.89 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=192 avail_mem=118.89 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=176 avail_mem=118.89 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=160 avail_mem=118.89 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=144 avail_mem=118.88 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=128 avail_mem=118.88 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=112 avail_mem=118.88 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=112 avail_mem=118.88 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s]Capturing num tokens (num_tokens=96 avail_mem=118.87 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=118.87 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s]Capturing num tokens (num_tokens=64 avail_mem=118.87 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s]Capturing num tokens (num_tokens=48 avail_mem=118.86 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s]Capturing num tokens (num_tokens=32 avail_mem=118.86 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s]Capturing num tokens (num_tokens=32 avail_mem=118.86 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=28 avail_mem=118.85 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=24 avail_mem=118.85 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=20 avail_mem=118.85 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=16 avail_mem=118.85 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=12 avail_mem=118.84 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.84 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=8 avail_mem=118.84 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.01it/s] Capturing num tokens (num_tokens=4 avail_mem=118.84 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=4 avail_mem=118.84 GB): 100%|██████████| 58/58 [00:01<00:00, 33.05it/s]


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
    Generated text:  Hannah, I have been attending Harvard University for the last 3 years and I am currently in my junior year. I love sports, in particular soccer, and playing basketball is one of my favorite sports. I enjoy watching the latest season of "The Super Bowl" and the current season of "100 Greatest Moments in Sports".
    
    I can't live without my cats, my blue tabby, Bubba, who has been with me for 7 years. I love him so much that I don't think he will ever leave me. He makes me smile, he plays with me, he takes my coffee with him and he
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 3 inches tall. The vice president is 3 feet 6 inches tall. If the president has a beard, his beard length is 1 inch less than half of his height, and the vice president has a beard, his beard length is 1 inch less than a third of his height. If the president doesn't have a beard, his beard length is twice his height. If the vice president doesn't have a beard, his beard length is three times his height. How many inches tall is the vice president without his beard? Let's first calculate the height and beard length for the president. The president's
    ===============================
    Prompt: The capital of France is
    Generated text:  (　　)
    A: Paris
    B: London
    C: Moscow
    D: Berlin
    
    To determine the capital of France, we need to identify the correct option from the given choices. The capital of France is typically the largest city in the country. Let's analyze each option:
    
    A: Paris - Paris is the capital of France, located in the south-central region of France. It is the largest city in France by population, with a population of approximately 1, million.
    
    B: London - London is the capital of the United Kingdom, not France. It is the largest city in the United Kingdom by population, with a
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and companies are no exception. Many have become successful in the business world because of their ability to work with AI. Here’s how companies can leverage AI to improve productivity and profitability.
    The future of AI is here, and companies are no exception. Many have become successful in the business world because of their ability to work with AI. Here’s how companies can leverage AI to improve productivity and profitability.
    (0) AI is a powerful tool that can transform the way that businesses operate. It can automate routine tasks, improve decision-making, and even help find new markets. But, how can companies harness this technology to drive growth?
    In


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination, known for its rich history, art, and cuisine. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. The city is known for its diverse population, including French, Spanish, and other European immigrants. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of art,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we can expect to see more automation and artificial intelligence in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for innovation and creativity.
    
    2. Enhanced privacy and security: As AI technology becomes more advanced, there will be an increased need for privacy and security measures to protect personal data. This could lead to new regulations and standards
    


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
    Generated text:  [Name] and I am a [age], [occupation], [status]. I've always had a [passion or interest] that I've been working on for years and I'm excited to share it with you. What's your name, and what's your occupation?
    Your description of yourself is a great start to building a neutral self-introduction. However, you could elaborate a bit more on what your passion or interest is, and what it is that you're working on. For example, you might say, "My passion is [insert something about what you're passionate about]." And you could also mention a little bit about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris, the capital of France, is known for its iconic Eiffel Tower, beautiful cathedrals, and diverse cultural scene. It's also home to iconic landmarks like the Louvre Museum and Notre-Dame Cathedral. Paris is a popular tourist destination and a popular destination for both locals and tourists alike. The French language is also spoken in Paris, making it a cultural and linguistic hub in Europe. It's a city full of history, culture, and food, making it a must-visit destination for anyone visiting France. Paris is often referred to as the "City of Light" and "The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and promising, with many potential developments and breakthroughs shaping the field. Here are some possible future trends in AI:
    
    1. Deep Learning: Deep learning is a subset of machine learning that uses neural networks with many layers to learn complex patterns and relationships in data. This approach is expected to become more advanced and powerful in the coming years, leading to new breakthroughs in areas like image and speech recognition, natural language processing, and autonomous driving.
    
    2. Explainable AI: As AI systems become more sophisticated, researchers are exploring ways to make them more transparent and explainable. This involves creating tools and frameworks that allow us to understand how AI


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

    'm

     a

     [

    job

     title

    ]

     at

     [

    company

    ].

     I

    'm

     [

    age

    ]

     years

     old

    ,

     [

    gender

    ]

     and

     I

    'm

     currently

     [

    career

     stage

    ].

     I

    've

     always

     loved

     [

    something

    ]

     since

     I

     was

     a

     child

    ,

     and

     I

    've

     always

     dreamed

     of

     [

    some

     aspiration

    ].

     I

    'm

     always

     looking

     for

     ways

     to

     [

    something

    ],

     so

     I

    've

     been

     trying

     to

     find

     a

     way

     to

     [

    something

    ].

     I

     love

     [

    an

     activity

     or

     hobby

    ],

     and

     I

     love

     [

    anything

     else

     that

     brings

     me

     joy

    ].

     I

    'm

     [

    born

     in

    ...

    ].

     I

    'm

     [

    born

     in

    ...

    ].

     I

    'd

     love

     to

     see

     [

    what

     it

     is

     about

     that

     I

     love

    ].

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     the

     largest

     city

     in

     France

     and

     is

     the

     seat

     of

     government

    ,

     administrative

    ,

     and

     cultural

     power

     in

     the

     country

    .

     It

     is

     located

     in

     the

     south

     of

     the

     country

     and

     is

     known

     for

     its

     stunning

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     cultural

     scene

    .

     Paris

     has

     been

     a

     popular

     tourist

     destination

     since

     ancient

     times

     and

     continues

     to

     attract

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

     home

     to

     many

     world

    -ren

    owned

     attractions

    ,

     including

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

     Notre

    -D

    ame

     Cathedral

    ,

     as

     well

     as

     a

     wide

     range

     of

     museums

    ,

     theaters

    ,

     and

     restaurants

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     and

     food

     scenes

    ,

     as

     well

     as

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     and

     we

     are

     seeing

     some

     exciting

     developments

     in

     this

     area

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

     Enhanced

     AI

    :

     As

     AI

     becomes

     more

     advanced

    ,

     we

     are

     seeing

     more

     complex

     and

     intelligent

     machines

     that

     can

     process

     and

     understand

     multiple

     inputs

     and

     contexts

    .

     This

     will

     enable

     machines

     to

     perform

     more

     complex

     tasks

     than

     we

     can

     imagine

    ,

     including

     natural

     language

     processing

    ,

     visual

     perception

    ,

     and

     decision

    -making

    .
    


    2

    .

     Autonomous

     vehicles

    :

     AI

     is

     being

     increasingly

     integrated

     into

     autonomous

     vehicles

    ,

     which

     are

     designed

     to

     drive

     on

     roads

     and

     highways

    .

     This will

     enable

     them

     to

     operate

     safely

    ,

     with

     high

     levels

     of

     autonomy

    ,

     and

     with

     improved

     efficiency

     and

     safety

    .
    


    3

    .

     Personal

    ization

    :

     With

    



```python
llm.shutdown()
```
