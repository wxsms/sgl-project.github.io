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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.68it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.66it/s]


    2026-04-14 19:52:07,014 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 19:52:07] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.70s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.65it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.65it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.96it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]

    Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.98it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.53it/s]

    Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.53it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.53it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.53it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.73it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 44.03it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 44.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.98 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.96 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.96 GB):   3%|▎         | 2/58 [00:00<00:05, 10.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.95 GB):   3%|▎         | 2/58 [00:00<00:05, 10.07it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.95 GB):   3%|▎         | 2/58 [00:00<00:05, 10.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.95 GB):   3%|▎         | 2/58 [00:00<00:05, 10.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:03, 15.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:03, 15.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:03, 15.81it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.94 GB):   9%|▊         | 5/58 [00:00<00:03, 15.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.94 GB):   9%|▊         | 5/58 [00:00<00:03, 15.81it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=118.94 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.94 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.93 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.93 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.93 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.93 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.47it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.93 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.47it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.92 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.47it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.92 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.47it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.92 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.91 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.47it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=118.91 GB):  31%|███       | 18/58 [00:00<00:01, 32.76it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.91 GB):  31%|███       | 18/58 [00:00<00:01, 32.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.90 GB):  31%|███       | 18/58 [00:00<00:01, 32.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.89 GB):  31%|███       | 18/58 [00:00<00:01, 32.76it/s]Capturing num tokens (num_tokens=960 avail_mem=118.90 GB):  31%|███       | 18/58 [00:00<00:01, 32.76it/s] Capturing num tokens (num_tokens=896 avail_mem=118.90 GB):  31%|███       | 18/58 [00:00<00:01, 32.76it/s]Capturing num tokens (num_tokens=896 avail_mem=118.90 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.28it/s]Capturing num tokens (num_tokens=832 avail_mem=118.89 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.28it/s]Capturing num tokens (num_tokens=768 avail_mem=118.89 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.28it/s]Capturing num tokens (num_tokens=704 avail_mem=118.89 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.28it/s]Capturing num tokens (num_tokens=640 avail_mem=118.88 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.28it/s]

    Capturing num tokens (num_tokens=576 avail_mem=118.88 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.28it/s]Capturing num tokens (num_tokens=576 avail_mem=118.88 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.41it/s]Capturing num tokens (num_tokens=512 avail_mem=118.87 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.41it/s]Capturing num tokens (num_tokens=480 avail_mem=118.89 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.41it/s]Capturing num tokens (num_tokens=448 avail_mem=118.89 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.41it/s]Capturing num tokens (num_tokens=416 avail_mem=118.88 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.41it/s]Capturing num tokens (num_tokens=384 avail_mem=118.88 GB):  48%|████▊     | 28/58 [00:01<00:00, 38.41it/s]Capturing num tokens (num_tokens=384 avail_mem=118.88 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.21it/s]Capturing num tokens (num_tokens=352 avail_mem=118.88 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.21it/s]Capturing num tokens (num_tokens=320 avail_mem=118.87 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.21it/s]Capturing num tokens (num_tokens=288 avail_mem=118.87 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.21it/s]

    Capturing num tokens (num_tokens=256 avail_mem=118.81 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.21it/s]Capturing num tokens (num_tokens=240 avail_mem=118.80 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.21it/s]Capturing num tokens (num_tokens=240 avail_mem=118.80 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=224 avail_mem=118.79 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=208 avail_mem=118.79 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.85it/s]

    Capturing num tokens (num_tokens=192 avail_mem=118.79 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=176 avail_mem=118.78 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=176 avail_mem=118.78 GB):  72%|███████▏  | 42/58 [00:01<00:00, 28.72it/s]Capturing num tokens (num_tokens=160 avail_mem=118.78 GB):  72%|███████▏  | 42/58 [00:01<00:00, 28.72it/s]Capturing num tokens (num_tokens=144 avail_mem=118.77 GB):  72%|███████▏  | 42/58 [00:01<00:00, 28.72it/s]

    Capturing num tokens (num_tokens=128 avail_mem=118.77 GB):  72%|███████▏  | 42/58 [00:01<00:00, 28.72it/s]Capturing num tokens (num_tokens=112 avail_mem=118.77 GB):  72%|███████▏  | 42/58 [00:01<00:00, 28.72it/s]Capturing num tokens (num_tokens=112 avail_mem=118.77 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.30it/s]Capturing num tokens (num_tokens=96 avail_mem=118.76 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.30it/s] Capturing num tokens (num_tokens=80 avail_mem=118.76 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.30it/s]Capturing num tokens (num_tokens=64 avail_mem=118.75 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.30it/s]Capturing num tokens (num_tokens=48 avail_mem=118.75 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.30it/s]Capturing num tokens (num_tokens=48 avail_mem=118.75 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.24it/s]Capturing num tokens (num_tokens=32 avail_mem=118.75 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.24it/s]Capturing num tokens (num_tokens=28 avail_mem=118.74 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.24it/s]

    Capturing num tokens (num_tokens=24 avail_mem=118.74 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.24it/s]Capturing num tokens (num_tokens=20 avail_mem=118.74 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.24it/s]Capturing num tokens (num_tokens=16 avail_mem=118.74 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.24it/s]Capturing num tokens (num_tokens=16 avail_mem=118.74 GB):  95%|█████████▍| 55/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=12 avail_mem=118.73 GB):  95%|█████████▍| 55/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=8 avail_mem=118.73 GB):  95%|█████████▍| 55/58 [00:01<00:00, 32.28it/s] Capturing num tokens (num_tokens=4 avail_mem=118.73 GB):  95%|█████████▍| 55/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=4 avail_mem=118.73 GB): 100%|██████████| 58/58 [00:01<00:00, 30.67it/s]


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
    Generated text:  Samantha. I am 17 years old and I live in China. I go to school every day and I can speak English. I have a friend named Jack. He lives in America and he is only 10 years old. Jack and I don't see each other much. Jack likes to play video games. He also likes to eat hamburgers. He likes to play with some of his friends. But I don't like playing video games. I don't like eating hamburgers. I like to read books. I have some books. I don't read newspapers or magazines. I don't like to play with my friends
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of a party that has been in office for the longest time, which is__________.
    A. the Democratic Party
    B. the Republican Party
    C. the Socialist Party
    D. the Farmer's Party
    Answer:
    
    A
    
    In a uniform electric field, the potential at a certain point is ____
    A. Equal to the average potential
    B. Equal to the difference between the potentials at the two adjacent points
    C. Greater than the difference between the potentials at the two adjacent points
    D. Less than the difference between the potentials at the two adjacent points
    Answer:
    
    Less than the difference between the potentials at the
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. London
    C. Moscow
    D. New York
    Answer: A
    
    According to the passage, a drop of blood from the top of a tree would...
    A. contain only white blood cells
    B. contain only red blood cells
    C. contain both white and red blood cells
    D. contain only white blood cells
    Answer: C
    
    In an economic context, what term is used to describe a situation where the prices of factors of production (such as land, labor, and capital) increase due to economic growth, leading to higher production costs?
    
    A) Diseconomies of scale
    ===============================
    Prompt: The future of AI is
    Generated text:  not just the future of our society, it is the future of our economy. But, what is AI? What makes AI unique? How will it impact the future of business and economy? AI will bring about significant changes to industries and society. Here is the first in our series on AI.
    Gartner’s CEO Dustin Moskovitz named artificial intelligence (AI) as one of the 10 most disruptive technologies of the 21st century. One of the few tech companies to predict the transition of artificial intelligence in the next five years, Gartner says AI will have a profound impact on both the future of the economy and society


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working at [company name] for [number of years] years. I have always been passionate about [job title] and have always wanted to be a [job title] myself. I am always looking for new challenges and opportunities to grow and learn. I am a [job title] and I am excited to be here at [company name]. I am looking forward to [job title] and I am looking forward to [job title]. I am looking forward to [job title] and I am looking forward to [job title]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also a major financial center and a major tourist destination. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a city that has a rich cultural heritage and is a major center of art, literature, and science. It is also a major center of business and commerce. Paris is a city that is constantly evolving and is a city that is always changing. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and the potential for AI to be used for malicious purposes.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This will allow for more complex and sophisticated AI systems that can perform a wide range of tasks
    


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
    Generated text:  [Name] and I am a [Age] year old [Occupation] with [Skill/Experience]. I'm a professional [X] and I'm always [X] in my life. I have [Skill/Experience] and I'm always [X] in my life. I'm an [X] and I'm always [X] in my life. I'm [Name] and I'm [Name]. I'm [Name] and I'm [Name]. I'm [Name]. I'm [Name]. I'm [Name]. I'm [Name]. I'm [Name]. I'm [Name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the second most populous city in the world and the economic and cultural center of the country.
    
    Briefly summarize the current economic condition of France. The current economic condition of France is characterized by high unemployment rates, low growth, and a lack of sustainable economic policies. The French government has been implementing various programs and initiatives aimed at stimulating economic growth, but progress has been slow due to political opposition and economic uncertainties. The country is also facing challenges related to globalization, technological advancements, and the aging population.
    
    Given the current economic situation, what measures have the French government taken to address the challenges faced by the country? The French government
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by a variety of trends and developments, including:
    
    1. Advancements in machine learning and deep learning techniques: These are the key enabling technologies that will drive the development of more intelligent and accurate AI systems in the future.
    
    2. Increased focus on ethical AI: There is growing concern about the impact of AI on human values and society at large. As such, there is a growing emphasis on developing ethical AI that is designed to minimize harm and maximize benefits.
    
    3. Increased integration with other technologies: AI will continue to be integrated into a broader range of other technologies, such as healthcare, transportation, and manufacturing, to create


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

    occupation

    ].

     I

    've

     always

     had

     a

     natural

     affinity

     for

     solving

     complex

     problems

    ,

     whether

     they

     be

     mathematical

    ,

     scientific

    ,

     or

     creative

    .

     Whether

     I

     find

     myself

     in

     a

     restaurant

     reviewing

     the

     menu

    ,

     at

     a

     conference

     reviewing

     the

     key

    notes

    ,

     or

     in

     my

     office

     discussing

     my

     latest

     project

    ,

     I

    'm

     always

     looking

     for

     new

     ways

     to

     creatively

     tackle

     problems

     and

     come

     up

     with

     innovative

     solutions

    .

     I

    've

     hon

    ed

     my

     skills

     through

     my

     years

     of

     experience

    ,

     from

     working

     on

     the

     front

     lines

     of

     healthcare

     with

     my

     team

     to

     leading

     a

     team

     of

     developers

     to

     working

     in

     a

     tech

     office

    .

     I

     enjoy

     collaborating

     with

     others

     and

     being

     a

     team

     player

    ,

     and

     I

    'm

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

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

     culture

    .

     The

     city

     is

     home

     to

     numerous

     famous

     landmarks

    ,

     including

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

     Paris

     is

     also

     known

     for

     its

     unique

     cuisine

     and

     art

     scene

    ,

     as

     well

     as

     its

     annual

     festivals

     and

     cultural

     events

    .

     The

     city

     is

     a

     bustling

     hub

     of

     activity

    ,

     with

     a

     population

     of

     over

     

    2

    .

    5

     million

     people

     and

     a

     rich

     tape

    stry

     of

     cultures

    ,

     languages

    ,

     and

     traditions

    .

     Paris

     is

     an

     iconic

     city

     that

     has

     capt

    ivated

     the

     world

     for

     centuries

    .

     


    Answer

     according

     to

    :

     The

     capital

     of

     France

     is

     Paris

    ,

     known

     for

     its

     stunning

     architecture

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

    :
    


    1

    .

     Integration

     of

     AI

     and

     IoT

    :

     The

     development

     of

     IoT

     devices

     will

     lead

     to

     the

     integration

     of

     AI

     into

     everyday

     life

    ,

     from

     smart

     homes

     to

     smart

     factories

    .

     This

     integration

     will

     enable

     new

     AI

     applications

     that

     will

     transform

     various

     sectors

    ,

     including

     healthcare

    ,

     transportation

    ,

     and

     manufacturing

    .
    


    2

    .

     Increased

     Use

     of

     AI

     for

     Personal

    ization

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     be

     able

     to

     provide

     personalized

     recommendations

     and

     experiences

    .

     This

     will

     enable

     businesses

     to

     improve

     customer

     satisfaction

    ,

     increase

     sales

    ,

     and

     enhance

     the

     user

     experience

    .
    


    3

    .

     AI

    -driven

     autonomous

     vehicles

    :

     AI

     will

     play

     a

     crucial

     role

     in

     the

     development

     of

     autonomous

     vehicles

    ,

     which

     will

    



```python
llm.shutdown()
```
