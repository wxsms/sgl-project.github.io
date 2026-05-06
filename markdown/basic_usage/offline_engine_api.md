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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.33it/s]


    2026-05-06 16:14:01,304 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 16:14:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.93it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.93it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.93it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.93it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.93it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.93it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.93it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.93it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.93it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.93it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.81it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.81it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.81it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.81it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.81it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.81it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.81it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.81it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.81it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.81it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.66it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.66it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.66it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.66it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.66it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.66it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.66it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.66it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.66it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.66it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.66it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 38.72it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 38.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 14.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 14.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 14.36it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.03 GB):   7%|▋         | 4/58 [00:00<00:03, 16.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   7%|▋         | 4/58 [00:00<00:03, 16.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   7%|▋         | 4/58 [00:00<00:03, 16.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   7%|▋         | 4/58 [00:00<00:03, 16.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.41it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.89it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=960 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.97it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.55it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.55it/s]Capturing num tokens (num_tokens=768 avail_mem=70.96 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.55it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.55it/s]

    Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.55it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.41it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.41it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.41it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.41it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.41it/s]Capturing num tokens (num_tokens=416 avail_mem=70.95 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.41it/s]Capturing num tokens (num_tokens=416 avail_mem=70.95 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.57it/s]

    Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s]Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s]

    Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s] Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  81%|████████  | 47/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  81%|████████  | 47/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  81%|████████  | 47/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  81%|████████  | 47/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  81%|████████  | 47/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  81%|████████  | 47/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.59it/s]

    Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.59it/s] Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.25it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.25it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 35.39it/s]


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
    Generated text:  Zina and I am an artist. I have a passion for creating art, both visual and auditory. I believe in the power of using art to bring people together and create a space where individuals can express themselves creatively.
    My art pieces are often inspired by the human experience and how we relate to each other. I believe that art is a reflection of our own emotions, and I try to capture these emotions in my work through my use of color, form, and texture. I have exhibited my work in various galleries and have received several awards for my artwork.
    My approach to creating art is to allow the process of creating my work to guide
    ===============================
    Prompt: The president of the United States is
    Generated text:  24 years older than the president of Brazil. The president of Brazil is 25 years younger than the president of China. If the president of China is 30 years old, how old is the president of Brazil? To determine the age of the president of Brazil, we need to follow the relationships given in the problem step by step.
    
    1. Identify the age of the president of China.
       The president of China is given as 30 years old.
    
    2. Determine the age of the president of Brazil.
       The problem states that the president of Brazil is 25 years younger than the president of China.
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. London
    C. Washington
    D. Moscow
    Answer: A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Washington
    D. Moscow
    Answer: A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Washington
    D. Moscow
    Answer: A
    
    Which of the following statements about the capital of France is true?
    A. Paris is the capital of France.
    B. The capital of France is located in the middle of France.
    C. The capital of France is the capital of the United States.
    D.
    ===============================
    Prompt: The future of AI is
    Generated text:  not linear. It is the future of machine learning, which is an important concept in artificial intelligence. This is the future of the web. It is the future of robotics. It is the future of the human race.
    As the future of the web becomes more and more fascinating, it is also becoming more and more apparent that the future of the web is not a single, linear path, but a complex web of interconnected paths. This is the future of robotics, which is an important concept in artificial intelligence. This is the future of the human race, which is the future of the web.
    So, how does the future of the web


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. Let's chat! [Name] [Job Title] [Company Name] [Company Address] [City, State, ZIP Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also known for its rich history, including the influence of French colonialism and the impact of the French Revolution. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to attract visitors from around the world. The city is home to many notable French artists, writers, and musicians, and its architecture and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there is likely to be a greater emphasis on ethical considerations. This could lead to the development of AI that is more transparent, accountable, and responsible.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, and transportation. It is likely that this trend will continue, with more integration of AI with other technologies to create even more
    


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
    Generated text:  [Your Name]. I'm [Your Age], a [Your Profession/Title] with a passion for [Your hobby or passion]. I enjoy [Your hobby] because [Your reasons for liking it]. If you are interested in learning more about my work or experiences, I'd be happy to share my story. What is your profession or title? (If you have not been told yet, just say "I'm a..." and then type your full name.) Thanks for asking. [Your Name]. Welcome to my world! As a [Your profession/Title], I am [Your profession/Title] with a passion for [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historic city with a rich and diverse cultural scene, and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also a major financial center, with many of the world's major banks and financial institutions located in the city. The city is home to numerous museums, art galleries, and cultural events throughout the year. Paris is a city of contrasts, with its towering buildings, narrow alleys, and lively atmosphere making it a popular tourist destination. It is also home to a diverse population, with many people of African, Asian, and Mediterranean descent
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be highly complex and multifaceted, with many potential trends that could shape the direction of this emerging technology. Here are some of the most likely future trends in AI:
    
    1. Increased focus on ethical and social implications: As AI systems become more complex and capable, they will become more involved in our daily lives. There will be increasing pressure to consider the ethical and social implications of AI systems, including their potential to affect human rights, privacy, and social equality.
    
    2. Enhanced machine learning capabilities: As AI systems become more powerful and capable, they will be able to learn and adapt to new situations more quickly and accurately than ever


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

    insert

     name

    ].

     I

    'm

     a

     versatile

     AI

     language

     model

    ,

     capable

     of

     processing

     complex

     information

     and

     generating

     coherent

     responses

     in

     various

     formats

    ,

     including

     text

    ,

     images

    ,

     and

     audio

    .

     With

     my

     extensive

     knowledge

     and

     vast

     experience

    ,

     I

    've

     been

     trained

     on

     a

     wide

     range

     of

     topics

    ,

     including

     but

     not

     limited

     to

     science

    ,

     technology

    ,

     history

    ,

     philosophy

    ,

     and

     many

     others

    .

     I

    'm

     always

     ready

     to

     assist

     you

     with

     any

     questions

     you

     may

     have

     or

     any

     tasks

     you

     may

     need

     assistance

     with

    .

     How

     can

     I

     help

     you

     today

    ?

     I

    'm

     here

     to

     assist

     you

     with

     any

     AI

    -related

     tasks

     you

     may

     need

    ,

     whether

     it

    's

     language

     translation

    ,

     data

     analysis

    ,

     or

     AI

     development

    .

     Feel

     free

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     of

     France

     and

     the

     largest

     city

     in

     both

     the

     European

     Union

     and

     the

     United

     Nations

    .

     It

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

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

     famous

     for

     its

     rich

     history

    ,

     art

    ,

     and

     culture

    ,

     including

     its

     vibrant

     arts

     scene

    .

     The

     city

     is

     home

     to

     many

     famous

     museums

    ,

     such

     as

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     as

     well

     as

     landmarks

     such

     as

     the

     Arc

     de

     Tri

    omp

    he

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     is

     a

     cultural

     and

     political

     center

     in

     central

     France

    ,

     and

     is

     home

     to

     many

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

     and

     is

     set

     to

     evolve

     rapidly

    .

     Here

     are

     some

     possible

     trends

     in

     the

     field

    :
    


    1

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

    ,

     and

     AI

     is

     playing

     a

     critical

     role

     in

     their

     development

    .

     The

     use

     of

     AI

     in

     autonomous

     vehicles

     will

     continue

     to

     increase

    ,

     and

     it

     is

     expected

     that

     AI

     will

     play

     a

     significant

     role

     in

     their

     design

     and

     implementation

    .
    


    2

    .

     Natural

     language

     processing

    :

     The

     field

     of

     AI

     is

     constantly

     evolving

    ,

     and

     natural

     language

     processing

     (

    N

    LP

    )

     is

     one

     of

     the

     most

     important

     areas

    .

     The

     use

     of

     AI

     in

     N

    LP

     will

     continue

     to

     increase

    ,

     and

     it

     is

     expected

     that

     AI

     will

     play

     a

     critical

     role

     in

     its

     development

    .
    


    3

    .

    



```python
llm.shutdown()
```
