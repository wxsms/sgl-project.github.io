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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.49it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.49it/s]


    2026-05-20 03:24:23,740 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 03:24:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03,  9.65it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:03,  9.65it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 15.29it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:01, 20.80it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:01, 20.80it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:01, 20.80it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:01, 20.80it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:04<00:01, 20.80it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:04<00:01, 20.80it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:04<00:01, 20.80it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:04<00:01, 20.80it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:04<00:01, 20.80it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:04<00:01, 20.80it/s]

    Compiling num tokens (num_tokens=128):  60%|██████    | 35/58 [00:04<00:01, 20.80it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 30.42it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]

    Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 34.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.15 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.15 GB):   3%|▎         | 2/58 [00:00<00:05,  9.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.41 GB):   3%|▎         | 2/58 [00:00<00:05,  9.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.41 GB):   3%|▎         | 2/58 [00:00<00:05,  9.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.41 GB):   3%|▎         | 2/58 [00:00<00:05,  9.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.41 GB):   9%|▊         | 5/58 [00:00<00:03, 15.91it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.40 GB):   9%|▊         | 5/58 [00:00<00:03, 15.91it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.39 GB):   9%|▊         | 5/58 [00:00<00:03, 15.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.39 GB):   9%|▊         | 5/58 [00:00<00:03, 15.91it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=60.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.37 GB):  21%|██        | 12/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=60.37 GB):  21%|██        | 12/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.37 GB):  21%|██        | 12/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.36 GB):  21%|██        | 12/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.36 GB):  21%|██        | 12/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.36 GB):  21%|██        | 12/58 [00:00<00:01, 26.42it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=60.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=960 avail_mem=60.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.52it/s] Capturing num tokens (num_tokens=960 avail_mem=60.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.74it/s]Capturing num tokens (num_tokens=896 avail_mem=60.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.74it/s]Capturing num tokens (num_tokens=832 avail_mem=60.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.74it/s]Capturing num tokens (num_tokens=768 avail_mem=60.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.74it/s]Capturing num tokens (num_tokens=704 avail_mem=60.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.74it/s]Capturing num tokens (num_tokens=640 avail_mem=60.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.74it/s]

    Capturing num tokens (num_tokens=640 avail_mem=60.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=576 avail_mem=60.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=512 avail_mem=60.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=480 avail_mem=60.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=448 avail_mem=60.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=416 avail_mem=60.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=416 avail_mem=60.33 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=384 avail_mem=60.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=352 avail_mem=60.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=320 avail_mem=60.31 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=288 avail_mem=60.31 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.93it/s]

    Capturing num tokens (num_tokens=256 avail_mem=60.31 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=256 avail_mem=60.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=240 avail_mem=60.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=224 avail_mem=60.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=208 avail_mem=60.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=192 avail_mem=60.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=176 avail_mem=60.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=176 avail_mem=60.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.04it/s]Capturing num tokens (num_tokens=160 avail_mem=60.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.04it/s]Capturing num tokens (num_tokens=144 avail_mem=60.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.04it/s]Capturing num tokens (num_tokens=128 avail_mem=60.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.04it/s]Capturing num tokens (num_tokens=112 avail_mem=60.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.04it/s]

    Capturing num tokens (num_tokens=96 avail_mem=60.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.04it/s] Capturing num tokens (num_tokens=96 avail_mem=60.28 GB):  81%|████████  | 47/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=80 avail_mem=60.27 GB):  81%|████████  | 47/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=64 avail_mem=60.27 GB):  81%|████████  | 47/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=48 avail_mem=60.27 GB):  81%|████████  | 47/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=32 avail_mem=60.26 GB):  81%|████████  | 47/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=28 avail_mem=60.26 GB):  81%|████████  | 47/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=28 avail_mem=60.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=24 avail_mem=60.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=20 avail_mem=60.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=16 avail_mem=60.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=12 avail_mem=60.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.53it/s]

    Capturing num tokens (num_tokens=8 avail_mem=60.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.53it/s] Capturing num tokens (num_tokens=8 avail_mem=60.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.21it/s]Capturing num tokens (num_tokens=4 avail_mem=60.24 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.21it/s]Capturing num tokens (num_tokens=4 avail_mem=60.24 GB): 100%|██████████| 58/58 [00:01<00:00, 37.06it/s]


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
    Generated text:  Tanya. I'm a science teacher at a middle school in America. I'm a woman. And I'm a big fan of movies. I like to watch movies when I have free time. I like to talk to my friends after I finish watching a movie. After school, I often play computer games. I can be very bored after school. But my friends and I like to watch movies. We often get together in the evening. We like to watch movies when we are tired. I usually go to bed at 9 o'clock in the evening. I don't like to be up late at night to watch movies. At
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military jets to buy. The preference ranking of his policies is as follows:
    
    1. To keep the country safe, a military aviation aircraft must be at least 6 times the size of the current aircraft.
    2. To guarantee that a nation can travel to the moon and back within 2 years, an aircraft must be able to travel at least twice as fast as the current aircraft.
    3. To increase international relations, the aircraft must have a minimum range of 10,000 miles, which allows for international travel within 50 miles of a target.
    4. To ensure a peaceful world,
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    B) Berlin
    C) London
    D) Rome
    E) Madrid
    
    To determine the capital of France, let's first identify the correct answer choices provided:
    
    A) Paris
    B) Berlin
    C) London
    D) Rome
    E) Madrid
    
    The capital of France is **Paris**. Therefore, the correct answer is:
    
    A) Paris
    
    The other choices are not capital cities of France. Paris is the capital of France, and the other options are not cities of France. The capital of France is indeed Paris. 
    
    So, the final answer is \boxed{A}.
    ===============================
    Prompt: The future of AI is
    Generated text:  looking very promising. It’s not just a matter of how AI is used, but whether it’s used at all. AI is a fascinating technology that offers many possibilities and can be used in many areas such as healthcare, finance, education, entertainment, etc. This article will look at how AI can be used in the future and how it can be used for both good and bad effects. We will also discuss how to navigate the ethical considerations of AI and explore some of the potential risks and benefits that it may bring. Let’s dive in.
    The Future of AI
    AI technology has been making significant strides in recent years. It has been


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


    Generated text:  [Name], and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm passionate about [What I Love to Do], and I'm always looking for new challenges and opportunities to grow and learn. I'm a [Personality Type] who is [What You Do Best], and I'm always ready to help others and make a positive impact. I'm [What You Do Best] and I'm excited to be here with you. [Your Name] [Your Job Title] [Your Contact Information] [Your LinkedIn Profile
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and the French Revolution. Paris is home to many famous museums, including the Louvre and the Musée d'Orsay, and is a popular tourist destination. It is also known for its cuisine, including its famous Parisian dishes such as croissants, escargot, and escargot. Paris is a vibrant and diverse city with a rich cultural heritage, and is a major hub for business and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration of AI into everyday life: As AI becomes more integrated into our daily lives, we are likely to see more and more of it being used in new and innovative ways. This could include things like voice assistants that can understand and respond to our commands, self-driving cars that can navigate and navigate themselves, and even virtual assistants that can help us with a wide range of tasks.
    
    2. Greater emphasis on
    


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
    Generated text:  [Name], and I am a [occupation] who is dedicated to [job or mission]. I am a [number] [skill level] [educational background] [personality traits] with [number] [number] [number] [number] [number] years of experience in [field]. I am a [number] [number] [number] [number] [number] in my field, with [number] [number] [number] [number] [number] of achievements and [number] [number] [number] [number] [number] of awards and accolades. In my free time
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That statement accurately captures the capital city of France, which is the third-largest city in Europe and is known for its rich history, cultural institutions, and unique architecture. Paris is home to iconic landmarks such as the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and the Champs-Élysées, as well as numerous museums, theaters, and restaurants. It is also known for its passionate French culture, cuisine, and fashion. Paris is a major economic and financial center, and its status as the world's most popular tourist destination is further cemented by its museums, parks, and shopping.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking very promising and exciting. Here are some possible trends in the AI field:
    
    1. Ubiquity and Extensibility: With the increasing availability of computing power and data, AI will become more ubiquitous and extensible. AI will be able to be implemented in different industries, from healthcare and finance to transportation and manufacturing.
    
    2. Explainability: AI models will become more explainable, allowing humans to understand how they arrived at their predictions. This will enable us to trust and trust AI more.
    
    3. Personalization: AI will be able to provide personalized experiences, whether it's in healthcare, finance, or marketing. This will help


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

     a

     [

    age

    ]

     year

     old

     [

    gender

    ]

     with

     [

    physical

     attributes

    ]

     body

     type

    .

     I

     have

     [

    physical

     traits

    ]

     and

     I

     am

     [

    person

    ality

    ].

     I

     am

     [

    occupation

    ]

     and

     I

     am

     [

    the

     first

     name

     of

     my

     pet

    ,

     animal

    ,

     or

     other

     item

    ].

     I

     have

     [

    relationship

     with

     my

     pet

    ,

     animal

    ,

     or

     other

     item

    ]

     and

     I

     am

     [

    occupation

    ].

     My

     favorite

     color

     is

     [

    color

    ],

     and

     I

     like

     to

     [

    activity

    ].

     I

     enjoy

     [

    activities

    ],

     and

     I

     am

     [

    occupation

    ]

     and

     [

    the

     first

     name

     of

     my

     pet

    ,

     animal

    ,

     or

     other

     item

    ].

     I

     have

     [

    number

    ]

     friends

    ,

     and

     I

     value

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Question

    :

     How

     did

     the

     French

     Revolution

     influence

     the

     French

     capital

     Paris

    ?

     The

     French

     Revolution

     had

     a

     significant

     impact

     on

     the

     French

     capital

     Paris

    .

     The

     revolution

     led

     to

     the

     fall

     of

     the

     monarchy

     and

     the

     establishment

     of

     the

     Republic

    ,

     which

     included

     the

     establishment

     of

     the

     Second

     Republic

     and

     the

     creation

     of

     the

     Third

     Republic

    .

     The

     revolution

     also

     led

     to

     the

     rise

     of

     radical

     political

     ideas

    ,

     including

     the

     belief

     in

     the

     divine

     right

     of

     kings

     and

     the

     idea

     of

     a

     constitutional

     monarchy

    .

     The

     French

     Revolution

     also

     contributed

     to

     the

     modern

    ization

     of

     the

     city

    ,

     as

     the

     capital

     transformed

     from

     an

     agr

    arian

     society

     to

     a

     more

     industrial

    ized

     one

    .

     Despite

     the

     revolution

    ,

     Paris

     continued

     to

     grow

     and

    
    
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

    ,

     including

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     is

     becoming

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     is

     a

     growing

     emphasis

     on

     ensuring

     that

     AI

     systems

     are

     designed

     and

     used

     in

     a

     way

     that

     is

     fair

    ,

     transparent

    ,

     and

     equitable

    .

     This

     includes

     addressing

     issues

     such

     as

     bias

    ,

     privacy

    ,

     and

     accountability

    .
    


    2

    .

     Development

     of

     more

     advanced

     natural

     language

     processing

    :

     As

     AI

     systems

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     they

     will

     likely

     become

     even

     more

     capable

     of

     understanding

     and

     generating

     natural

     language

    .

     This

     could

     lead

     to

     increased

     efficiency

     in

     fields

     such

     as

     healthcare

    ,

     customer

     service

    ,

     and

     research

    .
    


    3

    .

     Growth

     of

     AI

    -based

    



```python
llm.shutdown()
```
