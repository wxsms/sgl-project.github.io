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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.97it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.97it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.97it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.97it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.97it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.97it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 23.01it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 23.01it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 23.01it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 23.01it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 23.01it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 23.01it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 23.01it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 23.01it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.01it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.01it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.01it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.86it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.86it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.86it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.86it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.86it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.86it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.86it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.86it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.33 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.33 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.33 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.33 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.33 GB):   9%|▊         | 5/58 [00:00<00:02, 20.13it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.32 GB):   9%|▊         | 5/58 [00:00<00:02, 20.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.31 GB):   9%|▊         | 5/58 [00:00<00:02, 20.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.31 GB):   9%|▊         | 5/58 [00:00<00:02, 20.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.31 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.31 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.31 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.30it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=58.30 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.30 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.30 GB):  21%|██        | 12/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.29 GB):  21%|██        | 12/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.29 GB):  21%|██        | 12/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.29 GB):  21%|██        | 12/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.29 GB):  21%|██        | 12/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.28 GB):  21%|██        | 12/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.28 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.28 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.76it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.28 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.76it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=58.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.76it/s]Capturing num tokens (num_tokens=960 avail_mem=58.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.76it/s] Capturing num tokens (num_tokens=960 avail_mem=58.27 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.77it/s]Capturing num tokens (num_tokens=896 avail_mem=58.27 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.77it/s]Capturing num tokens (num_tokens=832 avail_mem=58.26 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.77it/s]Capturing num tokens (num_tokens=768 avail_mem=58.26 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.77it/s]Capturing num tokens (num_tokens=704 avail_mem=58.26 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.77it/s]Capturing num tokens (num_tokens=704 avail_mem=58.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.73it/s]Capturing num tokens (num_tokens=640 avail_mem=58.25 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.73it/s]Capturing num tokens (num_tokens=576 avail_mem=58.25 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.73it/s]

    Capturing num tokens (num_tokens=512 avail_mem=58.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.73it/s]Capturing num tokens (num_tokens=480 avail_mem=58.25 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.73it/s]Capturing num tokens (num_tokens=448 avail_mem=58.25 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.73it/s]Capturing num tokens (num_tokens=448 avail_mem=58.25 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.90it/s]Capturing num tokens (num_tokens=416 avail_mem=58.25 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.90it/s]Capturing num tokens (num_tokens=384 avail_mem=58.25 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.90it/s]Capturing num tokens (num_tokens=352 avail_mem=58.24 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.90it/s]Capturing num tokens (num_tokens=320 avail_mem=58.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.90it/s]

    Capturing num tokens (num_tokens=288 avail_mem=58.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.90it/s]Capturing num tokens (num_tokens=288 avail_mem=58.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.91it/s]Capturing num tokens (num_tokens=256 avail_mem=58.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.91it/s]Capturing num tokens (num_tokens=240 avail_mem=58.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.91it/s]Capturing num tokens (num_tokens=224 avail_mem=58.22 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.91it/s]Capturing num tokens (num_tokens=208 avail_mem=58.22 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.91it/s]Capturing num tokens (num_tokens=208 avail_mem=58.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=192 avail_mem=58.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=176 avail_mem=58.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=160 avail_mem=58.21 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.17it/s]

    Capturing num tokens (num_tokens=144 avail_mem=58.21 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=128 avail_mem=58.21 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=128 avail_mem=58.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.94it/s]Capturing num tokens (num_tokens=112 avail_mem=58.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.94it/s]Capturing num tokens (num_tokens=96 avail_mem=58.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.94it/s] Capturing num tokens (num_tokens=80 avail_mem=58.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.94it/s]Capturing num tokens (num_tokens=64 avail_mem=58.19 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.94it/s]Capturing num tokens (num_tokens=64 avail_mem=58.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.06it/s]Capturing num tokens (num_tokens=48 avail_mem=58.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.06it/s]Capturing num tokens (num_tokens=32 avail_mem=58.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.06it/s]

    Capturing num tokens (num_tokens=28 avail_mem=58.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.06it/s]Capturing num tokens (num_tokens=24 avail_mem=58.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.06it/s]Capturing num tokens (num_tokens=24 avail_mem=58.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.97it/s]Capturing num tokens (num_tokens=20 avail_mem=58.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.97it/s]Capturing num tokens (num_tokens=16 avail_mem=58.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.97it/s]Capturing num tokens (num_tokens=12 avail_mem=58.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.97it/s]Capturing num tokens (num_tokens=8 avail_mem=58.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.97it/s] Capturing num tokens (num_tokens=4 avail_mem=58.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.97it/s]

    Capturing num tokens (num_tokens=4 avail_mem=58.16 GB): 100%|██████████| 58/58 [00:01<00:00, 36.37it/s]Capturing num tokens (num_tokens=4 avail_mem=58.16 GB): 100%|██████████| 58/58 [00:01<00:00, 34.23it/s]


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
    Generated text:  Alex and I'm currently a 3rd year student at **Royal Holloway, University of London**. I have been in one to four year education for the past 2 years and I really enjoy it! 
    
    In my free time, I like to do some coding, either through online tutorials, or through solving puzzles and games. I also love to read books, and my favorite book is "The Great Gatsby" by F. Scott Fitzgerald. 
    
    I have a passion for travel and I have been to lots of places, and I love to see the world and learn about different cultures. I have a lot of experience in
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. He believes that if he has y military bases, he will have 2y + 1 military personnel. However, he also considers the impact of having bases in other countries, which increases the military personnel by 5% of the current number of bases. If the United States currently has 4 bases, how many military personnel will the United States need if it has military bases in other countries? To determine the total number of military personnel the United States will need, we need to follow these steps:
    
    1. Calculate the number of military personnel if the United States has no bases currently.
    
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    
    B) London
    
    C) Moscow
    
    D) Rome
    
    E) Berlin
    
    To determine the capital of France, we need to consider the capital cities of the countries in Europe. Let's list the capital cities of the countries in Europe:
    
    1. **Paris, France** - Capital of France
    2. **London, United Kingdom** - Capital of the United Kingdom
    3. **Moscow, Russia** - Capital of Russia
    4. **Berlin, Germany** - Capital of Germany
    5. **Rome, Italy** - Capital of Italy
    
    From this list, we can see that the capital
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but it is clear that the emergence of the ubiquitous, pervasive, and powerful AI that we are discussing in this interview is almost certainly on the horizon. And what does that mean for the future of work? It will require some adaptation, and I think many, many organizations will have to adapt to this shift in how they work and how they manage their people.
    AI is changing almost every aspect of the way we work, in some ways, and in others, it is not changing at all. We may be seeing the emergence of new products and services that are completely different from what we are used to. We may see a wide


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its fashion, art, and cuisine, and is a major center of politics, science, and education in France. It is also home to many famous landmarks and museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. Paris is a city of contrasts, with its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be increased emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as greater transparency and accountability in AI systems.
    
    3. Increased use of AI in healthcare: AI is
    


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
    Generated text:  [Your Name], and I am an [occupation/character] with a passion for [their interests or profession]. I am here to be your friendly, reliable assistant, capable of solving any issue you may have. I am always ready to help, and I always strive to provide solutions that are both efficient and effective. Whether you need to troubleshoot a problem, provide guidance, or simply engage in a conversation, I am here to assist you. So, why not reach out to me today and let's chat? I am [Your Name], and I look forward to our conversation! #Friendly #Assistant #Helpful #Friendly #
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that hosts the iconic Eiffel Tower and is known for its rich history, art, and cuisine. France’s capital is also home to the Louvre Museum, the most famous of which houses the Mona Lisa, while the Accademia di San Luca, a former school of art, is also located in Paris. Other important landmarks include Notre-Dame Cathedral, the Place de la Concorde, and the Seine River. Paris is a popular tourist destination known for its historical architecture, arts and culture, and food, making it a must-visit for Parisians and tourists alike.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve significant advancements and innovation, including the development of more advanced machine learning models, the integration of AI into a wide range of industries and applications, and the development of ethical considerations and guidelines for AI use. Additionally, the rise of artificial intelligence will likely lead to the creation of new job roles and industries, as well as the development of new technologies and platforms for AI use. Overall, the future of AI is likely to be characterized by continued growth, innovation, and change, as well as the potential for both positive and negative impacts on society and the environment. However, it is important to consider the ethical implications of AI use and


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

    ]

     and

     I

     am

     a

     [

    insert

     occupation

     or

     profession

    ].

     I

     am

     [

    insert

     the

     character

    's

     occupation

     or

     profession

    ],

     and

     I

     have

     been

     in

     this

     industry

     for

     [

    insert

     number

     of

     years

     or

     decades

    ]

     years

    .

     I

     am

     [

    insert

     age

     or

     age

     range

    ]

     years

     old

     and

     I

     live

     in

     [

    insert

     a

     location

    ],

     and

     I

     love

     [

    insert

     a

     hobby

     or

     passion

    ]

     as

     much

     as

     you

     do

    .
    


    What

    's

     the

     most

     interesting

     or

     unique

     experience

     that

     you

    've

     had

     as

     a

     professional

     in

     this

     industry

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

     personal

     experiences

     or

     emotions

    ,

     but

     I

     can

     tell

     you

     that

     many

     professionals

     enjoy

     the

     thrill

     of

     solving

     complex

     problems

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     world

    -ren

    owned

     city

     of

     the

     French

     Riv

    iera

     and

     a

     popular

     tourist

     destination

    ,

     boasting

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

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

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

     as

     the

     "

    City

     of

     Light

    "

     for

     its

     vibrant

     art

     and

     culture

    ,

     and

     is

     considered

     a

     major

     cultural

     and

     economic

     hub

     in

     Europe

    .

     It

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

    ,

     and

     is

     the

     birth

    place

     of

     many

     influential

     figures

    ,

     including

     Napoleon

     Bon

    ap

    arte

    ,

     and

     is

     known

     for

     its

     historical

     and

     cultural

     significance

    .

     As

     of

     

    2

    0

    2

    1

    ,

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     dynamic

    ,

     with

     several

     trends

     that

     are

     shaping

     the

     technology

    's

     direction

    .

     Here

     are

     some

     potential

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     AI

     ethics

    :

     As

     AI

     becomes

     more

     ubiquitous

    ,

     there

     will

     be

     increasing

     attention

     to

     how

     it

     is

     used

     and

     the

     ethical

     implications

     of

     its

     decisions

    .

     This

     could

     lead

     to

     stricter

     regulations

     and

     guidelines

     for

     the

     development

     and

     use

     of

     AI

    .
    


    2

    .

     Faster

     AI

     development

    :

     As

     AI

     algorithms

     become

     more

     complex

     and

     sophisticated

    ,

     there

     will

     be

     a

     push

     towards

     faster

     and

     more

     efficient

     development

    .

     This

     could

     lead

     to

     the

     development

     of

     new

     algorithms

     and

     techniques

     that

     are

     more

     powerful

     and

     flexible

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     is

     being

     increasingly

     used

     in

    



```python
llm.shutdown()
```
