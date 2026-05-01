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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.00it/s]


    2026-05-01 05:44:43,475 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-01 05:44:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 21.19it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 21.19it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 29.59it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 29.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.73it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.55it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.55it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.55it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.55it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.64it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.64it/s] Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.64it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.64it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.12it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.63it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.63it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.63it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.63it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.63it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.63it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.58it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.58it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.58it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.58it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.58it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.58it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.25it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.25it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.25it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.25it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.25it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.25it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.14it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.90it/s]Capturing num tokens (num_tokens=20 avail_mem=72.13 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.90it/s]Capturing num tokens (num_tokens=16 avail_mem=71.90 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.90it/s]Capturing num tokens (num_tokens=12 avail_mem=71.90 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.90it/s]Capturing num tokens (num_tokens=8 avail_mem=71.90 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.90it/s] Capturing num tokens (num_tokens=8 avail_mem=71.90 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.75it/s]Capturing num tokens (num_tokens=4 avail_mem=71.89 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.75it/s]

    Capturing num tokens (num_tokens=4 avail_mem=71.89 GB): 100%|██████████| 58/58 [00:01<00:00, 37.91it/s]


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
    Generated text:  Sam and I'm 16 years old. I live in San Jose, California, USA. I was born on September 2, 2002. I have a twin sister and a brother. I am a big fan of the American football team, the Seattle Seahawks. I love to play football with my family and friends. I have a dog named Cesar. He is a sweet little guy. I also like to play video games and watch movies on my computer. What did you do when you were younger? When I was younger, I loved to draw and color pictures. Then I tried to become a guitar player
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a new term at the beginning of 2025. If the president's term ends in 2025, what is the president's birth year? Assume the current year is 2023.
    To determine the president's birth year, we need to find a year when the president's term ends in 2025. The president's term starts in 2023 and ends in 2025. Therefore, the president's birth year must be a year that is 12 years before the president's term ends in 2025.
    
    Let's denote the president
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. London
    C. Sydney
    D. Rome
    
    The capital of France is Paris. Therefore, the answer is A. Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  still very much up in the air. But there are some key trends that are happening that could be the major forces driving the future of AI in 2021.
    1. Machine Learning is still at the stage of not being able to predict or control the environment around us.
    2. We are seeing more and more companies starting to work on AI powered tools that will automate their workforce.
    3. AI is going to drive the "automation" of human labor. In other words, not just replacing workers but taking on tasks that are more complex, yet easier, to automate.
    4. In healthcare, AI is also being used to


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


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is a cultural and political center, known for its art, music, fashion, and cuisine. The city is also home to many museums, theaters, and landmarks, including the Louvre and the Notre-Dame Cathedral. Paris is a popular tourist destination and a major economic hub, with a strong economy and a thriving economy. It is also a major center for science and research
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    2. Greater integration with human intelligence: AI will continue to become more integrated with human intelligence, allowing for more sophisticated and nuanced interactions. This will involve developing AI that can understand and respond to human emotions and behaviors.
    
    3. Increased use of AI in healthcare: AI will be used
    


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
    Generated text:  [Name] and I'm [Age] years old. I was born in [Birthdate] in [City] and grew up in [Home town]. I've always been a [Occupation] who enjoys [Favorite hobby or activity], [Job title]. I started my career as a [Initial job title] in [City], and I've been working in this role for [Number of years] years. Over the years, I've learned [Advantage of working in this field]. I've always had a passion for [Subject or area of interest]. I'm an [Type of Person] who is [Positive traits].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city located in the French region of Auvergne-Rhône-Alpes. It was founded by the Romans and became the largest city of the Roman Empire. Paris is one of the world's most popular tourist destinations. 
    
    Paris was named after the ancient city of Paris, the capital of the Gaulish kingdom, and the city was made the capital of France in 1806 during the French Revolution. 
    
    The city is home to the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and many other landmarks. It is also the birthplace of many famous French artists, including Pablo
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by increasing sophistication, autonomy, and integration with human capabilities. Here are some potential trends in AI:
    
    1. Increased automation: As AI becomes more sophisticated, it will be able to perform a wider range of tasks that were once performed by humans. Automation will allow machines to perform repetitive and time-consuming tasks more efficiently, freeing up human workers to focus on more complex tasks.
    
    2. Enhanced personalization: AI will allow machines to learn from user behavior and preferences, providing more accurate and personalized recommendations. This will enable humans to interact with AI systems more easily and effectively, without the need for direct human interaction.
    
    3. Integration


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

    Age

    ]

     year

     old

     [

    Occup

    ation

    ].

     I

    'm

     an

     [

    What

    's

     Your

     Main

     Skill

    ?

     (

    e

    .g

    .

     Invent

    or

    ,

     Hacker

    ,

     etc

    .)

    ]

     and

     I

     enjoy

     [

    Why

     do

     you

     love

     your

     job

    ?

     (

    e

    .g

    .

     Learning

     new

     things

    ,

     helping

     others

    ,

     being

     creative

    ,

     etc

    .)

    ].
    


    I

    'm

     [

    Occup

    ation

    ]

     and

     my

     [

    What

    's

     Your

     Main

     Skill

    ?

     (

    e

    .g

    .

     Invent

    or

    ,

     Hacker

    ,

     etc

    .)

    ]

     is

     [

    Skill

    ].

     I

    've

     been

     working

     in

     [

    Field

    ]

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

    'm

     a

     [

    What

    's

     Your

     Main

     Skill

    ?

     (

    e

    .g

    .

     Invent

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     on

     the

     left

     bank

     of

     the

     Se

    ine

     River

     in

     the

     Î

    le

     de

     la

     C

    ité

    .

     The

     city

     is

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

     and

     is

     one

     of

     the

     most

     famous

     and

     iconic

     landmarks

     in

     the

     world

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     diverse

     culture

    ,

     and

     stunning

     architecture

    .

     Its

     museums

    ,

     museums

    ,

     and

     famous

     cafes

     have

     made

     it

     a

     cultural

     and

     shopping

     destination

     in

     its

     own

     right

    .

     With

     its

     world

    -ren

    owned

     art

     museums

    ,

     museums

     of

     science

    ,

     and

     museums

     of

     fashion

    ,

     Paris

     is

     a

     popular

     destination

     for

     tourists

    ,

     locals

    ,

     and

     international

     visitors

     alike

    .

     As

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     one

     of

     rapid

     development

     and

     innovation

    .

     Here

     are

     some

     possible

     future

     trends

    :
    


    1

    .

     **

    Quant

    um

     Computing

    **:

     The

     ability

     to

     perform

     certain

     tasks

     using

     quantum

     mechanics

     could

     revolution

    ize

     AI

     by

     allowing

     machines

     to

     solve

     complex

     problems

     that

     are

     currently

     impossible

     to

     solve

     with

     classical

     computers

    .

     Quantum

     computers

     could

     handle

     certain

     types

     of

     complex

     tasks

     that

     are

     beyond

     the

     capabilities

     of

     classical

     AI

    .
    


    2

    .

     **

    Ne

    ural

     Networks

    **:

     Neural

     networks

    ,

     a

     type

     of

     machine

     learning

    ,

     are

     being

     explored

     as

     a

     replacement

     for

     traditional

     machine

     learning

     algorithms

    .

     These

     networks

     could

     potentially

     be

     more

     powerful

     and

     able

     to

     handle

     more

     complex

     tasks

     than

     current

     machine

     learning

     algorithms

    .
    


    3

    .

     **

    Art

    ificial

     General

     Intelligence

     (

    AG

    I

    )**

    



```python
llm.shutdown()
```
