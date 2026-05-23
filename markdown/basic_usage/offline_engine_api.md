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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.96it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.28it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.53it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.53it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.53it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.53it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.53it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.53it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.53it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.53it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.24it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 37.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 37.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 37.27it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 37.27it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 37.27it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 37.27it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.70it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.77it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.77it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.77it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.77it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.77it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.77it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.77it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.30it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.30it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.30it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.30it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.30it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.30it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.91it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.91it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.91it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.91it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.91it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.91it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.87it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.87it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.87it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.87it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.87it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.87it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.05it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 42.38it/s]


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
    Generated text:  Dwayne Jackson, I am a 20-year-old male from Ohio. I play basketball for the University of Cincinnati, and I play this game for the school and on the court. I have a passion for basketball, and I am an avid fan of the game. I also enjoy sports such as soccer and tennis. I am passionate about my team and the things we do for the game. I enjoy being able to make a positive impact on the game and the lives of others, and I believe that with a positive attitude and a positive mindset, I can achieve anything.
    I have been practicing and training for the basketball game for over
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a third term. He has 5 successful years. Each of his terms lasted 2 years. If he can increase his term length to 3 years, he could have run for a third term. If he were to have run for another three years but decided to end his second term, how many years would he have had to run for a third term? To determine how many years the president of the United States would have had to run for a third term if he had run for another three years and ended his second term, we need to break down the problem step by step.
    
    1. **First Term**: The president
    ===============================
    Prompt: The capital of France is
    Generated text:  located at ( )
    A: Paris
    B: Berlin
    C: Moscow
    D: New York
    
    To determine the capital of France, we need to identify the capital of the country that has been the political and administrative center since the 11th century. Let's analyze the given options:
    
    A: Paris - While Paris is the capital of France, it has not been the political and administrative center since the 11th century. It was not the capital of France until the 14th century.
    
    B: Berlin - Berlin is the capital of Germany, not France. It has been the capital of France since the 
    ===============================
    Prompt: The future of AI is
    Generated text:  being redefined by the development of a new generation of cognitive computing models. In this blog post, we will delve into the latest advancements in AI, including the development of a new generation of cognitive computing models, as well as the implications of these advancements on the field of artificial intelligence.
    The development of new cognitive computing models has been a major focus of AI research in recent years. These models are designed to simulate the cognitive processes of human brains and are intended to be more efficient and accurate than traditional machine learning algorithms. There are several different types of cognitive computing models, including neural networks, expert systems, and evolutionary algorithms.
    One of the most


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, art, and culture, and is a major tourist destination. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a center for science, technology, and innovation. Paris is a vibrant and dynamic city that continues to evolve and grow. It is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is a symbol of France and a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced machine learning capabilities: AI systems will become even more capable of learning from large amounts of data and making more accurate predictions and decisions.
    
    3. Increased use of AI in healthcare: AI will be used to improve patient care, reduce costs, and increase efficiency in healthcare delivery.
    
    4. Greater use of AI in finance: AI will be used to improve risk management, fraud detection, and investment decision
    


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
    Generated text:  [Your Name], and I'm a/an [insert age, nationality, or profession]. I'm [insert your first name] by [insert middle name or alias] and I am [insert occupation]. My favorite hobbies are [mention two or more hobbies], and I enjoy [insert something related to your hobbies, such as reading, listening to music, or travel]. I like to [mention an activity or hobby that you enjoy], and I also enjoy [mention an activity or hobby that you enjoy], and I hope to live to [insert your age]. I am always looking for opportunities to learn something new, and I am always eager
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. The capital city of France is Paris. Paris is the capital of France and is the largest city in France. It is located on the northern bank of the Seine River, near the ancient city of Rome, and is the seat of the French government, the head of state, and the capital of the French department of Paris. It is also the largest city in terms of population. Paris is known for its rich culture, diverse arts scene, and iconic landmarks such as the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and the Palace of Versailles. Despite its size, Paris remains a modern
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a variety of new and evolving trends that will shape the development and deployment of the technology. Here are some potential future trends in AI that may be considered:
    
    1. Increased focus on ethical considerations: With the potential for AI to impact a wide range of industries, including healthcare, finance, transportation, and more, it is becoming increasingly important to address the ethical implications of AI. This will likely lead to increased focus on designing and deploying AI systems that are fair, transparent, and accountable.
    
    2. Expansion of AI-driven applications: With the increasing reliance on AI in various industries, it is likely that we will see more and more


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

    age

    ]

     year

    -old

     girl

     who

     enjoys

     [

    interest

    s

     or

     hobbies

    ]

     and

     [

    respons

    ibilities

     or

     responsibilities

    ].

     I

     love

     [

    job

     or

     volunteer

     work

    ],

     [

    occupation

     or

     hobby

    ],

     and

     [

    other

     interests

     or

     hobbies

    ].

     I

    'm

     currently

     [

    age

    ]

     years

     old

    ,

     and

     I

    've

     been

     [

    time

    ]

     in

     this

     field

    ,

     [

    career

     goals

    ],

     and

     [

    work

    /l

    ife

     balance

    /

    other

     challenges

    ].

     What

    's

     your

     name

    ?

     What

    's

     your

     age

    ?

     And

     do

     you

     have

     any

     other

     hobbies

     or

     interests

    ?

     Your

     answer

     should

     be

     a

     brief

    ,

     neutral

     statement

     that

     highlights

     your

     positive

     qualities

     and

     interests

    .


    [

    Name

    ],

     how

     are

     you

    ?

     How

     would

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     bustling

     met

    ropolis

     and

     one

     of

     the

     world

    's

     most

     cosm

    opolitan

     cities

    ,

     known

     for

     its

     unique

     architecture

    ,

     vibrant

     culture

    ,

     and

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

     and

     the

     E

    iff

    el

     Tower

    .

     The

     city

     is

     home

     to

     a

     diverse

     population

    ,

     with

     many

     ethnic

     groups

     and

     cultures

    ,

     and

     is

     a

     major

     hub

     of

     commerce

    ,

     entertainment

    ,

     and

     diplomacy

     for

     France

     and

     Europe

    .

     Paris

     is

     also

     known

     for

     its

     art

     scene

    ,

     food

     culture

    ,

     and

     wine

     production

    ,

     which

     have

     made

     it

     a

     global

     center

     for

     cultural

     and

     culinary

     exploration

    .

     Its

     skyline

     features

     tall

     buildings

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     and

     its

     historic

     center

     includes

     numerous

     museums

    ,

     galleries

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     significant

     growth

     in

     automation

    ,

     AI

     systems

     becoming

     more

     sophisticated

    ,

     and

     AI

     systems

     becoming

     increasingly

     integrated

     into

     the

     natural

     world

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     in

     the

     next

     few

     years

    :
    


    1

    .

     Increased

     automation

    :

     With

     the

     increasing

     automation

     of

     various

     tasks

    ,

     AI

     will

     become

     more

     and

     more

     integrated

     into

     almost

     every

     aspect

     of

     our

     lives

    .

     This

     will

     involve

     the

     creation

     of

     more

     sophisticated

     robots

     and

     machines

     that

     can

     perform

     tasks

     that

     were

     previously

     done

     by

     humans

    .

     This

     will

     lead

     to

     a

     reduction

     in

     the

     need

     for

     human

     labor

     and

     a

     more

     efficient

     use

     of

     resources

    .
    


    2

    .

     AI

     systems

     becoming

     more

     sophisticated

    :

     AI

     systems

     will

     become

     more

     sophisticated

     and

     capable

     of

     performing

    



```python
llm.shutdown()
```
