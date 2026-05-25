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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 15.02it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 15.02it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 15.02it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 15.02it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 15.02it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 23.06it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.06it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.53it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.53it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.53it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.53it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.53it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  50%|█████     | 29/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 42.76it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.69it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.69it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.69it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.69it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.69it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.97it/s]

    Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]

    Capturing num tokens (num_tokens=28 avail_mem=75.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=24 avail_mem=75.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=20 avail_mem=75.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=20 avail_mem=75.50 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=16 avail_mem=74.06 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=12 avail_mem=72.44 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=8 avail_mem=72.43 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.45it/s] Capturing num tokens (num_tokens=4 avail_mem=72.43 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.45it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 40.40it/s]


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
    Generated text:  Chris. I'm a computer programmer and a photographer. I work at the U. S. Geological Survey in the Middle East and serve as a photographer for a national company.
    
    ### What is my interest in writing and photography?
    
    My interest in writing and photography was sparked at an early age. I always enjoyed drawing, and I did very well in school. As an adult, I discovered photography. I really enjoyed doing it, and it has been one of my favorite hobbies. I love to capture moments in the world around me and share them with others. I hope to continue to do so, and my photography will play a significant part in
    ===============================
    Prompt: The president of the United States is
    Generated text:  30 years older than the president of Brazil. The president of Brazil is half the age of the president of the United States. If the president of the United States is currently 50 years old, calculate the average age of all three presidents.
    
    To find the average age of the president of the United States, the president of Brazil, and the president of the United States, we can follow these steps:
    
    1. Determine the age of the president of Brazil.
    2. Determine the age of the president of the United States.
    3. Calculate the average age of the three presidents.
    
    First, we know that the president of the United
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. Madrid
    C. Rome
    D. Berlin
    Answer:
    A
    
    Which of the following is an example of a drug?
    A. Iodine
    B. Mercaptan
    C. Aspirin
    D. Alcohol
    E. Benzoic acid
    Answer:
    C
    
    The correct way to write the word 'Rice' is ____
    A. Rice
    B. rice
    C. rices
    D. rice
    Answer:
    B
    
    Which of the following is a simple form of expression? A. A cake was eaten by her B. He is sleeping C.
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of business leaders, not technology developers. Read on to learn what business leaders can do to advance the field.
    The business leaders who will lead the future of AI are not the software developers or the tech companies. In fact, they are not even the software developers or the tech companies. They are the leaders of the business world.
    They are the ones who will make the AI systems work and where the AI systems will fail.
    The business leaders are the ones who will lead the AI transformation of their industries. They are the ones who will lead the shift to AI and will make the AI systems work and fail.
    The business leaders


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your career and interests. Let's chat! [Name] [Job Title] [Company Name] [Company Address] [City, State, Zip Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. The city is known for its vibrant nightlife, fashion, and food scene, as well as its role in hosting major international events such as the Olympics and the World Cup. Paris is a city that is both a cultural and political center of France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that could be expected in the AI field:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve the quality of care. As AI technology continues to advance, we can expect to see even more widespread use of AI in healthcare, with the potential to revolutionize the way we treat and diagnose diseases.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management,
    


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
    Generated text:  [Name], and I'm a [occupation] with [personal interest or passion]. I've always been [what do you think makes you uniquely special?]. 
    
    I believe in [what is your core belief or principle?]. I enjoy [what activity or hobby that brings me joy]? And I try to [a habit or behavior that reflects your character]. 
    
    Please share more about yourself, your journey, and what you're currently working on. Let's create a connection that resonates with them. 
    
    [Name], your journey, your personal story, and what you're working on, are important parts of your self-introduction.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The answer is: Paris is the capital city of France. 
    
    The historical and cultural center of the country, Paris is known for its iconic landmarks, rich history, and diverse neighborhood scenes, making it a major global city with an impressive array of attractions. It's also home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the famous Sainte-Chapelle. Paris is known as the "city of love" and is a significant cultural hub, home to numerous world-renowned festivals and events. The city is also renowned for its fashion and art scene, and has become a major center for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to evolve and expand in many interesting ways. Here are some potential trends that could shape the future of AI:
    
    1. Advancements in Machine Learning: With the advancement of machine learning algorithms and neural networks, AI systems will become more capable of recognizing patterns and making decisions with greater accuracy and efficiency.
    
    2. Increased Use of AI in Healthcare: AI will play an increasingly important role in healthcare, with the ability to analyze medical images, diagnose diseases, and develop personalized treatment plans.
    
    3. AI in Finance: AI will be used in finance to predict market trends, detect fraud, and manage investments more effectively.
    
    4. AI in


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

     friendly

     and

     outgoing

     individual

     who

    's

     always

     ready

     to

     lend

     a

     helping

     hand

    .

     Whether

     it

    's

     organizing

     a

     party

    ,

     cleaning

     the

     house

    ,

     or

     just

     chatting

     with

     someone

    ,

     I

    'm

     here

     to

     assist

     you

     in

     any

     way

     I

     can

    .

     My

     greatest

     strength

     is

     my

     ability

     to

     be

     patient

    ,

     listen

    ,

     and

     offer

     my

     time

    ,

     which

     I

     value

     highly

    .

     I

    'm

     open

     to

     learning

     from

     others

     and

     always

     eager

     to

     improve

     my

     skills

    .

     So

     please

     feel

     free

     to

     ask

     me

     anything

    ,

     and

     I

    'll

     be

     more

     than

     happy

     to

     help

    !

     Let

    's

     connect

     and

     start

     our

     friendly

     little

     chat

    !

     [

    Your

     Name

    ]

     [

    Your

     Contact

     Information

    ]

     [

    Your

     Contact

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     

    1

    8

    th

     and

     

    1

    9

    th

     century

     Gothic

     city

     famous

     for

     its

     E

    iff

    el

     Tower

    .

     French

     Vice

     President

     Nicolas

     Sark

    ozy

     is

     the

     Mayor

     of

     Paris

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     including

     the

     Lou

    vre

    ,

     which

     houses

     the

     Mona

     Lisa

     and

     the

     Se

    ine

    .

     Paris

     is

     renowned

     for

     its

     nightlife

     and

     world

    -ren

    owned

     landmarks

    .

     The

     city

     is

     also

     known

     for

     its

     diversity

     and

     multicultural

     population

    ,

     which

     has

     contributed

     to

     its

     cultural

     richness

    .

     The

     city

     is

     often

     called

     the

     "

    City

     of

     Light

    "

     due

     to

     its

     rich

     history

     and

     culture

    .

     Paris

     is

     a

     modern

     city

     known

     for

     its

     high

    -tech

     industries

     and

     access

     to

     international

     destinations

    ,

     including

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     dominated

     by

     new

     developments

     in

     areas

     such

     as

     deep

     learning

    ,

     natural

     language

     processing

    ,

     robotics

    ,

     and

     computer

     vision

    .

     With

     the

     increasing

     availability

     of

     large

     amounts

     of

     data

     and

     computing

     power

    ,

     AI

     systems

     are

     likely

     to

     become

     more

     sophisticated

     and

     capable

     of

     performing

     tasks

     that

     were

     previously

     thought

     to

     be

     impossible

    .

     Additionally

    ,

     as

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     may

     see

     more

     widespread

     adoption

     of

     AI

     in

     industries

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .

     Finally

    ,

     as

     AI

     becomes

     more

     complex

     and

     able

     to

     operate

     autonom

    ously

    ,

     we

     may

     see

     more

     innovative

     uses

     of

     AI

     in

     areas

     such

     as

     education

     and

     entertainment

    .

     Overall

    ,

     the

     future

     of

     AI

     looks

     bright

    ,

    



```python
llm.shutdown()
```
