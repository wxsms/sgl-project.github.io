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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.42it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.74it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.56it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.56it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.56it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.56it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.56it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.56it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.56it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.56it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 13.58it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 13.58it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 13.58it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 13.58it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:02, 13.58it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:02, 13.58it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:02, 13.58it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:02, 13.58it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:02, 13.58it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:02, 13.58it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 20.56it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 20.56it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 20.56it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 20.56it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 20.56it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 20.56it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 20.56it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 20.56it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 20.56it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 20.56it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:04<00:00, 28.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.15 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.15 GB):   2%|▏         | 1/58 [00:00<00:11,  5.13it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.12 GB):   2%|▏         | 1/58 [00:00<00:11,  5.13it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=58.11 GB):   2%|▏         | 1/58 [00:00<00:11,  5.13it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.11 GB):   2%|▏         | 1/58 [00:00<00:11,  5.13it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.11 GB):   7%|▋         | 4/58 [00:00<00:03, 13.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.11 GB):   7%|▋         | 4/58 [00:00<00:03, 13.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.11 GB):   7%|▋         | 4/58 [00:00<00:03, 13.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.10 GB):   7%|▋         | 4/58 [00:00<00:03, 13.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.10 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.09 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.87it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.09 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.09 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.08 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.08 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.08 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.08 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.07 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.07 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.27it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=58.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.06 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.18it/s]Capturing num tokens (num_tokens=960 avail_mem=58.05 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.18it/s] Capturing num tokens (num_tokens=896 avail_mem=58.05 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.18it/s]Capturing num tokens (num_tokens=832 avail_mem=58.05 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.18it/s]

    Capturing num tokens (num_tokens=832 avail_mem=58.05 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=768 avail_mem=58.04 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=704 avail_mem=58.04 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=640 avail_mem=58.04 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=576 avail_mem=58.04 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=512 avail_mem=58.02 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=512 avail_mem=58.02 GB):  50%|█████     | 29/58 [00:00<00:00, 38.10it/s]Capturing num tokens (num_tokens=480 avail_mem=58.04 GB):  50%|█████     | 29/58 [00:00<00:00, 38.10it/s]Capturing num tokens (num_tokens=448 avail_mem=58.03 GB):  50%|█████     | 29/58 [00:01<00:00, 38.10it/s]Capturing num tokens (num_tokens=416 avail_mem=58.03 GB):  50%|█████     | 29/58 [00:01<00:00, 38.10it/s]Capturing num tokens (num_tokens=384 avail_mem=58.03 GB):  50%|█████     | 29/58 [00:01<00:00, 38.10it/s]Capturing num tokens (num_tokens=352 avail_mem=58.02 GB):  50%|█████     | 29/58 [00:01<00:00, 38.10it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.02 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=320 avail_mem=58.02 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=288 avail_mem=58.02 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=256 avail_mem=58.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=240 avail_mem=58.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=224 avail_mem=58.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=224 avail_mem=58.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=208 avail_mem=58.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=192 avail_mem=58.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=176 avail_mem=58.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=160 avail_mem=58.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=144 avail_mem=57.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.31it/s]

    Capturing num tokens (num_tokens=144 avail_mem=57.99 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.88it/s]Capturing num tokens (num_tokens=128 avail_mem=57.99 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.88it/s]Capturing num tokens (num_tokens=112 avail_mem=57.99 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.88it/s]Capturing num tokens (num_tokens=96 avail_mem=57.99 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.88it/s] Capturing num tokens (num_tokens=80 avail_mem=57.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.88it/s]Capturing num tokens (num_tokens=64 avail_mem=57.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.88it/s]Capturing num tokens (num_tokens=64 avail_mem=57.98 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=48 avail_mem=57.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=32 avail_mem=57.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=28 avail_mem=57.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=24 avail_mem=57.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.34it/s]

    Capturing num tokens (num_tokens=20 avail_mem=57.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=20 avail_mem=57.96 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.85it/s]Capturing num tokens (num_tokens=16 avail_mem=57.96 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.85it/s]Capturing num tokens (num_tokens=12 avail_mem=57.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.85it/s]Capturing num tokens (num_tokens=8 avail_mem=57.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.85it/s] Capturing num tokens (num_tokens=4 avail_mem=57.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.85it/s]Capturing num tokens (num_tokens=4 avail_mem=57.95 GB): 100%|██████████| 58/58 [00:01<00:00, 35.79it/s]


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
    Generated text:  Stella and I'm a fantastic Musician, a good speaker and a very reliable engineer. My clients and I are customers of the BBC, MTV, CBT and many other radio stations, I have built an impressive archive of over 7000 tracks and an extensive library of music. I have a degree in Music from the University of the West of England, I have tutored music at the Student Music Centre at the University of London and I have been a member of the BBC Radio School of Music from 1999-2011.
    I'm the owner of this website, I'm Stella and I'm
    ===============================
    Prompt: The president of the United States is
    Generated text:  married to Ann and lives in the United States. She has a son named Andrew. Andrew has two brothers, and he and his brothers are all single. How many children does Andrew have?
    To determine how many children Andrew has, we need to follow these steps:
    
    1. Identify the number of children Andrew has.
    2. Confirm that Andrew has a son, and his son is the only male child.
    
    From the problem, we know:
    - Andrew has two brothers.
    - Andrew is the son of Ann.
    - Andrew is also a single child.
    
    Therefore, Andrew has 1 child. The problem does not mention any other children or the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a city in which part of the world? To determine the correct answer, I'll analyze the information step by step:
    
    1. Identify the capital of France: The capital of France is Paris.
    2. Identify the location: Paris is located in a country.
    3. Identify the country: France is a country in Western Europe.
    4. Identify the geographical feature: Paris is situated on the Côte d'Azur, a coastal region in the Mediterranean Sea.
    5. Identify the world: The Côte d'Azur is located in the southern part of the Mediterranean Sea, in the northernmost part of the
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the users. Here are some of the applications of AI that are to be developed.
    A) Image Recognition: Image recognition is a process where an AI system attempts to learn from a large amount of data and recognizes objects and people in images. It has many applications, including helping people with visual impairment, recognizing pets, and even helping law enforcement in solving crimes. There are many AI applications of image recognition, and the development of these applications will increase in the future.
    B) Recommendation Systems: Recommendation systems are a type of AI that uses a large amount of data to provide recommendations to users. The systems use algorithms to analyze


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting the annual Eiffel Tower Tour and hosting the World Cup of football. Paris is a popular tourist destination and is home to many famous museums, including the Louvre and the Musée d'Orsay. The city is also known for its cuisine, including its famous croissants and its traditional French wine. Paris is a vibrant and dynamic city with a rich history and a diverse population. Its status as the capital of France is recognized by
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well
    


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
    Generated text:  Emily. I am a friendly, outgoing, and compassionate person who enjoys reading, writing, and exploring new experiences. I am also very passionate about helping others, and I am always willing to lend a hand whenever I can. So if you need anything, don't hesitate to reach out to me. I look forward to meeting you! 😊😊😊
    
    I'm new to the game, but I'm up for any role. Can you give me a sample line to describe my character to someone new? Sure! Here's an example line:
    
    "Hey there! I'm Emily, a friendly, outgoing, and compassionate character who enjoys
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic and culturally rich city located on the Seine River, known for its stunning architecture, rich history, and vibrant cultural life. Its status as the nation's political, economic, and cultural capital is recognized globally. Paris is also known for its famous landmarks such as Notre-Dame Cathedral, Louvre Museum, Eiffel Tower, and many more. Additionally, the city is home to numerous museums, theaters, and other cultural institutions, providing visitors with an unparalleled experience of France's unique cultural landscape. Paris has become one of the world's most popular tourist destinations, attracting millions of visitors every year. As one of the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid and significant advancements in several key areas, driven by increasing computational power, data availability, and the growth of connected devices. Here are some potential trends to watch out for in AI:
    
    1. Deep Learning: The development of deeper learning models, with more layers and more complex architectures, is likely to drive more accurate and efficient AI. This trend is already evident in applications such as natural language processing and computer vision, where deep learning is already widely used.
    
    2. Natural Language Processing: As AI technology continues to advance, the ability to understand and generate human language will likely become more accurate and natural. This trend is


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

    /an

     [

    occupation

    /

    field

    ]

     specialist

    ,

     working

     to

     improve

     the

     quality

     of

     life

     for

     those

     around

     me

    .

     I

     enjoy

     sharing

     my

     knowledge

     and

     experience

     through

     workshops

     and

     training

     sessions

    ,

     and

     mentoring

     others

     as

     needed

    .

     My

     approach

     to

     problem

    -solving

     is

     collaborative

    ,

     and

     I

    'm

     always

     open

     to

     learning

     new

     things

    .

     I

    'm

     confident

     in

     my

     abilities

     and

     enjoy

     the

     challenge

     of

     helping

     others

     achieve

     their

     goals

    .

     Let

     me

     know

     if

     you

    'd

     like

     to

     meet

     me

     in

     person

     or

     through

     email

     to

     discuss

     possible

     collaboration

    .

     [

    Name

    ]

     [

    Contact

     Information

    ]

     (

    Feel

     free

     to

     add

     any

     other

     relevant

     information

     you

    'd

     like

     to

     include

    )

     Hello

    ,

     my

     name

     is

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     in

     the

     country

     and

     home

     to

     many

     of

     France

    ’s

     most

     famous

     landmarks

     and

     cultural

     institutions

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     iconic

     buildings

    ,

     and

     lively

     cultural

     scene

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

    ,

     but

     there

     are

     several

     trends

     that

     could

     shape

     the

     technology

     in

     the

     years

     to

     come

    .

     Some

     of

     these

     trends

     include

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     The

     integration

     of

     AI

     with

     other

     technologies

     such

     as

     machine

     learning

    ,

     blockchain

    ,

     and

     cybersecurity

     could

     lead

     to

     more

     complex

     and

     sophisticated

     AI

     systems

    .
    


    2

    .

     Adv

    ancements

     in

     natural

     language

     processing

    :

     Advances

     in

     natural

     language

     processing

     could

     lead

     to

     more

     advanced

     AI

     systems

     that

     can

     understand

     and

     interpret

     human

     language

     in

     new

     ways

    .
    


    3

    .

     Improved

     privacy

     and

     data

     protection

    :

     As

     AI

     becomes

     more

     prevalent

    ,

     there

     will

     be

     a

     growing

     need

     for

     improved

     privacy

     and

     data

     protection

     measures

     to

     protect

     the

     privacy

     of

     individuals

     and

     organizations

    .
    


    4

    .

    



```python
llm.shutdown()
```
