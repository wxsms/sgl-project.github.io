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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.97it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.97it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.97it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.97it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.97it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.97it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.97it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  8.97it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  8.97it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.97it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.81it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.81it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.81it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.81it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.81it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.81it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.81it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.81it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.81it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.81it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.81it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.76it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.76it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.76it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.76it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.76it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.76it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.76it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.76it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.76it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.76it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.76it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.02it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.02it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.02it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.58it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.58it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.58it/s] Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.59it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.98it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.98it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.98it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.98it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.98it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.98it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.82it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.82it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.82it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.82it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.82it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.82it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.97it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=128 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.57it/s] Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=80 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 43.67it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=32 avail_mem=74.35 GB):  81%|████████  | 47/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.31it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 38.39it/s]


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
    Generated text:  Alexis and I am a college student. I was born in the United States and have lived my whole life in the United Kingdom. My English is very good and I have been to many universities to study English Literature and English Language, with a major in Contemporary English Literature. I have been a British writer for many years and have had the privilege of learning and enjoying many diverse literary experiences such as visiting, traveling and living in countries around the world. I have also been a teacher in London for many years, teaching subjects including English, Spanish, French and German. I believe that literature is a universal language that everyone can understand and appreciate. My
    ===============================
    Prompt: The president of the United States is
    Generated text:  an executive officer of the United States government. The current president of the United States is Donald Trump. Based on the above information, please answer: The president of the United States was born in the year ____. 
    A. 1776 
    B. 1789 
    C. 1876 
    D. 1889 
    Answer:
    
    A
    
    Which of the following sets of numbers is both prime and odd?
    A. 7
    B. 2
    C. 1
    D. 0
    Answer:
    
    A
    
    [Multiple Choice Question] Which of the following statements about protein
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is a city with a population of over 6 million people, making it the third largest city in the world by population. It is also the capital of the department of the Loire. Its climate is generally cool and wet during the wet season, which can be from May to July, and warm and dry during the dry season, which can be from September to December. The capital of France has a population of more than 6 million people, making it the third largest city in the world by population. It is also the capital of the department of the Loire. Its climate is generally cool and wet during the wet
    ===============================
    Prompt: The future of AI is
    Generated text:  expected to change rapidly as technology improves and applications become more complex. AI systems are not just about machine learning, but they also involve the use of algorithms, data storage, and computational power. Therefore, it is essential to understand the basics of AI systems before diving into the complex and technical aspects of implementing them.
    
    In this article, we will discuss the basics of AI systems and how they are being used in today’s world. We will also explore some of the most popular and advanced AI technologies and how they are being used in various industries.
    
    Firstly, let’s understand what is AI and how it works:
    
    ### What is AI?
    
    AI


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and economic hub, hosting numerous world-renowned museums, theaters, and festivals. Paris is a popular tourist destination and a major center for business and commerce. The city is known for its rich history, diverse culture, and vibrant nightlife. It is the largest city in France and a major economic and political center. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has been a center of French culture and identity for centuries. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and artificial intelligence: As automation and AI become more prevalent, we are likely to see more jobs being automated, which could lead to job displacement. However, this could also create new opportunities for people to work in areas such as data analysis, machine learning, and robotics.
    
    2. Improved privacy and security: As AI becomes more advanced, we are likely to see more sophisticated ways of collecting and analyzing data
    


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
    Generated text:  [Name]. I'm a [age] year old who has been [job title] for [number] years. I enjoy [one or more hobbies or interests] and have been a [subject] for the past [number] years. I believe in [values or beliefs] and am committed to [my career goals]. I am [age] years old and I enjoy [a hobby or interest]. I am a [occupation] who has been [number] years in [job title]. I strive to be [one or more qualities or traits] in everything I do. I am [one or more attributes or qualities], and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The statement is: Paris is the capital of France. 
    
    A more precise statement could be: Paris is the capital city of France. 
    
    Both statements convey the same information but use slightly different wording. The first one is a straightforward declarative statement, while the second one includes the term "city" which could potentially refer to a smaller political entity in some contexts. However, for the purposes of a concise factual statement, the first option is more accurate and universally understood. 
    
    Please note that the capital city of France is often referred to as "Paris" or simply "Paris". 
    
    Does this response meet your needs? If so
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve significant advances in several key areas. Here are some possible trends in the near and long term:
    
    1. Natural Language Processing: AI will continue to advance in natural language processing, allowing machines to understand and interpret human language more accurately. This will enable robots and other AI-powered devices to understand human emotions and intentions, and to communicate with people in a more natural way.
    
    2. Deep Learning: Deep learning is a subset of machine learning that involves training large neural networks to perform tasks that are currently difficult or impossible for humans to solve. This is expected to become more prevalent as AI applications continue to grow, including in healthcare and autonomous


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

    First

     Name

    ]

     and

     I

    'm

     [

    Last

     Name

    ].

     I

    'm

     an

     AI

     language

     model

     created

     by

     Anth

    ropic

    .

     I

    'm

     here

     to

     assist

     you

     with

     any

     questions

     or

     tasks

     you

     have

    ,

     and

     I

    'm

     always

     here

     to

     help

    .

     If

     you

     need

     anything

    ,

     just

     let

     me

     know

    .

     #

    Anth

    ropic

     #

    AI

     #

    Language

    Model

     #

    Personal

    AI

     #

    Chat

    bot

     #

    Tech

    G

    uru

     #

    G

    aming

    Fan

     #

    Tech

    L

    over

     #

    Tech

    C

    ult

    ivation

     #

    Tech

    Insp

    iration

     #

    Tech

    L

    over

     #

    Tech

    G

    uru

     #

    Tech

    G

    uru

     #

    Tech

    G

    uru

     #

    Tech

    G

    uru

     #

    Tech

    G

    uru

     #

    Tech

    G

    uru

     #

    Tech

    G

    uru

     #

    Tech

    G

    uru

     #

    Tech

    G

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     "

    City

     of

     Light

    "

     and

     the

     "

    Emp

    ire

     State

    "

     due

     to

     its

     historical

     significance

     and

     cultural

     importance

    .

     Located

     in

     the

     heart

     of

     the

     French

     Riv

    iera

    ,

     Paris

     is

     home

     to

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

     many

     other

     world

    -ren

    owned

     landmarks

    .

     Its

     charming

     rivers

    ide

     area

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Op

    éra

     Garn

    ier

     are

     also

     popular

     tourist

     destinations

    .

     Paris

     is

     also

     known

     for

     its

     rich

     cultural

     and

     artistic

     scene

    ,

     with

     many

     famous

     artists

    ,

     writers

    ,

     and

     musicians

     from

     around

     the

     globe

    .

     The

     city

     has

     a

     diverse

     population

    ,

     with

     around

     

    1

    0

     million

     residents

     as

     of

     

    2

    0

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     rapidly

     evolving

     landscape

    ,

     shaped

     by

     a

     multitude

     of

     factors

    .

     Some

     potential

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     Integration

     with

     Human

     Decision

    -M

    aking

    :

     AI

     systems

     are

     expected

     to

     become

     more

     integrated

     with

     human

     decision

    -making

    ,

     enabling

     more

     nuanced

     and

     holistic

     analysis

    .
    


    2

    .

     AI

     Ethics

     and

     Bias

    :

     The

     ethical

     and

     social

     implications

     of

     AI

     will

     become

     increasingly

     important

    .

     As

     AI

     systems

     are

     used

     in

     a

     wide

     range

     of

     contexts

    ,

     it

     is

     crucial

     to

     ensure

     that

     they

     are

     developed

     and

     deployed

     in

     a

     way

     that

     is

     fair

     and

     unbiased

    .
    


    3

    .

     AI

     as

     a

     Tool

     for

     Innovation

    :

     AI

     is

     already

     playing

     a

     significant

     role

     in

     innovation

    ,

     from

     drug

     development

     to

     financial

     analysis

    



```python
llm.shutdown()
```
