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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.69it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:11,  4.15it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.34it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.34it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.31it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.31it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.31it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.31it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.31it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.31it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.31it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.31it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.31it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.31it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.31it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.12it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.12it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 30.88it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 30.88it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 30.88it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 30.88it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 30.88it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 30.88it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 30.88it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 30.88it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 30.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.74it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 24.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 24.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:02, 24.46it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:02, 24.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:02, 24.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 34.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 34.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 34.41it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.10 GB):  31%|███       | 18/58 [00:00<00:01, 34.41it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 34.41it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 34.41it/s]Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.36it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.36it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.08 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.36it/s]Capturing num tokens (num_tokens=448 avail_mem=74.07 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.36it/s]Capturing num tokens (num_tokens=416 avail_mem=74.07 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.36it/s]Capturing num tokens (num_tokens=384 avail_mem=74.07 GB):  48%|████▊     | 28/58 [00:01<00:00, 40.36it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.07 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.35it/s]Capturing num tokens (num_tokens=352 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.35it/s]Capturing num tokens (num_tokens=320 avail_mem=76.66 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.35it/s]Capturing num tokens (num_tokens=288 avail_mem=76.64 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.35it/s]Capturing num tokens (num_tokens=256 avail_mem=76.57 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.35it/s]Capturing num tokens (num_tokens=256 avail_mem=76.57 GB):  64%|██████▍   | 37/58 [00:01<00:00, 27.74it/s]Capturing num tokens (num_tokens=240 avail_mem=76.16 GB):  64%|██████▍   | 37/58 [00:01<00:00, 27.74it/s]Capturing num tokens (num_tokens=224 avail_mem=76.07 GB):  64%|██████▍   | 37/58 [00:01<00:00, 27.74it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 27.74it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 27.74it/s]

    Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 27.74it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.88it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.88it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.88it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.88it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.88it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.88it/s] Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  81%|████████  | 47/58 [00:01<00:00, 35.44it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 35.44it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 35.44it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 35.44it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 35.44it/s]

    Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 35.44it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.15it/s] Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 33.65it/s]


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
    Generated text:  Daniel, and I work as a sales manager in a local office.
    
    Please write me a 500-word personal marketing strategy to help me increase my sales.
    
    Sales Strategy for Daniel: Daniel's Personal Marketing
    
    One of Daniel's primary responsibilities is sales, but he also has a passion for learning and enhancing his skills, so he believes in taking an active role in his professional development. Therefore, he has set up a personal marketing strategy that includes two main components:
    
    1. Social media marketing:
    Daniel uses social media platforms such as Facebook, Instagram, and Twitter to showcase his products, share content, and engage with potential customers. He
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. There are two primary candidates in the race: candidate A and candidate B. The candidate with the better overall performance score will win the election. The president is going to give the candidate with the higher overall performance score a larger budget than the other candidate. 
    
    If the president decides to use a budget of $100,000 for the election, and the president wants candidate A to have a higher score than candidate B, what is the minimum score the president needs to achieve in order to ensure that candidate A wins the election, given that candidate B has a score of 2000 on
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and the population is around 2.7 million. What is the area of Paris?
    
    To determine the area of Paris, we need to follow these steps:
    
    1. Identify the shape of the city: Paris is a large square city with a square-shaped city center surrounded by a series of residential areas.
    2. Calculate the side length of the square:
       - Paris has a square-shaped city center, which is the largest square in the city.
       - Let's assume the side length of the city center is \( s \).
       - The area of the city center, which is a square, is \( s^2
    ===============================
    Prompt: The future of AI is
    Generated text:  in machine learning. The future of AI is in machine learning. The future of AI is in machine learning.
    
    Select your answer from the options. What is the sentiment of this tweet?
    OPTIONS: +negative +positive
    
    The sentiment of this tweet is positive. The tweet expresses enthusiasm and excitement about the future of artificial intelligence (AI), particularly machine learning. It uses positive language like "future of AI" and "machine learning" to convey a positive sentiment. The other option is not correct because there is no negative sentiment expressed in the tweet. Therefore, the answer is +positive.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. The city is known for its vibrant nightlife, art scene, and diverse food and beverage options. Paris is a popular tourist destination, attracting millions of visitors each year. The city is also home to many international organizations and institutions, including UNESCO and the European Union
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient care.
    
    2. Greater integration of AI into everyday life: As AI technology becomes more advanced, we can expect to see even more integration into our daily lives.
    


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
    Generated text:  [Name]. I'm a person who loves to explore the world and discover new things. I'm also someone who is very friendly and easy to talk to. I have a natural curiosity and a love for learning new things, and I'm always eager to learn about different cultures, cuisines, languages, and historical events. I enjoy meeting new people and trying to understand their perspectives. I love to travel, to explore new places, to try new foods, to listen to different cultures, and to learn about history and science. I'm always looking for new experiences and adventures, and I'm always eager to share my knowledge with others.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Here's the factual statement in the form of a JavaScript code snippet:
    
    ```javascript
    const capitalOfFrance = "Paris";
    console.log(`The capital of France is ${capitalOfFrance}.`);
    ```
    
    This code snippet defines the capital city of France as "Paris" and prints it out. You can modify the capital city name by replacing `"Paris"` with another name. The output will be a message stating the capital city of France. 
    
    For example:
    ```javascript
    // Output: The capital of France is Paris.
    ``` 
    
    This code is a simple example in JavaScript that demonstrates how to store and print a capital city name
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be shaped by several key trends that are expected to shape the future of the field. Here are some of the most promising trends:
    
    1. The rise of deep learning: Deep learning is the primary driving force behind the progress of AI. Deep learning is the ability of an AI system to learn and improve its performance by training on large, complex datasets. As deep learning techniques continue to improve, the ability to achieve complex tasks will continue to increase.
    
    2. The integration of AI into everyday life: AI is already transforming the way we live our lives, from self-driving cars to virtual assistants that can assist with tasks such as grocery shopping


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

     [

    Name

    ]

    !

     I

     am

     a

     [

    Occup

    ation

    ]

     with

     [

    Title

    ]

     at

     [

    Company

    ].

     I

     love

     [

    Purpose

     of

     Your

     Job

    ],

     and

     I

     strive

     to

     [

    Achie

    ve

     Your

     Goal

    /

    Per

    fection

    ].

     My

     dedication

     to

     my

     job

     and

     my

     love

     for

     my

     field

     are

     infectious

     and

     inspiring

    .

     I

    'm

     here

     to

     help

     you

     achieve

     your

     goals

     and

     make

     your

     life

     better

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     ...

     And

     that

    's

     a

     very

     short

    ,

     neutral

     self

    -int

    roduction

    !

     Can

     you

     please

     provide

     more

     details

     on

     your

     occupation

    ,

     title

    ,

     company

    ,

     and

     purpose

     of

     your

     job

    ?

     That

     way

    ,

     I

     can

     tailor

     my

     response

     to

     your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     

    1

    8

    th

     largest

     city

     in

     the

     world

     by

     population

    ,

     located

     in

     the

     south

     of

     the

     country

    .


    France

     is

     a

     large

     country

     with

     a

     rich

     history

     and

     diverse

     culture

    .

     The

     capital

     city

     of

     Paris

     is

     known

     for

     its

     iconic

     landmarks

     and

     museums

    ,

     as

     well

     as

     its

     elegant

     and

     historic

     architecture

    .

     French

     cuisine

     is

     a

     beloved

     part

     of

     the

     French

     culture

    ,

     with

     dishes

     such

     as

     cro

    iss

    ants

    ,

     esc

    arg

    ot

    ,

     and

     cr

    ê

    pes

     being

     popular

    .

     France

     is

     also

     known

     for

     its

     fine

     art

    ,

     with

     museums

     such

     as

     the

     Lou

    vre

     and

     the

     Centre

     Pom

    pid

    ou

     serving

     as

     iconic

     landmarks

    .

     Paris

     is

     a

     beautiful

     city

     with

     a

     rich

     cultural

     and

     artistic

     heritage

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     full

     of

     potential

    .

     Here

     are

     some

     possible

     trends

     that

     could

     emerge

     in

     the

     near

     future

    :
    


    1

    .

     Increased

     AI

     integration

     with

     other

     technologies

    :

     As

     AI

     continues

     to

     evolve

    ,

     it

     is

     likely

     to

     become

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .

     This

     integration

     could

     lead

     to

     new

     applications

     of

     AI

     and

     enable

     more

     efficient

     and

     personalized

     service

     to

     users

    .
    


    2

    .

     AI

     becoming

     more

     human

    -like

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     possible

     that

     it

     will

     become

     more

     human

    -like

    ,

     with

     emotions

     and

     intelligence

    .

     This

     could

     lead

     to

     AI

     systems

     that

     are

     able

     to

     understand

     and

     respond

     to

     complex

     human

     emotions

     and

     personalities

    .
    


    3

    .

     AI

     becoming

     more

    



```python
llm.shutdown()
```
