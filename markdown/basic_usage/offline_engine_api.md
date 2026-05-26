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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.99it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.99it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 25.02it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 34.42it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 34.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 36.62it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 36.62it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 36.62it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.96it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.96it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.96it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.96it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.96it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.96it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.99it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.23it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.23it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.23it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.23it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.05it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.10it/s]

    Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.92it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.27it/s]


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
    Generated text:  Maria, and I am 28 years old. I have a little girl named Bella, 10 years old.
    
    Bella's name is the same as my last name. However, Bella's younger sister, Sarah, is also named after my last name.
    
    What is the last name of the person who has a little girl named Bella? Maria has a little girl named Bella, 10 years old. 
    
    Therefore, the last name of the person who has a little girl named Bella is Maria. Maria has a little girl named Bella, who is 10 years old. 
    
    Maria is 28 years old,
    ===============================
    Prompt: The president of the United States is
    Generated text:  interested in calculating the number of presidential tweets he would receive in a year if he posts tweets every day. On average, he posts tweets 6 times per day. Given that there are 365 days in a year, how many tweets does the president post in a year?
    
    To determine the number of presidential tweets the president of the United States would receive in a year, we need to follow these steps:
    
    1. Identify the average number of tweets posted per day.
    2. Multiply the daily average by the number of days in a year.
    
    Step 1: The president posts tweets 6 times per day.
    Step 2:
    ===============================
    Prompt: The capital of France is
    Generated text:  _________. Paris
    
    Based on the context provided, the capital of France is Paris. Paris is the capital city of France. The question does not contain any contradictions with historical facts or commonly known facts, so it is correct.
    
    Therefore, the answer is: $\boxed{\text{Paris}}$.
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. However, the large amount of information in today’s world makes it hard for AI to understand and interpret everything. It requires a significant amount of time and effort to learn how to use AI to understand the meaning of words and phrases. This can make it difficult to effectively communicate and collaborate with others. Additionally, there are some individuals who may not be comfortable with AI and prefer to work with humans.
    
    To address these issues, it is important to provide clear and concise communication, and use natural language processing techniques to understand the context of the conversation. Additionally, it is also important to provide adequate training and resources to ensure that all participants have


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm a [Skill] who has been [Number of Years] years in the industry. I'm a [Skill] who has been [Number of Years] years in the industry. I'm a [Skill] who has been [Number of Years] years in the industry. I'm a [Skill] who has been [Number of Years] years in the industry. I'm a [Skill] who has been [Number of Years] years in the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Parliament building. Paris is a bustling city with a rich history and culture, and it is a popular tourist destination. Its location in the heart of the French countryside makes it a popular destination for tourists and locals alike. The city is known for its cuisine, including its famous French fries and its traditional French wine. Paris is a city that is a must-visit for anyone interested in French culture and history.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will likely lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to advance, we can expect to see even more applications in healthcare,
    


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
    Generated text:  [Name], and I am a [Age] year old, [Occupation] [Position]. I come from [Location], a [Country], and I am [Nationality], [Gender]. I have a [Skill or Habit], [Description], and I am [Yourself]. I am a [Role], [Position] [Role], where [Role] refers to a specific role in my life. I am a [Favorite Activity], [Favorite Food], [Favorite Clothing], or [Favorite Album]. I am a [Yourself]. As I know, my [Self-Personality], [Yourself], and [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city where Napoleon Bonaparte established his iconic Bastille fortress and where the Eiffel Tower is located. Despite being a large, sprawling city, it remains a UNESCO World Heritage site for its historical significance. The French capital is located on the Île de la Cité, which is known for its narrow streets, bustling cafes, and stunning views of the surrounding countryside. Paris is home to many world-renowned museums, including the Louvre and the Musée d'Orsay, as well as a variety of dining and entertainment options. The city's vibrant culture, including its annual Bastille Day celebrations and a rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by rapid technological advancements, as well as increasing ethical and social considerations. Here are some potential future trends in AI:
    
    1. Greater automation: One of the biggest trends in AI is automation, which could lead to the development of more efficient and less human-operated systems.
    
    2. AI-powered healthcare: AI is already being used in the healthcare industry to help diagnose and treat diseases. As AI technology continues to advance, it is likely to become more advanced and be used in even more areas of healthcare.
    
    3. AI in finance: AI is already being used in finance to assist in fraud detection and risk management. As AI


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

    ’m

     a

     [

    Job

     Title

    /

    Role

    ].

     I

     currently

     live

     in

     [

    City

    ].

     I

    ’ve

     always

     been

     passionate

     about

     [

    Your

     Passion

    /

    Interest

    ].

     I

     enjoy

     [

    My

     Primary

     Hobby

    /

    Activity

    ].

     In

     my

     free

     time

    ,

     I

    ’m

     really

     good

     at

     [

    What

     I

     Love

     to

     Do

    ].

     How

     do

     you

     spend

     your

     time

     outside

     of

     work

    ?

     A

    )

     Trying

     to

     learn

     new

     things

    .


    B

    )

     Working

     out

     at

     the

     gym

    


    C

    )

     Social

    izing

     with

     friends

     or

     family

    


    D

    )

     Taking

     a

     hike

     or

     playing

     basketball

     outside

    
    


    Thank

     you

    .

     You

    're

     welcome

    !

     What

     is

     your

     job

     title

    /

    role

    ,

     and

     where

     do

     you

     currently

     live

    ?

     I

    'm

     curious

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

    ,

     cultural

     institutions

    ,

     and

     rich

     history

    .

     
    


    -

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     and

     is

     the

     largest

     city

     in

     the

     country

    .


    -

     The

     capital

     is

     located

     in

     the

     north

    western

     part

     of

     the

     country

    .


    -

     It

     is

     the

     capital

     city

     of

     France

    ,

     and

     is

     known

     for

     its

     beautiful

     architecture

    ,

     vibrant

     streets

    ,

     and

     rich

     cultural

     heritage

    .


    -

     Paris

     is

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

     and

     is

     known

     for

     its

     annual

     Spring

     Festival

     and

     Summer

     Olympics

    .


    -

     The

     city

     has

     a

     complex

     history

     and

     is

     home

     to

     many

     famous

     historical

     sites

    ,

     including

     Notre

     Dame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     innovation

    ,

     collaboration

    ,

     and

     transformation

     of

     existing

     industries

    .

     Here

     are

     some

     potential

     trends

    :
    


    1

    .

     Increased

     focus

     on

     ethics

     and

     fairness

    :

     As

     more

     companies

     and

     governments

     recognize

     the

     need

     for

     AI

     to

     be

     used

     eth

    ically

     and

     fairly

    ,

     the

     focus

     on

     ethical

     and

     fairness

     considerations

     will

     continue

     to

     grow

    .

     This

     will

     likely

     lead

     to

     more

     robust

     algorithms

     that

     can

     be

     programmed

     to

     consider

     biases

     and

     create

     a

     more

     accurate

     representation

     of

     the

     world

    .
    


    2

    .

     AI

     will

     become

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

     expected

     to

     become

     more

     and

     more

     like

     humans

     in

     terms

     of

     intelligence

     and

     learning

     ability

    .

     This

     will

     likely

     lead

     to

     more

     sophisticated

     forms

     of

    



```python
llm.shutdown()
```
