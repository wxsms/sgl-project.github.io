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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:33,  4.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:33,  4.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:33,  4.80s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:33,  4.80s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:33,  4.80s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.63it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.30it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.30it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.30it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.30it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.30it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.30it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.30it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.30it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.30it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.30it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.30it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 21.98it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 21.98it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 21.98it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 21.98it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 21.98it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 21.98it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 21.98it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 21.98it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 21.98it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 21.98it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.31it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 37.54it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.80it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.80it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.80it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.97it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.46it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.99it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.99it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.99it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.99it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.99it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.99it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:00<00:00, 43.47it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:00<00:00, 43.47it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.47it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.47it/s]

    Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.47it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.47it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.28it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.28it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.28it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.28it/s]

    Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.28it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.28it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.25it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 39.29it/s]


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
    Generated text:  Dean, and I am the head of the math department at a fictional high school. I will be speaking to a group of high school students interested in mathematics and would like you to write an essay on the topic. The topic is "The Impact of Mathematics on Daily Life and Career." You must provide a detailed essay that covers the following points: 1. Discuss the role of mathematics in everyday life, including basic arithmetic, algebra, calculus, and statistics. 2. Explain how mathematics is used in career fields such as science, technology, and engineering. 3. Discuss the importance of mathematics in education, such as in the development
    ===============================
    Prompt: The president of the United States is
    Generated text:  a three - member committee that decides which of the two major parties will be the country's official party. It is formed by the Democratic and Republican parties. Its members are selected by the people who vote in the national election. The president is the leader of the state's government, but it's the official party that will have the final say in all the important decisions of the country.
    Can we conclude that is the president of the united states a member of the democratic party? A single sentence to support your answer.
    
    To determine if the president of the United States is a member of the Democratic Party, we need to consider the composition of the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and it is located in which continent?
    A continent
    B. Asia
    C. Europe
    D. North America
    
    To determine where the capital of France is located, we need to follow these steps:
    
    1. Identify the capital of France.
    2. Confirm the capital of France is in Europe.
    
    Step 1: Identify the capital of France.
    The capital of France is Paris.
    
    Step 2: Confirm the capital of France is in Europe.
    Europe includes countries such as Germany, Italy, Belgium, the Netherlands, and Switzerland. None of these are directly located in France. The capital of France is also not located in any
    ===============================
    Prompt: The future of AI is
    Generated text:  a rapidly evolving field, with significant advancements expected to reshape the way we live, work, and interact with technology. As the AI landscape continues to evolve, it is important to understand the different components of artificial intelligence (AI) and how they contribute to the development of AI systems. In this article, we will explore the different components of AI and how they work together to create advanced AI systems that can perform a wide range of tasks and solve complex problems.
    1. Machine Learning
    Machine learning is one of the most popular components of AI, and it is used to train machines to learn from data and improve their performance over time. Machine learning


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working in this field for [number of years] years. I am passionate about [reason for being in the field]. I am always looking for new challenges and opportunities to grow and learn. I am a [character trait] and I am always ready to help others. I am [character trait] and I am always willing to go above and beyond to make a positive impact. I am [character trait] and I am always willing to take risks and try new things. I am [character trait] and I am always ready to adapt to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Academy of Sciences. Paris is a bustling metropolis with a rich cultural heritage and is a major economic and political center in Europe. It is also known for its fashion industry and its role in hosting major international events such as the Olympics and the World Cup. Paris is a city that is both a cultural and historical center, and is a popular tourist destination for many visitors. It is a city that is known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into our daily lives, from manufacturing to healthcare. We may see more automation in manufacturing, where machines will be able to perform tasks that were previously done by humans. We may also see more automation in healthcare, where AI will be used to diagnose and treat diseases more accurately and efficiently.
    
    2. Enhanced human-AI collaboration: As AI becomes more integrated into our lives, we may see more collaboration between humans and
    


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
    Generated text:  [Your Name]. I'm a [Your Profession] who has always been passionate about [Your Hobby, Interest, or Passion]. I enjoy [Your Skill or Trait], and I'm always looking for new challenges and opportunities to grow as a professional. I believe in [Your Core Belief, Philosophy, or Ethic], and I strive to be [Your Ideal Character Traits or Values]. So, if you'd like to know more about me, I'm here to share my experiences, skills, and journey. How can I help you today? I look forward to hearing from you! 🌟✨🌟
    Yourself:
    **
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That statement is accurate, and here are a few additional details to provide a more comprehensive answer:
    
    Paris is the capital city of France, located in the Paris region on the left bank of the Seine River in eastern France. The city is known for its rich history, beautiful architecture, and iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum. Paris is also the cultural and economic hub of France, hosting numerous famous landmarks, museums, theaters, and events throughout the year. The city is known for its vibrant nightlife, delicious food, and world-renowned museums and art galleries. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast, and there are many potential trends that could shape the technology and applications that it will be used for. Some possible trends include:
    
    1. Integration of AI into other industries: As AI becomes more prevalent in various fields, it is likely to be integrated into a wider range of industries, from healthcare and finance to transportation and manufacturing.
    
    2. AI as a commodity: As AI becomes more accessible and affordable, it is possible that AI will become a commodity, available to anyone with a computer and enough software to use it.
    
    3. AI as a tool for social good: As AI is used to improve the lives of people, it could


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

    Occup

    ation

    ].

     I

     am

     here

     to

     provide

     you

     with

     a

     unique

     experience

     that

     will

     change

     your

     life

     for

     the

     better

    .

     Tell

     me

     a

     little

     bit

     about

     yourself

    ,

     and

     I

     will

     tailor

     my

     experience

     to

     fit

     your

     needs

    .

     How

     would

     you

     like

     to

     begin

    ?

     Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

     am

     a

     [

    Occup

    ation

    ].

     I

     am

     here

     to

     provide

     you

     with

     a

     unique

     experience

     that

     will

     change

     your

     life

     for

     the

     better

    .

     Tell

     me

     a

     little

     bit

     about

     yourself

    ,

     and

     I

     will

     tailor

     my

     experience

     to

     fit

     your

     needs

    .

     How

     would

     you

     like

     to

     begin

    ?

     [

    Tell

     me

     a

     little

     bit

     about

     yourself

    .]

     [

    Tell

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     most

     populous

     city

     in

     the

     country

    .

     It

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

     and

     is

     a

     major

     cultural

    ,

     artistic

    ,

     and

     financial

     center

    .

     Paris

     is

     home

     to

     many

     world

    -ren

    owned

     museums

    ,

     landmarks

    ,

     and

     iconic

     buildings

    ,

     including

     Notre

    -D

    ame

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

     Museum

    .

     The

     city

     is

     also

     known

     for

     its

     artistic

     and

     intellectual

     scene

    ,

     with

     many

     notable

     composers

    ,

     writers

    ,

     and

     artists

     residing

     or

     working

     in

     Paris

    .

     Paris

    's

     status

     as

     the

     capital

     is

     due

     to

     its

     strategic

     location

     in

     the

     center

     of

     Europe

     and

     its

     rich

     history

     and

     cultural

     heritage

    .

     It

     is

     also

     home

     to

     several

     international

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     trends

     and

     developments

    ,

     including

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

     and

     deep

     learning

    :

     These

     technologies

     are

     likely

     to

     continue

     to

     improve

    ,

     leading

     to

     more

     accurate

     and

     sophisticated

     AI

     systems

    .
    


    2

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     concerns

     about

     the

     potential

     dangers

     of

     AI

     increase

    ,

     there

     will

     likely

     be

     more

     efforts

     to

     address

     these

     concerns

     and

     develop

     AI

     that

     is

     both

     useful

     and

     ethical

    .
    


    3

    .

     Expansion

     of

     AI

     to

     more

     industries

    :

     As

     the

     economic

     and

     technological

     landscape

     changes

    ,

     it

     is

     likely

     that

     AI

     will

     be

     more

     widely

     deployed

     across

     a

     wider

     range

     of

     industries

    .
    


    4

    .

     Greater

     use

     of

     AI

     for

     decision

    -making

    :

     As

     AI

     systems

    



```python
llm.shutdown()
```
