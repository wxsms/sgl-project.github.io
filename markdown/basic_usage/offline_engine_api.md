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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.44it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.45it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.45it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.25it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.25it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.25it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.25it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.25it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.25it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.25it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.25it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.25it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.25it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.19it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.19it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.19it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 40.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.57 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.57 GB):   3%|▎         | 2/58 [00:00<00:02, 19.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.57 GB):   3%|▎         | 2/58 [00:00<00:02, 19.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.56 GB):   3%|▎         | 2/58 [00:00<00:02, 19.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.56 GB):   3%|▎         | 2/58 [00:00<00:02, 19.27it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.56 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.56 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.55 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.54 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.54 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.54 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.54 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.53 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.53 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.06it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.53 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.06it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.53 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.53 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.52 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.73it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.52 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.73it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.52 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.73it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.51 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.51 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.49 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=960 avail_mem=71.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s] Capturing num tokens (num_tokens=896 avail_mem=71.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]

    Capturing num tokens (num_tokens=832 avail_mem=71.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=832 avail_mem=71.50 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=768 avail_mem=71.49 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=704 avail_mem=71.49 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=640 avail_mem=71.49 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=576 avail_mem=71.49 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=512 avail_mem=71.47 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=512 avail_mem=71.47 GB):  50%|█████     | 29/58 [00:00<00:00, 41.86it/s]Capturing num tokens (num_tokens=480 avail_mem=71.48 GB):  50%|█████     | 29/58 [00:00<00:00, 41.86it/s]Capturing num tokens (num_tokens=448 avail_mem=71.48 GB):  50%|█████     | 29/58 [00:00<00:00, 41.86it/s]Capturing num tokens (num_tokens=416 avail_mem=71.48 GB):  50%|█████     | 29/58 [00:00<00:00, 41.86it/s]

    Capturing num tokens (num_tokens=384 avail_mem=71.48 GB):  50%|█████     | 29/58 [00:00<00:00, 41.86it/s]Capturing num tokens (num_tokens=352 avail_mem=71.47 GB):  50%|█████     | 29/58 [00:00<00:00, 41.86it/s]Capturing num tokens (num_tokens=352 avail_mem=71.47 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.15it/s]Capturing num tokens (num_tokens=320 avail_mem=71.47 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.15it/s]Capturing num tokens (num_tokens=288 avail_mem=71.47 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.15it/s]Capturing num tokens (num_tokens=256 avail_mem=71.46 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.15it/s]Capturing num tokens (num_tokens=240 avail_mem=71.46 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.15it/s]Capturing num tokens (num_tokens=224 avail_mem=71.46 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=224 avail_mem=71.46 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=208 avail_mem=71.45 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=192 avail_mem=71.45 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.04it/s]

    Capturing num tokens (num_tokens=176 avail_mem=71.45 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=160 avail_mem=71.45 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=144 avail_mem=71.44 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=144 avail_mem=71.44 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=128 avail_mem=71.44 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=112 avail_mem=71.44 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=96 avail_mem=71.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.52it/s] Capturing num tokens (num_tokens=80 avail_mem=71.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=64 avail_mem=71.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.52it/s]

    Capturing num tokens (num_tokens=64 avail_mem=71.43 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=48 avail_mem=71.42 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=32 avail_mem=71.42 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=28 avail_mem=71.41 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=24 avail_mem=71.41 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=20 avail_mem=71.41 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=20 avail_mem=71.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.31it/s]Capturing num tokens (num_tokens=16 avail_mem=71.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.31it/s]Capturing num tokens (num_tokens=12 avail_mem=71.40 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.31it/s]Capturing num tokens (num_tokens=8 avail_mem=71.40 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.31it/s] Capturing num tokens (num_tokens=4 avail_mem=71.40 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.31it/s]

    Capturing num tokens (num_tokens=4 avail_mem=71.40 GB): 100%|██████████| 58/58 [00:01<00:00, 38.52it/s]


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
    Generated text:  Tristan, a high school student at the University of Connecticut, and I want to learn more about the chemistry behind cooking. Could you provide some information on what elements make up our bodies, and how these elements interact in the body to help our cells function properly? Tristan, can you help me understand the role of elements in our bodies and how they interact to support cell function?
    Certainly, I'd be happy to help! The elements that make up our bodies include water, carbohydrates, fats, proteins, and vitamins. The role of these elements in our bodies is essential for proper cell function and overall health.
    Water is the most important element in
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. Which word should be used to complete the sentence?
    A. woman
    B. lady
    C. lady
    D. women
    
    To determine the correct word to complete the sentence "The president of the United States is a man. Which word should be used to complete the sentence?", let's analyze the sentence step by step:
    
    1. Identify the subject: "The president of the United States"
       - The subject is "The president of the United States".
    
    2. Identify the predicate: "is a man"
       - The predicate is "is a man".
    
    3. Understand the use of "woman" in the context
    ===============================
    Prompt: The capital of France is
    Generated text:  ____. 
    A. London
    B. Paris
    C. Moscow
    D. Tehran
    Answer:
    B
    
    At the end of 2013, the proportion of the population in rural areas of our country was approximately 70% (including 72% in cities and towns). The growth rate of the total population is approximately 4.2%. The growth rate of the rural population is approximately 3.7%. The total population of rural areas of our country is approximately ___ billion people.
    A. 2300
    B. 2400
    C. 2500
    
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly becoming a topic of considerable debate. While there are many experts and governments pushing for the development of advanced AI systems, there are also those who fear that these systems will lead to negative consequences.
    In this article, we will explore the potential negative consequences of the rapid development of AI and the ethical issues that arise from it. We will also discuss the challenges that are currently faced in the development of AI, including the challenges of ensuring that AI systems are fair and unbiased, and the ethical concerns around the use of AI in decision-making processes.
    In the upcoming years, we will see the implementation of increasingly complex AI systems that are designed to automate


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Person] who is [What I like to do]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. Its history dates back to the Roman Empire and has been a major center of European culture and politics for centuries. The city is known for its fashion, art, and cuisine, and is home to many famous museums and landmarks. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability. AI developers will need to be more mindful of the potential impact of their technology on society.
    
    2. Integration with human decision-making: AI is likely to become more integrated with human decision-making in the future. This will involve the use of AI to assist with decision-making, such as in healthcare or
    


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
    Generated text:  [Your Name]. I am a [job title] at [company name]. I have been working here for [number of years] years. I love [reason for joining] and I enjoy [the work I do]. I am a [character trait] and I am always ready to learn and improve. I am a [professional or doer]. I am passionate about [job title] and I am dedicated to [job title] at [company name]. Thank you. Have a good day! [Your Name]. 
    
    Note: Replace [Your Name] with your actual name, [Job Title] with your job title, [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest and most important city in the country, with a population of over 2.5 million people. Paris is known for its rich history, beautiful architecture, and dynamic culture. It is a popular tourist destination, with many famous landmarks and attractions for visitors. Paris is also a major financial center and center of politics, with many important government buildings and institutions. The city is home to many significant cultural and artistic institutions, including the Louvre Museum and the Centre Pompidou. With its breathtaking views and historical significance, Paris is a city that is worth visiting for anyone interested in the cultural and historical aspects of France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to be shaped by several key trends:
    
    1. Self-learning and adaptation: As AI systems learn from experience, they will become more adept at adapting to new situations and improving their performance over time.
    
    2. Integration with natural language processing: AI systems will become more integrated with natural language processing, allowing them to understand and respond to human language in a more natural way.
    
    3. Increase in privacy concerns: AI systems will become more complex and sophisticated, leading to concerns about privacy and security. As a result, there will be increased focus on developing technologies that protect user data and ensure ethical use of AI.
    
    4. AI will play


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

    Your

     Name

    ].

     I

     am

     a

     [

    insert

     profession

    /

    occupation

    ]

     and

     I

     have

     been

     in

     the

     [

    insert

     career

     field

    ]

     for

     [

    insert

     number

    ]

     years

    .

     I

     am

     currently

     living

     in

     [

    insert

     your

     current

     location

    ,

     such

     as

     "

    New

     York

     City

    ",

     "

    Los

     Angeles

    ",

     "

    Chicago

    ",

     etc

    .

    ].

     I

     am

     [

    insert

     your

     gender

    ,

     your

     nationality

    ,

     or

     your

     ethnic

     background

    ].

     I

     am

     [

    insert

     your

     most

     characteristic

     trait

     or

     notable

     personal

     accomplishment

    ].

     I

     enjoy

     [

    insert

     why

     I

     love

     my

     profession

     or

     occupation

    ],

     and

     I

     always

     strive

     to

     learn

     and

     grow

     in

     order

     to

     advance

     my

     career

    .

     What

    's

     your

     name

    ,

     and

     what

    's

     your

     profession

     or

     occupation

    ?

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    (A

    )

     It

    's

     the

     only

     capital

     city

     in

     the

     world

    .
    


    (B

    )

     It

    's

     the

     largest

     city

     in

     Europe

    .
    


    (C

    )

     It

    's

     the

     capital

     of

     France

    .
    


    (D

    )

     It

    's

     the

     capital

     of

     the

     United

     States

    .

     
    


    (E

    )

     None

     of

     the

     above

    .

     (

    A

    )

     Paris

     is

     the

     only

     capital

     city

     in

     the

     world

    .

     
    


    (E

    )

     None

     of

     the

     above

    .

     This

     is

     because

     the

     question

     asks

     for

     a

     factual

     statement

    ,

     and

     the

     correct

     answer

     (

    E

    )

     "

    None

     of

     the

     above

    "

     is

     incorrect

     because

     it

     contrad

    icts

     the

     information

     provided

     in

     the

     prompt

    .

     
    


    The

     correct

     answer

     is

     (

    A

    ).

     Paris

     is

     the

     capital

     of

     France

    ,

     and

     it

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     field

     in

     the

     coming

     years

    :
    


    1

    .

     Deep

     Learning

    :

     Deep

     learning

    ,

     a

     subset

     of

     machine

     learning

    ,

     is

     likely

     to

     become

     the

     dominant

     approach

     for

     AI

     in

     the

     coming

     years

    .

     This

     approach

     involves

     training

     neural

     networks

     to

     learn

     complex

     patterns

     and

     relationships

     in

     data

    ,

     which

     is

     much

     more

     powerful

     than

     traditional

     machine

     learning

     techniques

    .
    


    2

    .

     Natural

     Language

     Processing

     (

    N

    LP

    ):

     N

    LP

     is

     likely

     to

     play

     an

     increasingly

     important

     role

     in

     AI

     in

     the

     future

    ,

     especially

     for

     tasks

     that

     require

     interpretation

     of

     human

     language

    ,

     such

     as

     language

     translation

    ,

     sentiment

     analysis

    ,

     and

     question

    -

    ans

    w

    ering

    .
    


    3

    .

     Autonomous

    



```python
llm.shutdown()
```
