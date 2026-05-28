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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.02it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.01it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.87it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.87it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 14.18it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 14.18it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 14.18it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 14.18it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 14.18it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 14.18it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 14.18it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 14.18it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 14.18it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:02, 14.18it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 21.26it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 21.26it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 21.26it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 21.26it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 21.26it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 21.26it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 21.26it/s]

    Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 29.27it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 29.27it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 29.27it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 29.27it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 29.27it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 29.27it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 29.27it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 29.27it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 29.27it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 29.27it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 37.99it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 37.99it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 37.99it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 37.99it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.21it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.54 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.51 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.51 GB):   3%|▎         | 2/58 [00:00<00:03, 17.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.51 GB):   3%|▎         | 2/58 [00:00<00:03, 17.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.51 GB):   3%|▎         | 2/58 [00:00<00:03, 17.15it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.51 GB):   3%|▎         | 2/58 [00:00<00:03, 17.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.51 GB):   9%|▊         | 5/58 [00:00<00:02, 20.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.50 GB):   9%|▊         | 5/58 [00:00<00:02, 20.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.49 GB):   9%|▊         | 5/58 [00:00<00:02, 20.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.49 GB):   9%|▊         | 5/58 [00:00<00:02, 20.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.49 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.49 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.48 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.45it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=58.48 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.48 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.48 GB):  21%|██        | 12/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.47 GB):  21%|██        | 12/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.47 GB):  21%|██        | 12/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.47 GB):  21%|██        | 12/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.46 GB):  21%|██        | 12/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.46 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.46 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.46it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=58.46 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.45 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.46it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.45 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.46it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.45 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.43 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.89it/s]Capturing num tokens (num_tokens=960 avail_mem=58.45 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.89it/s] Capturing num tokens (num_tokens=896 avail_mem=58.44 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.89it/s]Capturing num tokens (num_tokens=832 avail_mem=58.44 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.89it/s]Capturing num tokens (num_tokens=768 avail_mem=58.44 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.89it/s]

    Capturing num tokens (num_tokens=768 avail_mem=58.44 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.11it/s]Capturing num tokens (num_tokens=704 avail_mem=58.43 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.11it/s]Capturing num tokens (num_tokens=640 avail_mem=58.43 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.11it/s]Capturing num tokens (num_tokens=576 avail_mem=58.43 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.11it/s]Capturing num tokens (num_tokens=512 avail_mem=58.41 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.11it/s]Capturing num tokens (num_tokens=480 avail_mem=58.43 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.11it/s]Capturing num tokens (num_tokens=480 avail_mem=58.43 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.38it/s]Capturing num tokens (num_tokens=448 avail_mem=58.43 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.38it/s]Capturing num tokens (num_tokens=416 avail_mem=58.43 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.38it/s]Capturing num tokens (num_tokens=384 avail_mem=58.42 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.38it/s]Capturing num tokens (num_tokens=352 avail_mem=58.42 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.38it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.42 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=320 avail_mem=58.41 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=288 avail_mem=58.41 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=256 avail_mem=58.41 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=240 avail_mem=58.40 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=224 avail_mem=58.13 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=224 avail_mem=58.13 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=208 avail_mem=58.12 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=192 avail_mem=58.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=176 avail_mem=57.42 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=160 avail_mem=57.41 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.13it/s]

    Capturing num tokens (num_tokens=160 avail_mem=57.41 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=144 avail_mem=57.41 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=128 avail_mem=57.41 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=112 avail_mem=57.41 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=96 avail_mem=57.40 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.57it/s] Capturing num tokens (num_tokens=80 avail_mem=57.40 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=80 avail_mem=57.40 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=64 avail_mem=57.40 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=48 avail_mem=57.39 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=32 avail_mem=57.39 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=28 avail_mem=57.38 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.37it/s]

    Capturing num tokens (num_tokens=24 avail_mem=57.38 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.37it/s]Capturing num tokens (num_tokens=24 avail_mem=57.38 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=20 avail_mem=57.38 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=16 avail_mem=57.38 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=12 avail_mem=57.37 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=8 avail_mem=57.37 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.91it/s] Capturing num tokens (num_tokens=4 avail_mem=57.37 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=4 avail_mem=57.37 GB): 100%|██████████| 58/58 [00:01<00:00, 40.59it/s]Capturing num tokens (num_tokens=4 avail_mem=57.37 GB): 100%|██████████| 58/58 [00:01<00:00, 34.91it/s]


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
    Generated text:  Luis and I'm an AI developed by Alibaba Cloud. I can do tasks like answering your questions and assisting you with your projects in various ways. I'm here to help you whenever you need it. 
    
    As an AI, I'm programmed to help you with a wide range of topics, including technology, business, education, health, and more. I'm committed to providing you with the best possible assistance.
    
    How can I use you to assist me with my projects or tasks? Can you give me some examples of how I can use your services?
    
    Certainly! To use the services of an AI like Luis, I can assist you in answering
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political head of a country, and he/she has significant power, and a great many elections are held. In order to choose the president, which of the following groups of people would be best for you to choose? Choose the best option to answer the question. Options are:
    A) Your parents
    B) A member of your church
    C) Your friends
    D) A group of friends who have never met
    E) Your neighbor
    F) A political party leader who has never been elected
    G) A government official who has never been elected
    H) A political party leader who has been elected
    To determine which group
    ===============================
    Prompt: The capital of France is
    Generated text: : [Answer]
    The capital of France is Paris. Paris is the capital city of France and is located in the south of the country. It is one of the most famous cities in the world and is known for its historic architecture, world-famous museums, and annual celebrations. Paris has a rich history dating back over 1,000 years, and it continues to be a major cultural and tourism hub for France. The city is also home to important universities, including the University of Paris-Sorbonne, the École des Ponts et Chaussées, and the Sorbonne. As of 202
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly changing. As companies increasingly invest in AI, it is becoming a key differentiator in the competitive landscape. However, the ever-evolving landscape means that a team of AI engineers must adapt to stay ahead of the curve. This requires a deep understanding of AI, its capabilities, and limitations. To do this, it is important to understand the different types of AI, their strengths, and limitations, and how to apply them in different contexts.
    
    Understanding the different types of AI and their strengths and limitations is crucial for AI engineers to create a successful AI project. Here are the different types of AI and how they can be used:
    
    1.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for ways to improve my skills and knowledge, and I'm always eager to learn new things. I'm a [job title] at [company name], and I'm always looking for ways to improve my skills and knowledge, and I'm always eager to learn new things. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and being the birthplace of the French Revolution. Paris is a popular tourist destination, known for its fashion, art, and cuisine. It is also home to many famous landmarks and museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. The city is also known for its diverse population, with many languages spoken and a rich cultural heritage. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency, productivity, and cost savings for businesses and individuals.
    
    2. AI-powered healthcare: AI will play a significant role in healthcare, with the development of more accurate and personalized medical diagnoses and treatments. AI-powered medical devices will also be used to improve patient care and reduce costs.
    
    
    


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
    Generated text:  [Name], and I'm a [Occupation] with a passion for [What I love to do]. I'm a [Type of Person] who is always looking for new adventures and experiences. I'm always eager to learn and grow, and I enjoy sharing my knowledge with others. I believe in the power of [What I believe in], and I'm passionate about using my skills and knowledge to make a positive impact in the world. I'm confident in my ability to contribute to the community and help others achieve their goals. I'm excited to meet new people and expand my horizons. [Name], what is your occupation and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, unique architecture, and lively nightlife.
    
    France's capital city, Paris, is renowned for its iconic landmarks, diverse cultural scene, and vibrant nightlife. Its rich history, unique architecture, and lively nightlife make it a must-visit destination for those interested in France's cultural and historical attractions. Paris is a bustling metropolis with an active nightlife, including clubs, bars, and nightclubs, and a rich cultural scene with museums, art galleries, and theaters. The city's cuisine, art, and music are also highly regarded, making it a destination for music lovers, art enthusiasts, and foodies alike
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting and likely to involve a wide range of technological developments and advancements. Here are some potential future trends in AI:
    
    1. Increased automation and AI-based automation: As AI becomes more advanced, it is likely to become more prevalent in manufacturing, customer service, and other industries where it can perform tasks more efficiently and accurately than human workers.
    
    2. Biometric and facial recognition technology: As AI continues to improve, we can expect to see more sophisticated biometric and facial recognition technologies that can be integrated into everyday life.
    
    3. Augmented and virtual reality: AI-powered augmented and virtual reality will continue to evolve, bringing new forms of entertainment


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

    'm

     a

    /an

     [

    Your

     Profession

    ]

     with

     [

    Your

     Education

    ]

     and

     a

     [

    Your

     H

    obbies

    ]

     background

    .

     In

     my

     spare

     time

    ,

     I

     enjoy

     [

    Your

     Favorite

     Activity

    ].

     What

     brings

     you

     to

     this

     profession

    ?

     Let

     me

     know

    !

     

    🤖

    🔍

    ✨

    
    


    Feel

     free

     to

     vary

     the

     details

     to

     make

     your

     introduction

     more

     unique

     and

     engaging

    !

     

    🙋

    ‍

    ♂

    ️

    💼

    ✨

    
    


    This

     is

     a

     book

     review

    .

     Please

     provide

     a

     title

     and

     cover

     image

    .

     

    📚

    📚

    
    


    Title

    :

     "

    The

     Great

     Escape

    :

     A

     Journey

     Through

     Time

    's

     Dark

     H

    oles

    "
    


    Cover

     Image

    :

     A

     striking

     image

     of

     a

     massive

     black

     hole

    ,

     puls

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     historical

     landmarks

    ,

     vibrant

     arts

     scene

    ,

     and

     rich

     cultural

     heritage

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     a

     population

     of

     over

     

    2

    .

    4

     million

     people

    ,

     and

     is

     a

     major

     city

     of

     tourism

     and

     commerce

    .

     The

     city

     is

     home

     to

     many

     famous

     historical

     sites

    ,

     such

     as

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    ,

     and

     is

     an

     important

     center

     of

     government

     and

     politics

     in

     France

    .

     It

     is

     also

     known

     for

     its

     excellent

     cuisine

    ,

     including

     its

     famous

     cro

    iss

    ants

     and

     savory

     dishes

    ,

     and

     for

     its

     annual

     Festival

     de

     la

     Mus

    ique

     which

     features

     a

     variety

     of

     musical

     performances

     throughout

     the

     year

    .

     The

     city

     is

     also

     known

     for

     its

     beaches

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     with

     many

     possible

     trends

     to

     watch

     out

     for

    .

     Some

     possible

     trends

     include

    :
    


    1

    .

     Increasing

    ly

     autonomous

     machines

    :

     AI

     is

     advancing

     rapidly

    ,

     and

     we

     may

     see

     more

     machines

     that

     are

     capable

     of

     making

     decisions

     on

     their

     own

    ,

     without

     human

     intervention

    .
    


    2

    .

     Natural

     language

     processing

    :

     As

     AI

     becomes

     more

     advanced

    ,

     we

     may

     see

     more

     natural

     language

     processing

     capabilities

    ,

     allowing

     machines

     to

     understand

     and

     respond

     to

     human

     language

     in

     the

     same

     way

     humans

     do

    .
    


    3

    .

     Enhanced

     AI

    :

     With

     the

     help

     of

     machine

     learning

     and

     deep

     learning

    ,

     AI

     may

     be

     able

     to

     improve

     its

     performance

     over

     time

    ,

     making

     it

     more

     accurate

     and

     efficient

     in

     its

     tasks

    .
    


    4

    .

     Human

    -

    robot

     interaction

    :

    



```python
llm.shutdown()
```
