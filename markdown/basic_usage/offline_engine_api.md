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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.39it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.38it/s]


    2026-05-16 11:52:47,793 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 11:52:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.41it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 22.99it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 22.99it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 22.99it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 22.99it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 22.99it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 22.99it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 22.99it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 22.99it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 22.99it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.99it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.99it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.75 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.74 GB):   9%|▊         | 5/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.74 GB):   9%|▊         | 5/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.73 GB):   9%|▊         | 5/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.72 GB):   9%|▊         | 5/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.72 GB):   9%|▊         | 5/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.71 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.71 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.71 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.61it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=73.71 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.69 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.69 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.67 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]Capturing num tokens (num_tokens=960 avail_mem=73.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s] Capturing num tokens (num_tokens=896 avail_mem=73.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]

    Capturing num tokens (num_tokens=832 avail_mem=73.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]Capturing num tokens (num_tokens=768 avail_mem=73.67 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]Capturing num tokens (num_tokens=768 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=704 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=640 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=576 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=512 avail_mem=73.65 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=480 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=448 avail_mem=73.66 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=448 avail_mem=73.66 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.68it/s]Capturing num tokens (num_tokens=416 avail_mem=73.66 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.68it/s]Capturing num tokens (num_tokens=384 avail_mem=73.66 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.68it/s]Capturing num tokens (num_tokens=352 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.68it/s]

    Capturing num tokens (num_tokens=320 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.68it/s]Capturing num tokens (num_tokens=288 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.68it/s]Capturing num tokens (num_tokens=256 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.68it/s]Capturing num tokens (num_tokens=256 avail_mem=73.65 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.46it/s]Capturing num tokens (num_tokens=240 avail_mem=73.64 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.46it/s]Capturing num tokens (num_tokens=224 avail_mem=73.64 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.46it/s]Capturing num tokens (num_tokens=208 avail_mem=73.63 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.46it/s]Capturing num tokens (num_tokens=192 avail_mem=73.63 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.46it/s]Capturing num tokens (num_tokens=176 avail_mem=73.63 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.46it/s]Capturing num tokens (num_tokens=160 avail_mem=73.63 GB):  64%|██████▍   | 37/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=160 avail_mem=73.63 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=144 avail_mem=73.62 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=128 avail_mem=73.62 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.72it/s]

    Capturing num tokens (num_tokens=112 avail_mem=73.62 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=96 avail_mem=73.62 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.72it/s] Capturing num tokens (num_tokens=80 avail_mem=73.61 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=80 avail_mem=73.61 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.06it/s]Capturing num tokens (num_tokens=64 avail_mem=73.61 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.06it/s]Capturing num tokens (num_tokens=48 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.06it/s]Capturing num tokens (num_tokens=32 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.06it/s]Capturing num tokens (num_tokens=28 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.06it/s]Capturing num tokens (num_tokens=24 avail_mem=73.59 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.06it/s]Capturing num tokens (num_tokens=24 avail_mem=73.59 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=16 avail_mem=73.59 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]

    Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s] Capturing num tokens (num_tokens=4 avail_mem=73.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:01<00:00, 38.98it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:01<00:00, 40.54it/s]


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
    Generated text:  Zelma Yevgenyevna. My birth name is Zelma Evgenyevna, and I am a famous Russian actress. I am a model, a composer, and a ballet dancer. I was born on the 13th of February 1972. And I am from the city of Yaroslavl in the Russian Federation. I was the second daughter of the famous actor Vasiliy Yevgenyev, and the daughter of the famous ballet dancer Princess Maria of Westermarck. My parents were very kind, and they always supported me in my life. And I
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to buy a new airplane or to keep his old one. Currently, he has 700,000 gallons of fuel and plans to use 10% of it each week for maintenance. How many gallons of fuel will the president have left after 6 weeks?
    To determine how many gallons of fuel the president will have left after 6 weeks, we need to follow these steps:
    
    1. Calculate the amount of fuel used each week for maintenance.
    2. Determine the total amount of fuel used over 6 weeks.
    3. Subtract the total amount of fuel used from the initial amount of fuel.
    
    
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Paris
    C. Lyon
    D. Marseille
    Answer: A
    
    In China, what is the capital?
    A. Hangzhou
    B. Hangzhou
    C. Hangzhou
    D. Hangzhou
    Answer: A
    
    What is the capital of France?
    A. Paris
    B. Paris
    C. Lyon
    D. Marseille
    Answer: A
    
    What is the capital of France?
    A. Paris
    B. Paris
    C. Lyon
    D. Marseille
    Answer: A
    
    What is the capital of France?
    A. Paris
    B. Paris
    C. Lyon
    D.
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting, with many uses being created. The use of AI in the workplace has been a major driver of change in the tech industry. The rise of machine learning algorithms has helped companies automate jobs, increase efficiency, and reduce the time and cost of production. With AI in the workplace, it has also led to the creation of new roles, including those in artificial intelligence and data science.
    AI in the workplace is a rapidly growing field, and there are a variety of companies and organizations working on AI projects. Some of the most popular AI projects in the workplace include:
    1. Robotics: Robotics is the field of study that involves the design,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and [job title]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, art, and cuisine, and is a UNESCO World Heritage site. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. Its population is approximately 2
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the potential trends that could be expected in the future of AI:
    
    1. Increased automation: As AI continues to improve, it is likely to become more and more integrated into our daily lives. This could lead to a greater automation of tasks, such as manufacturing, transportation, and customer service, which could result in increased efficiency and productivity.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be a greater need for privacy and security. This could lead
    


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
    Generated text:  [Name], and I'm a [occupation] who has been following [or researching] for the past [number] years. I've written a few novels, but my most recent was [the title]. And I've published [number] books, with [number] of them being self-published. I've also won a few awards, including [award name], which was [award number]. I've been a [profession] for [number] years, and my hobbies include [list of hobbies]. I'm also an avid [activity] and [gaming/reading] addict. I enjoy keeping myself busy and have a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its diverse culture, cuisine, and music, and has been a major hub for politics, culture, and business for centuries. Paris is a sprawling metropolis with a rich history and vibrant nightlife, attracting millions of tourists each year. Its cuisine includes dishes like croissants and boudin, and its music includes popular genres like rock, jazz, and pop. The city's architecture is a testament to its rich history, including the Baroque style and the Gothic Revival. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and there are many potential trends that are expected to shape the development of this technology. Some of the most significant trends in AI include:
    
    1. Increased Integration with Human-Centered Design: AI systems are becoming more closely integrated with human-centered design principles, such as user experience, usability, and accessibility. As a result, AI systems are becoming more human-like, and more human-powered.
    
    2. AI to Replace Human Workers: As automation and AI become more integrated into the economy, there is a growing concern that AI could replace human workers. While some see this as a positive outcome, others are concerned about the potential for job


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

    ].

     I

    'm

     a

     [

    Role

    ]

     in

     the

     [

    Company

    ]

     who

     works

     [

    Respons

    ibility

    ].

     I

     have

     [

    Number

     of

     years

    ]

     years

     of

     experience

     in

     this

     role

    .

     I

     enjoy

     [

    What

     I

     like

     to

     do

    ].

     I

     also

     enjoy

     [

    Other

     hobbies

     or

     interests

    ].

     In

     my

     spare

     time

    ,

     I

     like

     [

    Anything

     I

     enjoy

    ].

     And

     I

    'm

     always

     looking

     for

     [

    What

     I

     am

     looking

     for

    ].

     I

    'm

     always

     ready

     to

     learn

     and

     grow

    .

     I

    'm

     a

     [

    Type

     of

     Person

    ]

     who

     values

     [

    What

     they

     believe

     in

    ].

     I

     strive

     to

     [

    What

     I

     try

     to

     achieve

    ].

     I

    'm

     proud

     to

     be

     [

    What

     I

    'm

     proud

     of

    ].

     I

     believe

     that

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     central

     region

     of

     the

     country

    .

     
    


    This

     statement

     is

     accurate

     and

     concise

    .

     Paris

     is

     the

     capital

     of

     France

    ,

     a

     country

     known

     for

     its

     rich

     history

    ,

     culture

    ,

     and

     beautiful

     architecture

    .

     The

     city

    ,

     situated

     on

     the

     Se

    ine

     River

    ,

     is

     home

     to

     many

     famous

     landmarks

     such

     as

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

     also

     plays

     a

     significant

     role

     in

     France

    's

     economy

    ,

     with

     a

     diverse

     population

     that

     includes

     immigrants

     from

     all

     over

     the

     world

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     restaurants

     that

     celebrate

     its

     unique

     cultural

     heritage

    .

     The

     city

     has

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     there

     are

     many

     possible

     trends

     that

     could

     shape

     the

     direction

     of

     development

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Integration

     with

     Human

     Intelligence

    :

     AI

     is

     becoming

     more

     integrated

     with

     human

     intelligence

     in

     a

     variety

     of

     ways

    .

     This

     could

     include

     AI

     being

     integrated

     into

     the

     human

     brain

     or

     body

    ,

     allowing

     for

     enhanced

     cognitive

     functions

     or

     even

     human

    -like

     intelligence

    .

     It

     could

     also

     include

     AI

     being

     integrated

     into

     human

     decision

    -making

     processes

     to

     improve

     outcomes

     and

     reduce

     human

     error

    .
    


    2

    .

     Autonomous

     and

     Semi

    -A

    ut

    onomous

     Vehicles

    :

     As

     autonomous

     and

     semi

    -aut

    onomous

     vehicles

     become

     more

     common

    ,

     AI

     will

     play

     an

     increasingly

     important

     role

     in

     transportation

     systems

    .

     This

     could

     lead

     to

     increased

    



```python
llm.shutdown()
```
