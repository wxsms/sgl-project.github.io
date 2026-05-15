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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.51it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.51it/s]


    2026-05-15 21:20:12,571 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 21:20:12] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.34it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.60it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.60it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.60it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.60it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.60it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.60it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.60it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.60it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.60it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.60it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.60it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.51it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.51it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.51it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.51it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.51it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.51it/s]

    Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.51it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.51it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.51it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.51it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.51it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.28it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.28it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.28it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.28it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.28it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.28it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.28it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.60 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.59 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.59 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=75.59 GB):   7%|▋         | 4/58 [00:00<00:03, 17.00it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.59 GB):   7%|▋         | 4/58 [00:00<00:03, 17.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.58 GB):   7%|▋         | 4/58 [00:00<00:03, 17.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.58 GB):  10%|█         | 6/58 [00:00<00:02, 17.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.57 GB):  10%|█         | 6/58 [00:00<00:02, 17.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.57 GB):  10%|█         | 6/58 [00:00<00:02, 17.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.57 GB):  10%|█         | 6/58 [00:00<00:02, 17.51it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=75.57 GB):  10%|█         | 6/58 [00:00<00:02, 17.51it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.57 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.56 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.56 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.56 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.56 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.55 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.55 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.55 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.54 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.54 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.96it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=75.54 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.54 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.54 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=75.52 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.35it/s]Capturing num tokens (num_tokens=960 avail_mem=75.53 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.35it/s] Capturing num tokens (num_tokens=896 avail_mem=75.53 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.35it/s]Capturing num tokens (num_tokens=832 avail_mem=75.52 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.35it/s]Capturing num tokens (num_tokens=768 avail_mem=75.52 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.35it/s]Capturing num tokens (num_tokens=768 avail_mem=75.52 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=704 avail_mem=75.52 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=640 avail_mem=75.51 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=576 avail_mem=75.51 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.99it/s]

    Capturing num tokens (num_tokens=512 avail_mem=75.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=480 avail_mem=75.35 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.99it/s]Capturing num tokens (num_tokens=480 avail_mem=75.35 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.02it/s]Capturing num tokens (num_tokens=448 avail_mem=75.35 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.02it/s]Capturing num tokens (num_tokens=416 avail_mem=75.35 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.02it/s]Capturing num tokens (num_tokens=384 avail_mem=75.35 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.02it/s]Capturing num tokens (num_tokens=352 avail_mem=75.34 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.02it/s]Capturing num tokens (num_tokens=320 avail_mem=75.34 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.02it/s]Capturing num tokens (num_tokens=320 avail_mem=75.34 GB):  60%|██████    | 35/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=288 avail_mem=75.33 GB):  60%|██████    | 35/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=256 avail_mem=75.33 GB):  60%|██████    | 35/58 [00:01<00:00, 43.78it/s]Capturing num tokens (num_tokens=240 avail_mem=75.33 GB):  60%|██████    | 35/58 [00:01<00:00, 43.78it/s]

    Capturing num tokens (num_tokens=224 avail_mem=75.32 GB):  60%|██████    | 35/58 [00:01<00:00, 43.78it/s]Capturing num tokens (num_tokens=208 avail_mem=75.05 GB):  60%|██████    | 35/58 [00:01<00:00, 43.78it/s]Capturing num tokens (num_tokens=208 avail_mem=75.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=192 avail_mem=75.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=176 avail_mem=75.04 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=160 avail_mem=74.34 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=144 avail_mem=74.33 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=128 avail_mem=74.33 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=128 avail_mem=74.33 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.47it/s]Capturing num tokens (num_tokens=112 avail_mem=74.33 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.47it/s]Capturing num tokens (num_tokens=96 avail_mem=74.33 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.47it/s] Capturing num tokens (num_tokens=80 avail_mem=74.32 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.47it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.32 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.47it/s]Capturing num tokens (num_tokens=48 avail_mem=74.32 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.47it/s]Capturing num tokens (num_tokens=48 avail_mem=74.32 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=32 avail_mem=74.31 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=28 avail_mem=74.31 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=24 avail_mem=74.31 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=20 avail_mem=74.30 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=16 avail_mem=74.30 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=16 avail_mem=74.30 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=12 avail_mem=74.30 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=8 avail_mem=74.29 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.79it/s] Capturing num tokens (num_tokens=4 avail_mem=74.29 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.79it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.29 GB): 100%|██████████| 58/58 [00:01<00:00, 38.90it/s]


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
    Generated text:  Anna and I am a teacher from Australia. I love to share my knowledge and help others. I have worked in Australia and even in other countries, and now I am a teacher. I have taught many languages and languages have been my life. Now I want to share my stories and knowledge. I am an English teacher and I want to help others by teaching them English. I hope to improve the English language of the students. My parents, who are also teachers, took me to America to help me. They are very strict and have no patience. They say that they want to help me. I do not like that. I think
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ceremonial and political post. He or she is appointed by the President of the United States and by the U.S. Congress, in accordance with the U.S. Constitution. The U. S. Constitution has been amended three times, most recently in 1964. The president serves for a term of 4 years.
    The president represents the United States on the international stage and is a member of the U. S. Congress. They also serve on the Peace Corps. They are a member of the U. S. Navy.
    In the 2018 and 2019 election, the president is the
    ===============================
    Prompt: The capital of France is
    Generated text:  a beautiful, enchanting city that is home to many old and historic sites. For example, the Palace of Versailles is the former residence of the French kings, and the Champs-Elysées is the "greenest" avenue in the world. While you might not be able to visit the Palace of Versailles, you can easily visit the other important old and historic sites. Whether you're in Paris, Lyon, or another city in France, these sites will surely captivate your imagination and leave you with unforgettable memories.
    I am interested in visiting the Palace of Versailles. Can you provide me with a list of important sites to
    ===============================
    Prompt: The future of AI is
    Generated text:  in hand
    
    What are the four key characteristics that define AI?
    
    There are several approaches that can be taken to define the characteristics that define AI. For example, the commonly used definition of AI is as a cognitive system that can reason, learn, and interact with the world by mimicking human-like behavior. The definitions of AI are also divided into three main categories: hard, soft, and pragmatic. Each of these categories focuses on different aspects of AI.
    
    The following four characteristics define AI:
    
    1. Human-like behavior: In order for an AI system to function as a good AI, it must be able to mimic human-like behavior. In


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and art galleries. Paris is a popular tourist destination, attracting millions of visitors each year. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on modern French culture. Paris is also home to many famous French artists, writers, and musicians. The city is known for its cuisine, including its famous Parisian cuisine, and its fashion industry. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and productive machines that can perform tasks that were previously done by humans.
    
    2. AI ethics and privacy concerns: As AI technology becomes more advanced, there will be increasing concerns about its ethical implications and potential privacy violations. This could lead to the development of new
    


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
    Generated text:  [First name] and I'm a [Last name] [First name] from [location]. I'm an [age], [nationality], [occupation] and I'm a [role/character]. I'm currently [role] and I'm passionate about [occupation] as [reason for passion]. I love [exciting fact about my personality] and I'm always looking for ways to [challenge or improve myself]. I have [number of friends], [number of hobbies], and [number of interests]. My biggest challenge is [challenge or problem I'm facing], and my greatest success is [reason for success]. I believe
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks, rich history, and French cuisine. It is also a city with a diverse population and a rich cultural heritage, including the Louvre Museum and the Notre-Dame Cathedral. It is the largest and most populous city in France, and has a population of over 2 million people. The city is a hub of innovation and technology, and is known as the “City of Light” due to its lights and neon signs. Paris is also known for its romantic atmosphere, and is home to numerous iconic locations and events, including the Eiffel Tower and the Louvre Museum. The French capital is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but several trends are likely to shape it in the coming years:
    
    1. Increased automation and robotics: As AI becomes more advanced, it will become even more capable of performing repetitive tasks, reducing the need for human labor. This will create new jobs, but also lead to increased automation, which could lead to increased efficiency and productivity.
    
    2. AI ethics and governance: As AI systems become more integrated into our daily lives, there will be increasing pressure to address the ethical implications of AI, such as bias, privacy, and transparency. This will drive the development of new ethical standards and regulations.
    
    3. AI in healthcare: AI is


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

    insert

     character

    's

     name

    ],

     and

     I

    'm

     a

     [

    insert

     main

     character

    's

     profession

     or

     occupation

    ]

     who

     has

     been

     in

     the

     game

     for

     [

    insert

     number

     of

     years

    ].

     What

     can

     you

     tell

     me

     about

     yourself

     and

     your

     experience

     with

     the

     game

     industry

    ?

     As

     a

     [

    insert

     main

     character

    's

     profession

     or

     occupation

    ],

     I

    've

     been

     in

     the

     industry

     for

     [

    insert

     number

     of

     years

    ]

     years

    ,

     and

     I

     have

     a

     passion

     for

     [

    insert

     something

     that

     you

     could

     describe

     about

     yourself

    ,

     such

     as

     your

     interests

    ,

     hobbies

    ,

     or

     strengths

    ].

     I

    'm

     always

     eager

     to

     learn

     new

     things

     and

     stay

     up

    -to

    -date

     with

     the

     latest

     trends

     and

     techniques

     in

     the

     game

     industry

    .

     How

     do

     you

     feel

     about

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     summarizes

     the

     key

     fact

     about

     France

    's

     capital

     city

    ,

     stating

     that

     Paris

     is

     its

     capital

    .

     It

     does

     not

     provide

     additional

     context

     about

     what

     Paris

     is

     known

     for

     or

     any

     other

     specific

     details

     about

     it

    .


    Your

     response

     should

     be

     one

     or

     two

     complete

     sentences

     long

     and

     clearly

     express

     the

     fact

     being

     described

    .

     The

     key

     point

     should

     be

     the

     identity

     of

     Paris

    's

     capital

     city

    .

     The

     statement

     should

     be

     concise

     enough

     to

     convey

     the

     meaning

     in

     a

     single

     sentence

    .

     Please

     keep

     in

     mind

     that

     the

     answer

     should

     not

     exceed

     three

     words

    .

     


    Example

    :

     "

    The

     capital

     of

     France

     is

     Paris

    ."

     This

     answer

     meets

     the

     criteria

     by

     identifying

     Paris

     as

     its

     capital

     and

     not

     including

     any

     additional

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     difficult

     to

     predict

    .

     However

    ,

     there

     are

     a

     few

     potential

     trends

     that

     could

     influence

     the

     development

     of

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     There

     is

     growing

     awareness

     of

     the

     need

     to

     ensure

     that

     AI

     is

     developed

     and

     used

     eth

    ically

    .

     This

     will

     likely

     lead

     to

     greater

     emphasis

     on

     ethical

     considerations

     such

     as

     bias

    ,

     fairness

    ,

     and

     transparency

    .

     As

     a

     result

    ,

     there

     may

     be

     more

     focus

     on

     creating

     AI

     systems

     that

     are

     designed

     to

     be

     more

     socially

     responsible

    .
    


    2

    .

     Faster

     progress

     in

     natural

     language

     processing

    :

     Natural

     language

     processing

     is

     a

     key

     area

     of

     focus

     for

     AI

     researchers

    ,

     as

     it

     is

     central

     to

     many

     applications

     of

     AI

    .

     Advances

     in

    



```python
llm.shutdown()
```
