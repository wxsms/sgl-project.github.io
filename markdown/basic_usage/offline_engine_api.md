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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.15it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.15it/s]


    2026-05-05 22:11:40,431 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 22:11:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.70it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.51it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.51it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.51it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.51it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.51it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.51it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.51it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.51it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.51it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.51it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 26.24it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 26.24it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 26.24it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 26.24it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 26.24it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 26.24it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 26.24it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 26.24it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 26.24it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:04<00:00, 26.24it/s]

    Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:04<00:00, 26.24it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 35.51it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 35.51it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 35.51it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 35.51it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 35.51it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 35.51it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 35.51it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.91it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.91it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.91it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.91it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.91it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.91it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.59it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.59it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.59it/s]

    Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.00it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.00it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.00it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.00it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.00it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.00it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.01it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.01it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.01it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.01it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.01it/s]

    Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.01it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.40it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.40it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.40it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.40it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 42.69it/s]


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
    Generated text:  Mihaela and I am a PhD student in the Computer Science department at the University of Texas at Austin. My research focuses on developing new algorithms and computational models that make it possible to analyze, model and optimize many real-world systems, from ecosystems to supply chains and from audio to visual speech. My research uses both theoretical tools and computational approaches to find answers to real-world problems. I am broadly interested in how to find meaningful solutions to complex problems, and I am particularly interested in how to find solutions that take into account the effect of an external environment (e.g. temperature, noise, light) and the impact of different systems interactions.
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  a representative of the ____.
    A. Senate
    B. House of Representatives
    C. Supreme Court
    D. Cabinet
    
    To determine the correct answer, let's first understand the structure of the United States government. The United States is a federal republic, where the president serves as the head of state and the commander-in-chief of the armed forces. Therefore, the president is a representative of the Senate.
    
    Let's analyze each option:
    
    A. Senate: The Senate is the lower house of the United States Congress and is responsible for approving laws and making decisions. While the Senate plays a role in important legislative matters, it is not the primary
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It was established in 905 when Charles Martel, the first known French king, began to control the duchy of Normandy. The city was first called Paris by the Romans and later it became the capital of France. Paris is famous for its many monuments such as the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and the Arc de Triomphe. It has been a city for centuries and its history has been shaped by events like the French Revolution and World Wars. It is a beautiful city with a rich culture and history. Located in the center of the country, it is famous
    ===============================
    Prompt: The future of AI is
    Generated text:  changing rapidly and the potential applications of these machines are immense. Here is an overview of how AI is transforming various sectors and its potential impact on society. It can be further divided into four categories: healthcare, education, transportation, and manufacturing.
    
    ### 1. Healthcare
    
    In healthcare, AI is revolutionizing the field by providing more accurate diagnoses, personalized treatment plans, and predictive analytics. For instance, AI can analyze medical images and biometric data to identify abnormalities and determine the most likely diagnoses. This can help medical professionals make more informed decisions and provide better patient care.
    
    ### 2. Education
    
    AI is being used in the education sector


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am a [job title] at [company name] and I have been with the company for [number of years] years. I am a [job title] at [company name] and I have been with the company for [number of years] years. I am a [job title] at [company name] and I have been with the company for [number of years] years. I am a [job title] at [company name] and I have been with the company for [number of years] years. I am a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Museum. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is known for its cuisine, fashion, and art scene. It is also home to the French Parliament and the French National Museum. Paris is a vibrant and diverse city with a rich cultural heritage and is a popular tourist destination. The city is known for its iconic landmarks, museums, and cuisine. It is also home to the French Parliament and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated forms of AI, such as those that can learn and adapt to new situations based on human feedback.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be a need to address privacy and security concerns. This could lead to the development of new technologies that are more
    


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
    Generated text:  [Name] and I'm a [Title] who specialize in [Role]. I'm [Age] years old, [Occupation], and I bring a unique blend of [Strengths and Weaknesses]. If you're a fan of my work, my passions include [Work] and [Importance]. I thrive on collaborating with dynamic personalities and navigating the intricacies of complex projects with precision and passion. My ultimate goal is to create an experience that resonates with [Target Audience] and expands their understanding of [Subject matter]. I'm always eager to learn and improve, so I encourage you to visit my website to explore more
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    You are an AI assistant that helps you understand the answers. Don't be out of place. Don't hesitate in asking if you're not sure about the answer!
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  complex, and the trend is likely to continue to evolve and change over the coming years. Some possible trends that could be expected include:
    
    1. Increased automation: AI is already transforming industries, from manufacturing to finance to healthcare. The trend is expected to continue, with more automation being implemented in different areas.
    
    2. Artificial intelligence becoming more conscious: While AI is currently programmed to perform tasks, it is also becoming more conscious. This means that AI systems can learn and adapt based on their environment, rather than simply following instructions.
    
    3. Enhanced privacy and security: As AI systems become more sophisticated, there is a risk that they could be used


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

     [

    role

    ]

    !

     I

     love

     [

    main

     character

    's

     job

     or

     hobby

    ]

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     [

    describe

     an

     opportunity

     or

     goal

    ].

     I

    'm

     excited

     to

     get

     to

     know

     you

     and

     help

     you

     with

     your

     [

    mention

     a

     specific

     challenge

     or

     task

    ].

     What

     is

     your

     name

     and

     what

     kind

     of

     work

     do

     you

     do

    ?

     [

    Personal

     information

    ]

     [

    Ex

    periences

    ]

     [

    Achie

    vements

    ]

     [

    Op

    port

    unities

    ]

     [

    Ch

    allenges

    ]

     [

    Skills

    ]

     [

    Personal

     Traits

    ]

     As

     a

     [

    main

     character

    's

     profession

     or

     hobby

    ],

     I

    'm

     known

     for

     my

     [

    describe

     the

     activity

     or

     skill

    ].

     I

    'm

     always

     looking

     for

     new

     ways

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    la

     Par

    o

    isse

     Rouge

    "

     or

     simply

     "

    Paris

    ,"

     and

     is

     the

     largest

     city

     in

     both

     the

     country

     and

     the

     world

    .


    Paris

     is

     the

     capital

     of

     France

     and

     one

     of

     the

     largest

     cities

     in

     both

     the

     country

     and

     the

     world

    .

     It

     is

     known

     for

     its

     beautiful

     architecture

    ,

     vibrant

     culture

    ,

     and

     delicious

     food

    .

     The

     city

     is

     also

     famous

     for

     its

     annual

     celebration

     of

     Paris

     Festival

    ,

     which

     attracts

     millions

     of

     visitors

     every

     year

    .

     However

    ,

     its

     economic

     and

     political

     power

     has

     declined

     over

     time

    ,

     and

     it

     is

     now

     considered

     a

     small

     city

     with

     a

     declining

     population

    .

     Despite

     this

    ,

     Paris

     remains

     a

     cultural

     and

     historical

     center

     of

     France

    ,

     and

     it

     continues

     to

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     very

     different

     from

     where

     we

     are

     today

    ,

     but

     it

     is

     possible

     that

     it

     could

     develop

     in

     quite

     a

     different

     direction

    .

     
    


    One

     potential

     area

     of

     focus

     for

     future

     AI

     development

     is

     developing

     algorithms

     that

     can

     learn

     from

     and

     adapt

     to

     new

     data

    .

     For

     example

    ,

     researchers

     are

     currently

     working

     on

     developing

     algorithms

     that

     can

     recognize

     patterns

     in

     un

    structured

     data

    ,

     such

     as

     speech

     or

     images

    .

     If

     we

     can

     create

     algorithms

     that

     can

     learn

     from

     and

     adapt

     to

     new

     data

    ,

     it

     could

     be

     possible

     for

     AI

     to

     provide

     more

     personalized

     and

     accurate

     recommendations

     and

     predictions

    .
    


    Another

     area

     of

     focus

     for

     AI

     development

     is

     developing

     AI

     that

     can

     interact

     with

     humans

     in

     new

     ways

    .

     For

     example

    ,

     some

     researchers

    



```python
llm.shutdown()
```
