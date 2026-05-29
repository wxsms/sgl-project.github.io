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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.71it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.71it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.52it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.84it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.84it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.84it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.84it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.84it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.84it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.84it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.84it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.84it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.84it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.84it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.24it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.24it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.24it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.24it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.24it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.24it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.24it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.24it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.24it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.24it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.24it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.29it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.29it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.29it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.29it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.29it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.29it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.29it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   9%|▊         | 5/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.12 GB):   9%|▊         | 5/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.16it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.16it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.16it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.16it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.75 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.76it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.76it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.76it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.76it/s] Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=896 avail_mem=74.64 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.43it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.15it/s]Capturing num tokens (num_tokens=384 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.15it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.15it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.15it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.15it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.15it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.02it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.64it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.64it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.64it/s]Capturing num tokens (num_tokens=112 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.64it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.64it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.64it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.61it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.61it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=20 avail_mem=74.55 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=16 avail_mem=74.55 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.63it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 47.24it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 40.62it/s]


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
    Generated text:  Peter and I'm a high school student. My name is in the back of my backpack, and my class is at the front of my backpack. My last name is in the back and my name is on the front. Who has the last name in the back?
    A) Peter
    B) My last name
    C) My name
    D) My class
    E) I
    To determine who has the last name in the back of their backpack, let's analyze the information given step by step:
    
    1. **Peter**: This person is a high school student and is clearly identified by their name in the back of their backpack.
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-level official of the government of the United States. The President of the United States is elected by the citizens of the United States. Before he/she is elected, he/she has to be chosen by the state legislatures, the people or both of the two options. It is the very first person in charge of the country. During the time of the first three terms of the presidency, the president gets the title of the "Father of the Country" and also, the right to the United States flag. The president is the commander-in-chief of the armed forces of the United States. The president's powers are made up of the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Rome
    D. Athens
    
    The capital of France is Paris. Therefore, the correct answer is:
    
    A. Paris
    B. London
    C. Rome
    D. Athens
    A. Paris
    B. London
    C. Rome
    D. Athens
    The capital of France is Paris, located in the western part of the country, at the mouth of the Seine River. The city is the seat of the French government and is home to many landmarks and cultural institutions. Paris is also known for its elegant architecture, Parisian cuisine, and its role in the French Revolution.
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to be different from what we think.
    Tech companies are investing in AI to help improve healthcare, create smart homes, and improve the quality of life for millions of people.
    In the process of making progress in AI, we have to be careful about the potential risks. The general public should be informed about the benefits of AI and the potential risks.
    The AI revolution is not over. There are many areas where AI can help us make progress. But it is essential to understand the potential risks. The following 10 points will help you understand AI risks and how to minimize them.
    1. Machine Learning Risks
    1. 1


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and a diverse population of over 2 million people. The city is known for its fashion, art, and cuisine, and is a popular tourist destination. Paris is a city of contrasts, with its historical architecture and modern skyscrapers blending seamlessly into the urban landscape. The city is also home to many international organizations and events, making it
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from smart home devices to self-driving cars. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption of AI in our daily routines.
    
    2. AI will become more ethical and transparent: As AI becomes more integrated into our daily lives, we can expect to see more ethical and transparent AI. This
    


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
    Generated text:  [Your Name] and I'm a [your occupation or profession] with [how long you've been working in the field]. I enjoy [your specialty or area of expertise]. I'm a [your age, gender, etc.] who [what makes you stand out]. I love [what interests you or hobbies that make you unique]. I believe in [a particular belief or principle]. I'm excited to learn more about your career and personal journey. How can we connect on a deeper level?
    
    Hello, my name is [Your Name] and I'm a [your occupation or profession] with [how long you've been working in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city where the Eiffel Tower and the Louvre Museum are located. It is the country's economic and cultural center and is known for its rich history and vibrant art scene. The city is also a major transportation hub and has a large population, making it an important cultural and political center in France. The French government has been centered in Paris since the 18th century, and the city remains an important cultural and political center in France. Its history and architecture are also famous around the world, including in UNESCO World Heritage sites. Paris is known for its vibrant nightlife, famous food and wine, and its popularity as
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  complex and constantly evolving. Here are some possible trends that are expected to shape the industry:
    
    1. Increased Integration with Human Intuition and Experience: The future of AI is likely to see greater integration with human intuition and experience. This will allow AI systems to learn from and adapt to human behavior, enabling more accurate and efficient decision-making.
    
    2. More Autonomous AI: Autonomous AI, which is the ability of AI to make decisions and actions without human intervention, is expected to become more prevalent. This will lead to more autonomous vehicles, automated healthcare systems, and other applications that can operate on their own without human oversight.
    
    3. Enhanced Privacy and


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

     name

     here

    ],

     and

     I

    'm

     a

     [

    insert

     profession

     here

    ].

     I

    've

     always

     loved

     learning

     and

     understanding

    ,

     and

     I

    'm

     always

     eager

     to

     share

     what

     I

    've

     learned

     with

     others

    .

     Whether

     it

    's

     through

     my

     passion

     for

     writing

     or

     through

     my

     love

     for

     travel

     and

     outdoor

     activities

    ,

     I

    'm

     always

     looking

     for

     ways

     to

     connect

     with

     others

     and

     share

     my

     experiences

    .

     And

     when

     it

     comes

     to

     my

     hobbies

    ,

     I

     love

     to

     spend

     time

     outdoors

     and

     exploring

     new

     places

    ,

     whether

     it

    's

     hiking

    ,

     camping

    ,

     or

     just

     loung

    ing

     on

     the

     beach

    .

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

     too

    !

     [

    Add

     any

     relevant

     personal

     information

     or

     background

    ]

     [

    Add

     any

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Here

     is

     a

     summary

     of

     the

     previous

     statement

    :
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     serves

     as

     its

     capital

    .

     It

     is

     also

     the

     most

     populous

     city

     in

     Europe

    ,

     with

     over

     

    2

    0

     million

     inhabitants

    .

     The

     city

    's

     historical

     significance

     and

     cultural

     attractions

     make

     it

     an

     important

     center

     for

     politics

    ,

     entertainment

    ,

     and

     art

    .

     Paris

     is

     also

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     romantic

     and

     beautiful

     landscape

    .

     
    


    A

     brief

     and

     concise

     summary

     could

     be

    :


    Paris

     is

     the

     capital

     city

     of

     France

    ,

     the

     largest

     and

     most

     populous

     city

     in

     Europe

    ,

     famous

     for

     its

     historical

     significance

    ,

     cultural

     attractions

    ,

     and

     romantic

     landscape

    .

     
    


    I

    'm

     trying

     to

     do

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     increasingly

     complex

     and

     uncertain

    .

     Here

     are

     some

     possible

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Human

     Interaction

    :

     As

     AI

     technology

     advances

    ,

     more

     people

     will

     interact

     with

     it

     in

     the

     form

     of

     virtual

     assistants

     and

     companions

    .

     These

     will

     enhance

     the

     user

     experience

     by

     providing

     personalized

     assistance

     and

     recommendations

     based

     on

     their

     preferences

     and

     needs

    .
    


    2

    .

     Adv

    ancements

     in

     Machine

     Learning

    :

     Machine

     learning

     algorithms

     will

     continue

     to

     improve

    ,

     enabling

     AI

     systems

     to

     learn

     from

     more

     data

     and

     make

     more

     accurate

     predictions

     and

     decisions

    .

     This

     will

     lead

     to

     more

     efficient

     and

     effective

     AI

     systems

    .
    


    3

    .

     Eth

    ical

     and

     Eth

    ical

     Concern

    s

    :

     As

     AI

     continues

     to

     develop

    ,

     there

     will

     be

     growing

     concerns

     about

     issues

     such

     as

     bias

    ,

    



```python
llm.shutdown()
```
