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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.08it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.08it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.89it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.69it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.69it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.69it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.69it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.69it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.69it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.69it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.69it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.69it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 23.69it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:04<00:00, 31.98it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.12 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.00it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.32it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=75.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=960 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.32it/s] Capturing num tokens (num_tokens=960 avail_mem=75.07 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.81it/s]Capturing num tokens (num_tokens=896 avail_mem=75.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.81it/s]Capturing num tokens (num_tokens=832 avail_mem=75.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.81it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.81it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.81it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.81it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.74it/s]Capturing num tokens (num_tokens=384 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.74it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.74it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.74it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.74it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.74it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.59it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.59it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.59it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.59it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.59it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.59it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.42it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.42it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.42it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.42it/s]Capturing num tokens (num_tokens=112 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.42it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.42it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 41.66it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 41.66it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 41.66it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 41.66it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 41.66it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 41.66it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=20 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=16 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.61it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.61it/s] Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.41it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.41it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 36.98it/s]


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
    Generated text:  Fred Thompson. I am a student at the University of Southern California studying atmospheric science and climate change. I specialize in research and teaching. I am also a member of the Geophysical Fluid Dynamics Laboratory, a research facility at the University of Southern California.
    I am fascinated by climate change and its effects on the environment, the ocean, and the human species. My research interests include modeling, analysis, and forecasts of climate change, ocean currents and waves, and global change processes. I have also been involved in several studies related to the effects of climate change on global food production.
    As a teacher, I help my students to develop critical thinking,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a famous leader in the world. The most important job of the president is to be ________ the country. ____ A. talk B. tell C. talk to D. to talk To determine the correct answer, we need to understand the sentence structure and context.
    
    The sentence is: "The most important job of the president is to be ________ the country."
    
    Let's break down the sentence:
    - "The most important job" means the most important job in the context.
    - "of the president" is a relative pronoun meaning "of the president," which fits the blank.
    - "is" is a linking verb meaning "
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Marseille
    C. Lyon
    D. Toulouse
    Answer: A
    
    The RMB can be traded on the inter-bank foreign exchange market under the ___ system.
    A. RMB foreign exchange market
    B. Foreign currency foreign exchange market
    C. Renminbi foreign exchange market
    D. Central bank foreign exchange market
    Answer: C
    
    To set up a general partnership, a partnership agreement should be signed with ____.
    A. More than 3 people
    B. More than 2 people
    C. More than 4 people
    D. More than 5 people
    Answer
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, and there are many exciting developments in the field. From creating more accurate models for language processing, to developing more advanced algorithms for self-driving cars, to exploring new applications for AI in healthcare and finance, there is so much to be excited about.
    One of the key areas that is gaining attention in the field of AI is the development of more advanced and sophisticated models. These models are being used to achieve a range of important applications, including natural language processing, computer vision, and natural language understanding.
    One of the most exciting developments in the field of AI is the use of deep learning, a type of machine learning that involves training models


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [Age], [Gender], [Nationality], [Occupation], and I have [Number] years of experience in [Field of Work]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [Favorite Activity], and I'm always looking for new ways to expand my skills and knowledge. What's your dream job? I dream of [Dream Job], where I can use
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for business, finance, and tourism, making it a popular destination for tourists and locals alike. The city is known for its fashion industry, with many famous fashion designers and boutiques located in the city. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as
    


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
    Generated text:  [Name] and I am a [occupation] with over [number] years of experience. I come from [location], and I've always been passionate about [field of interest]. I have a genuine interest in [interest or hobby], and I'm always eager to learn new things. I thrive on challenges and I'm always looking for ways to grow and improve. What excites me the most is [motivational quote or statement]. I'm always open to feedback and am always striving to improve myself. Thank you. [Name] looks forward to meeting you and shares their enthusiasm for their profession and passions.
    The introduction is neutral
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a major city in the French region of the same name, and serves as the seat of government, capital, and largest city of the French Republic. The city is located on the Seine River in the Loire Valley, and is home to many notable landmarks, including the Eiffel Tower and Notre-Dame Cathedral. Paris is known for its rich history, including its role in the French Revolution and its influence on art, literature, and music. The city is also famous for its fashion industry, especially for couture and haute couture, and its role in the French cultural life. Paris is one of the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  one of exploration and exploration. While AI has already made great progress in various areas such as machine learning, natural language processing, computer vision, and robotics, there is still much to be discovered. Here are some possible future trends in AI:
    
    1. Deep learning and neural networks: As deep learning and neural networks continue to improve, we are likely to see more sophisticated and accurate models that can handle complex and varied data. This could lead to even more advanced tasks like image and speech recognition, autonomous vehicles, and personalized medicine.
    
    2. Explainable AI: While AI has made great progress in recent years, there is still much to be done


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

     of

     character

    ],

     and

     I

    'm

     a

     [

    insert

     occupation

    ]

     from

     [

    insert

     country

    ].

     I

     love

     [

    insert

     hobby

     or

     interest

    ]

     and

     I

    'm

     always

     up

     for

     [

    insert

     activity

    ].

     What

    's

     your

     profession

     or

     job

     title

    ?

     Let

    's

     chat

     about

     something

     fun

     and

     interesting

    !

     [

    insert

     personality

     trait

     or

     characteristic

    ].

     Nice

     to

     meet

     you

    ,

     [

    insert

     name

     of

     character

    ].

     Feel

     free

     to

     ask

     me

     anything

    !

     [

    insert

     personality

     trait

     or

     characteristic

    ].

     I

     hope

     to

     have

     a

     great

     time

     with

     you

    !

     [

    insert

     name

     of

     character

    ].

     Welcome

     to

     our

     world

    ,

     [

    insert

     name

     of

     character

    ].

     And

     who

     knows

    ,

     maybe

     we

    'll

     end

     up

     working

     together

     one

     day

    !

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    .

     True

     B

    .

     False

    


    B

    .

     False

    
    


    The

     capital

     city

     of

     France

     is

     Paris

    .

     
    


    A

    .

     True

     B

    .

     False

    
    


    The

     capital

     of

     France

     is

     Paris

    .

     


    The

     capital

     of

     France

     is

     indeed

     Paris

    .

     As

     of

     my

     last

     update

     in

     

    2

    0

    2

    3

    ,

     Paris

     is

     the

     capital

     city

     of

     France

    .

     This

     is

     a

     factual

     statement

     that

     is

     commonly

     recognized

     and

     widely

     known

    .

     
    


    Therefore

    ,

     the

     correct

     answer

     is

     B

    .

     False

    .

     Paris

     is

     the

     capital

     of

     France

    ,

     not

     the

     other

     way

     around

    .

     
    


    If

     you

     need

     any

     more

     information

     or

     clarification

     on

     this

     topic

    ,

     feel

     free

     to

     ask

    !

     
    


    For

     example

    :


    -

     Can

     you

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

     and

     it

    's

     likely

     to

     continue

     changing

     rapidly

    .

     Some

     possible

     trends

     in

     the

     AI

     field

     include

    :
    


    1

    .

     Deep

     Learning

    :

     Deep

     learning

     is

     an

     algorithm

     that

     uses

     multiple

     layers

     of

     neural

     networks

     to

     solve

     complex

     problems

    .

     As

     the

     size

     and

     complexity

     of

     data

     increases

    ,

     deep

     learning

     is

     becoming

     more

     and

     more

     powerful

    .
    


    2

    .

     Explain

    ability

    :

     As

     AI

     systems

     get

     more

     complex

    ,

     they

     are

     becoming

     more

     difficult

     to

     understand

    .

     However

    ,

     researchers

     are

     developing

     techniques

     to

     explain

     how

     AI

     systems

     make

     decisions

     and

     to

     detect

     biases

     in

     their

     algorithms

    .
    


    3

    .

     Bias

     and

     Fair

    ness

    :

     AI

     systems

     can

     perpet

    uate

     biases

     if

     they

     are

     trained

     on

     biased

     data

    .

     To

     address

     this

     issue

    ,

     researchers

    



```python
llm.shutdown()
```
