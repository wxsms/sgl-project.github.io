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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.52it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.51it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.32it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.32it/s]

    Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.32it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.83it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.83it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.83it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.83it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.83it/s]

    Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.83it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.83it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.83it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.83it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.83it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 28.73it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 40.11it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.51it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.51it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.51it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  21%|██        | 12/58 [00:00<00:01, 28.69it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:01, 28.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:01, 28.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.70it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.70it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.70it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.70it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.89it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.89it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.89it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.89it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.89it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.89it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.00it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.00it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.00it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.00it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.00it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.00it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.21it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.62it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.62it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.62it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.62it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.62it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.62it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.75it/s] Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  81%|████████  | 47/58 [00:01<00:00, 46.23it/s]

    Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.67it/s] Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 40.77it/s]


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
    Generated text:  Layla and I have been a coach for many years. I have been teaching and training in basketball for 15 years. I am a Certified Physical Fitness Instructor (CPFI), certified in Strength Training and Yoga and Certified Coaching Coach (CCC).
    I have worked with individuals from ages 6-80 and have coached teams for many years. I am passionate about helping athletes improve through the development of their physical fitness, overall skills and sportsmanship.
    For over 20 years, I have been teaching young people through our local and national clubs. I've taught both high school basketball, youth basketball and college basketball. I am a
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. Someone who holds the office of president of the United States is a person. Therefore, the president of the United States is a person. Which of the following, if true, would most weaken the conclusion that the president of the United States is a person?
    A: The president of the United States was once a hostage in a hostage crisis.
    B: The president of the United States is from a country that is significantly wealthy compared to other countries in the world.
    C: The president of the United States has been a candidate for president, but has not been elected.
    D: The president of the United States has 13
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, located in the heart of the Paris Basin, bounded by the Alps and the Garonne River on the west, the Seine and the Pyrenees on the east, and the Atlantic Ocean to the north and the Loire Valley to the south. The capital is situated at the base of the Paris Basin and is located in the heart of the Paris Basin. It is located in the 5th arrondissement and is divided into 31 wards, which are the administrative division of the city of Paris. The city of Paris is divided into 110 neighborhoods, 60 of which are reserved for public areas
    ===============================
    Prompt: The future of AI is
    Generated text:  here. The first wave of AI is changing the way we live and work. It is also changing the way we will work in the future.
    An AI is a machine that has the ability to perform tasks that would normally require human intelligence. AI is being used in industries like healthcare, finance, transportation, and manufacturing, among others. There are many different types of AI, including rule-based AI, semi-automatic AI, and intelligent agents.
    Here are 5 reasons why AI is changing the way we live and work in the future:
    1. Automation of tasks: AI is making tasks that were once done by humans easier and more efficient.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a unique skill or trait that sets me apart from other candidates]. And what's your background? I have [insert a relevant experience or education]. And what's your favorite hobby or activity? I enjoy [insert a hobby or activity that you enjoy doing]. And what's your favorite book or movie? I love [insert a favorite book or movie that you enjoy reading or watching]. And what's your favorite color? I love [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a bustling metropolis with a diverse population and is a major tourist destination. It is home to many famous French artists, writers, and musicians. The city is also known for its cuisine, with dishes like croissants, beignets, and escargot being popular. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique and fascinating city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As the technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more personalized and accurate diagnoses and treatments.
    
    2. Increased Use of AI in Manufacturing: AI is already being used in manufacturing to improve efficiency and reduce costs. As the technology continues to improve, we can
    


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
    Generated text:  Alex, and I am a talented writer and artist. I am a self-taught artist who has been painting and drawing for many years, and I am passionate about using my creativity to create engaging and thought-provoking artwork. I am a lifelong learner and always look for new and exciting ideas to incorporate into my work. I have been given a lot of responsibility in my art and creative endeavors, and I am excited to share my work with others. I am also an avid reader and enjoy exploring new genres and authors. Thank you for having me! [Alex's personality and style are not specified in the prompt, so this is
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a major cultural and economic center located in the Paris region, on the river Seine and on the Île de la Cité. It is the seat of the French government and is home to numerous historical landmarks and museums. Paris is known for its beautiful architecture, culinary delights, and annual festivals such as the Fête de la Feuille. It is also home to many famous artists and writers, including Claude Monet, Edouard Manet, and Gustave Courbet. With its rich history and lively nightlife, Paris is a must-visit destination for any traveler interested in French culture and history. 
    
    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be both exciting and rapidly changing, with many new applications and technologies expected to emerge in the coming years. Here are some possible future trends in AI:
    
    1. Increased AI Transparency: As AI systems become more complex, there will be more opportunities for transparency and accountability. AI developers will need to increase their transparency by sharing more details about how they make decisions and why they are made. This will help to ensure that AI systems are reliable and safe, and that users have a better understanding of how AI is being used.
    
    2. Integration of AI with Natural Language Processing (NLP): AI systems are becoming increasingly integrated into our daily lives


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

    ...

     [

    Name

    ].

     I

    'm

     a

     [

    type

     of

     character

    ]

     named

     [

    character

     name

    ],

     and

     I

    'm

     [

    character

    's

     age

    ]

     years

     old

    .

     I

    'm

     [

    role

     in

     the

     story

    ].

     [

    Name

    ]

     was

     born

     in

     [

    year

     of

     birth

    ]

     and

     grew

     up

     in

     [

    location

    ].

     I

    'm

     a

     [

    description

     of

     character

    ].

     I

    've

     been

     [

    character

    's

     interests

    ,

     hobbies

    ,

     or

     skills

    ]

     for

     [

    number

     of

     years

    ].

     I

    'm

     [

    character

    's

     personality

     traits

    ].

     And

     I

    'm

     [

    character

    's

     aspirations

     or

     dream

    ].

     Thank

     you

     for

     asking

    ,

     [

    Name

    ].

     I

    'm

     glad

     to

     meet

     you

    ,

     and

     I

     hope

     we

     can

     discuss

     our

     interests

     in

     more

     detail

    .

     [

    Name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    The

     statement

     provided

     is

     accurate

     and

     factual

    .

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     serving

     as

     the

     political

    ,

     economic

    ,

     cultural

    ,

     and

     administrative

     center

     of

     the

     country

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     vibrant

     culture

    ,

     and

     renowned

     museums

    ,

     among

     other

     things

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     economic

     center

     in

     the

     European

     Union

    .

     The

     city

     is

     also

     known

     for

     its

     wine

     production

     and

     gastr

    onomy

    .

     As

     the

     largest

     city

     in

     France

    ,

     Paris

     plays

     a

     significant

     role

     in

     the

     country

    's

     cultural

     and

     political

     landscape

    .

     The

     statement

     is

     a

     simple

     and

     accurate

     representation

     of

     the

     facts

     about

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     continued

     advancements

     and

     innovations

     as

     the

     technology

     continues

     to

     evolve

    .

     Some

     possible

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     automation

    :

     AI

     will

     continue

     to

     automate

     more

     tasks

    ,

     freeing

     up

     more

     time

     and

     resources

     for

     humans

     to

     focus

     on

     more

     complex

     tasks

    .
    


    2

    .

     Improved

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     integrated

     into

     society

    ,

     it

     is

     important

     that

     ethical

     considerations

     are

     taken

     into

     account

    .

     This

     could

     lead

     to

     the

     development

     of

     AI

     that

     is

     more

     ethical

     and

     accountable

    .
    


    3

    .

     AI

     integration

     with

     human

     intuition

    :

     The

     development

     of

     AI

     that

     can

     understand

     and

     interpret

     human

     emotions

    ,

     intuition

    ,

     and

     language

     could

     lead

     to

     more

     empath

    etic

     and

     effective

     AI

     systems

    .
    


    4

    .

     AI

     integration

    



```python
llm.shutdown()
```
