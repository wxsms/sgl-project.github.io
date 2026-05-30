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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.45it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.18it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.95it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 17.97it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 17.97it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 17.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 17.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 20.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 20.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 20.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 20.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.47it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.47it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.91it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.91it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.91it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.91it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.91it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.91it/s] Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.27it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.27it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.27it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.27it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.27it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.27it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.63it/s]

    Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.15it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.15it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.15it/s]Capturing num tokens (num_tokens=320 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.15it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.15it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.15it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.94it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.08it/s] Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 44.61it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  81%|████████  | 47/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.01it/s] Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 39.64it/s]


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
    Generated text:  Mike. I have a few questions about the economy. Could you tell me how the US economy compares to that of China? Sure, I'd be happy to help with that! The US economy is generally considered to be larger and more complex than China's. China, on the other hand, is a much more advanced country with a lower standard of living.
    
    China's economy is largely driven by its huge population and rapid growth, with urbanization and population growth driving much of its growth. China also has a large, well-developed infrastructure, with major cities and transportation networks that support its economic growth.
    
    On the other hand, the US
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 45 years old. If the president of the United States was 15 years old when he was 35 years old, how many years from now will it be until the president is 45 years old? To determine how many years from now the president of the United States will be 45 years old, we need to follow these steps:
    
    1. Identify the current age of the president.
    2. Determine the age the president was 35 years ago.
    3. Calculate the age the president will be in 35 years.
    4. Determine the number of years from now until the president is 
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. For centuries, it was the capital of the King of France. After the Revolution, it was restored to the King and made a capital. In the 19th century, the city was considered the capital of France. The French Revolution has also had an impact on the city. In 1789, the city was declared the capital of France. The city has a long and proud history, and it has always been a popular tourist destination. 
    
    Based on the information provided, what can you say about Paris? 
    
    A) It is a small country.
    B) It is a popular tourist destination.
    C)
    ===============================
    Prompt: The future of AI is
    Generated text:  in the data
    The future of AI is in the data. As technology advances, so do its applications. But what is the future of AI? And what do we need to focus on to harness its full potential? In this piece, I will answer both of those questions.
    One of the most significant developments in AI over the past few years has been the rise of "big data." The idea is that much of AI today is based on large datasets that can be used to learn patterns and make predictions. Big data has become increasingly important, and it will continue to be a key area of focus for AI researchers in the years ahead.
    


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? I'm a [insert a short description of your character or personality]. And what's your favorite hobby or activity? I enjoy [insert a short description of your favorite hobby or activity]. And what's your favorite book or movie? I love [insert a short description of your favorite book or movie]. And what's your favorite place to go? I love [insert a short description of your favorite place to go]. And what's your favorite color
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament and the French Quarter, where many famous French artists and writers live and work. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a popular tourist destination and a major economic center in Europe. The city is known for its cuisine, fashion, and art, and is home to many international organizations and institutions. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more rigorous testing and evaluation of AI systems, as well as greater transparency and accountability in their development and deployment.
    
    3. Increased reliance on machine learning: Machine learning
    


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
    Generated text:  [Your Name]. I'm a [majority] artist born and raised in [your hometown]. I'm currently working on my first [major] project, and I'm looking for someone to help me with [reason for project]. I enjoy painting and using my creativity to express myself through visual art, and I'm always looking for new inspiration and ideas. If you're interested in helping me with this project, I'd be happy to share more about my work and how I can contribute. I'm excited to talk to you! [Your Name] [Your Contact Information]
    
    This is a great start, but can we add some information
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the center of the country in the Loire Valley, and is the largest city in Europe by population. It is home to many iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. It has a rich cultural heritage and is known for its fashion, gastronomy, and food scene. Paris is a vibrant and multicultural city, attracting tourists from all over the world. Its complex and rich history, which includes a 500-year-old history dating back to the Roman Empire, has made it one of the world's most visited cities. It is also known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends:
    
      1. Increased AI Integration with Natural Language Processing (NLP): AI systems will become more integrated with NLP to improve their ability to understand and interpret natural language.
      2. Advancements in Machine Learning and Deep Learning: Machine learning and deep learning are areas of AI that are likely to see continued innovation and advancement in the coming years. These technologies are expected to become more powerful and efficient at recognizing patterns and making predictions.
      3. Enhanced Cybersecurity: With the increasing number of cyber threats, it is expected that AI systems will become even more sophisticated and secure.
     


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

    ]

     and

     I

    'm

     a

     [

    Title

    ]

     at

     [

    Company

    ].

     I

     have

     a

     passion

     for

     [

    Your

     passion

     or

     interest

    ].

     What

     exc

    ites

     you

     the

     most

     about

     your

     work

    ?

     I

     love

     to

     [

    Describe

     an

     activity

     that

     interests

     you

    ].

     In

     your

     free

     time

    ,

     I

     enjoy

     [

    List

     any

     hobbies

     or

     interests

     you

     have

    ].

     How

     do

     you

     stay

     motivated

     and

     engaged

     in

     your

     work

    ,

     especially

     when

     faced

     with

     new

     challenges

     or

     projects

    ?

     I

     find

     that

     [

    Describe

     a

     technique

     that

     helps

     me

     stay

     focused

     and

     motivated

    ].

     How

     do

     you

     balance

     your

     work

     and

     personal

     life

    ?

     I

     make

     sure

     to

     take

     breaks

    ,

     eat

     well

    ,

     and

     prioritize

     my

     sleep

    .

     What

     advice

     would

     you

     give

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ré

    pub

    lique

    ."

     It

     is

     the

     city

     where

     the

     French

     Revolution

     took

     place

     and

     where

     the

     E

    iff

    el

     Tower

     is

     located

    .

     The

     city

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     is

     also

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     which

     was

     constructed

     in

     

    1

    8

    8

    9

     and

     has

     since

     become

     a

     symbol

     of

     the

     city

    .

     The

     French

     capital

     is

     an

     important

     part

     of

     French

     culture

    ,

     history

    ,

     and

     identity

    .

     Paris

     is

     a

     cosm

    opolitan

     city

     with

     a

     diverse

     population

     and

     a

     vibrant

     cultural

     scene

    .

     It

     is

     a

     city

     that

     attracts

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     unpredictable

    ,

     with

     no

     one

     sure

     where

     it

     will

     take

     us

     in

     the

     long

     run

    .

     However

    ,

     based

     on

     current

     trends

     and

     advancements

    ,

     here

     are

     some

     potential

     future

     trends

     that

     could

     shape

     the

     AI

     landscape

    :
    


    1

    .

     Increased

     collaboration

     between

     humans

     and

     AI

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     a

     growing

     convergence

     between

     humans

     and

     AI

     systems

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     decision

    -making

    ,

     as

     well

     as

     more

     intuitive

     and

     natural

     user

     interfaces

    .
    


    2

    .

     Artificial

     general

     intelligence

    :

     While

     AI

     can

     perform

     specific

     tasks

     with

     high

     accuracy

    ,

     it

     may

     not

     be

     able

     to

     perform

     complex

     cognitive

     tasks

     such

     as

     learning

    ,

     creativity

    ,

     or

     problem

    -solving

    .

    



```python
llm.shutdown()
```
