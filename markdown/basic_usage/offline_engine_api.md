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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 14.07it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 14.07it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 14.07it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 14.07it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 14.07it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 14.07it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 14.07it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 14.07it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 14.07it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:02, 14.07it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 21.23it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 21.23it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 21.23it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 21.23it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 21.23it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 21.23it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 21.23it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 21.23it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 21.23it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:00, 21.23it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 29.38it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 29.38it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 29.38it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 29.38it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 29.38it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 29.38it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 29.38it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 29.38it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 29.38it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 29.38it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 29.38it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 39.32it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 39.32it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 39.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.08it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.07 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.06 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.06 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.06 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=69.06 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.04 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.04 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=69.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.03 GB):  21%|██        | 12/58 [00:00<00:01, 26.94it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.02 GB):  21%|██        | 12/58 [00:00<00:01, 26.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.02 GB):  21%|██        | 12/58 [00:00<00:01, 26.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.02 GB):  21%|██        | 12/58 [00:00<00:01, 26.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.02 GB):  21%|██        | 12/58 [00:00<00:01, 26.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.01 GB):  21%|██        | 12/58 [00:00<00:01, 26.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=69.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.82it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=68.98 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.82it/s]Capturing num tokens (num_tokens=960 avail_mem=69.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.82it/s] Capturing num tokens (num_tokens=960 avail_mem=69.00 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.38it/s]Capturing num tokens (num_tokens=896 avail_mem=69.00 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.38it/s]Capturing num tokens (num_tokens=832 avail_mem=68.99 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.38it/s]Capturing num tokens (num_tokens=768 avail_mem=68.99 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.38it/s]Capturing num tokens (num_tokens=704 avail_mem=68.99 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.38it/s]Capturing num tokens (num_tokens=640 avail_mem=68.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.38it/s]Capturing num tokens (num_tokens=640 avail_mem=68.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=576 avail_mem=68.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=512 avail_mem=68.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=480 avail_mem=68.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]

    Capturing num tokens (num_tokens=448 avail_mem=68.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=416 avail_mem=68.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=416 avail_mem=68.98 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.55it/s]Capturing num tokens (num_tokens=384 avail_mem=68.98 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.55it/s]Capturing num tokens (num_tokens=352 avail_mem=68.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.55it/s]Capturing num tokens (num_tokens=320 avail_mem=68.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.55it/s]Capturing num tokens (num_tokens=288 avail_mem=68.96 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.55it/s]Capturing num tokens (num_tokens=256 avail_mem=68.96 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.55it/s]Capturing num tokens (num_tokens=256 avail_mem=68.96 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=240 avail_mem=68.96 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=224 avail_mem=68.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.12it/s]

    Capturing num tokens (num_tokens=208 avail_mem=68.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=192 avail_mem=68.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=176 avail_mem=68.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=176 avail_mem=68.95 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=160 avail_mem=68.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=144 avail_mem=68.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=128 avail_mem=68.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=112 avail_mem=68.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=96 avail_mem=68.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.51it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=68.93 GB):  81%|████████  | 47/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=80 avail_mem=68.93 GB):  81%|████████  | 47/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=64 avail_mem=68.92 GB):  81%|████████  | 47/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=48 avail_mem=68.92 GB):  81%|████████  | 47/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=32 avail_mem=68.92 GB):  81%|████████  | 47/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=28 avail_mem=68.40 GB):  81%|████████  | 47/58 [00:01<00:00, 40.24it/s]

    Capturing num tokens (num_tokens=28 avail_mem=68.40 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.14it/s]Capturing num tokens (num_tokens=24 avail_mem=68.40 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.14it/s]Capturing num tokens (num_tokens=20 avail_mem=68.39 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.14it/s]Capturing num tokens (num_tokens=16 avail_mem=68.39 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.14it/s]Capturing num tokens (num_tokens=12 avail_mem=68.39 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.14it/s]Capturing num tokens (num_tokens=12 avail_mem=68.39 GB):  97%|█████████▋| 56/58 [00:01<00:00, 29.73it/s]Capturing num tokens (num_tokens=8 avail_mem=68.38 GB):  97%|█████████▋| 56/58 [00:01<00:00, 29.73it/s] Capturing num tokens (num_tokens=4 avail_mem=68.38 GB):  97%|█████████▋| 56/58 [00:01<00:00, 29.73it/s]Capturing num tokens (num_tokens=4 avail_mem=68.38 GB): 100%|██████████| 58/58 [00:01<00:00, 33.72it/s]


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
    Generated text:  Georgie. I'm a software engineer at a big tech company. I've been working on a project that involves analyzing and optimizing data, and I've been learning more and more about machine learning. I'm fascinated by how we can leverage machine learning to improve our systems and better understand the world around us. My latest project is an AI-powered language translation tool that can translate between multiple languages. Despite the challenges that come with it, I've been impressed by the results that the tool has achieved and I'm eager to learn more about how machine learning can be used in real-world applications. I enjoy exploring the intersection of technology and data science
    ===============================
    Prompt: The president of the United States is
    Generated text:  married to a woman named Claire. Who is president of the United States? The president of the United States is the Chief Executive Officer (CEO) of the country. When a person is a head of a country, they are known as its president. Claire is the married wife of a president. Since the President is the head of the government and is married to the head of the government, the President cannot be Claire's husband. The President is the head of the government, and the President's spouse is not the head of the government. That makes Claire's husband the president. The President is the head of the government, so the spouse
    ===============================
    Prompt: The capital of France is
    Generated text:  the capital of a different country, which is located 490 km away from the capital of France. Another capital of a different country is 60 km away from the capital of that country. If the distance between these two capitals is 120 km, find the distance between the first capital and the second capital.
    Let's denote the distance between the first capital and the second capital as \( d \) km.
    
    From the information given, we know the following:
    1. The capital of the first country is 490 km away from the capital of France.
    2. The capital of the second country is 
    ===============================
    Prompt: The future of AI is
    Generated text:  very interesting. I am a little worried about the amount of data that AI has access to. I am not sure how to deal with this issue.
    Sure, I can help you with that. AI systems need to have access to large amounts of data to learn and make predictions. However, as the volume and complexity of data increase, the amount of data that can be accessed can become overwhelming. 
    
    One solution to this problem is to use distributed computing. This involves dividing the data across multiple machines or processors, so that each machine can focus on a portion of the data. This can significantly reduce the amount of data that needs to be accessed


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Gender] [Gender] and I have always been [Positive Traits]. I am [Positive Traits] and I am [Positive Traits]. I am a [Positive Traits] and I am [Positive Traits]. I am a [Positive Traits] and I am [Positive Traits]. I am a [Positive Traits] and I am [Positive Traits]. I am a [Positive Traits] and I am [Positive Traits]. I am a [Positive Traits] and I am [Positive Traits]. I am a [Positive Traits] and I am [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Museum of Modern Art. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its cuisine, including its famous croissants and its traditional French cuisine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that is both ancient and modern, and is a must-visit destination for anyone
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there will be an increased focus on ethical AI. This will likely lead to more stringent regulations and guidelines for AI development and deployment.
    
    2. AI will become more integrated with other technologies: As AI becomes more integrated with other technologies, such as machine learning, natural language processing, and computer vision, it is likely that we will see even more complex and sophisticated AI systems emerge.
    
    3. AI will become
    


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
    Generated text:  [insert name] and I'm a fictional character. What's your background, education, and hobbies? As a fictional character, I don't have a physical presence, so I can't have a background or education. However, I can provide information about my hobbies and interests if you're interested. What's your favorite hobby or activity? As a fictional character, I don't have a hobby, so I can't provide information about my hobbies. However, I can tell you that I enjoy spending time with my friends and family, reading books, and watching movies. What's your favorite book or movie? As a fictional character, I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and one of the largest cities in the world. The city has a rich history dating back to the Roman Empire, and has been a major center of culture, politics, and religion for centuries. It is known for its beauty, art, cuisine, and fashion, and is a popular tourist destination. Paris has a diverse population of more than 2 million residents, and the city's vibrant arts scene is a major attraction for foreign visitors. The city is home to many iconic landmarks and museums, including the Louvre and the Eiffel Tower. Paris is a city of contrasts, with towering
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several key trends:
    
    1. Increased AI Transparency: As AI systems become more complex, their underlying algorithms and decision-making processes will become more apparent. This means that developers will be able to better understand how AI systems work and how they are making decisions. As a result, there will be a greater emphasis on transparency in AI development and deployment.
    
    2. AI Ethics and Responsibility: As AI becomes more integrated into our daily lives, there will be increasing concerns about its impact on society. This will likely lead to a greater emphasis on AI ethics and responsibility. This means that developers will need to be more careful about the AI


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

    name

    ],

     and

     I

     am

     a

     [

    occupation

    ]

     with

     [

    length

     of

     experience

    ],

     [

    number

     of

     previous

     positions

    ].

     I

     am

     dedicated

     to

     [

    job

     title

    ],

     [

    number

     of

     years

    ]

     years

     of

     experience

    ,

     and

     have

     [

    number

     of

     accomplishments

    ]

     accomplishments

     in

     my

     career

    .

     I

     am

     passionate

     about

     [

    job

     title

    ]

     and

     would

     love

     to

     have

     the

     opportunity

     to

     [

    job

     title

    ]

     at

     [

    company

     name

    ].

     Thank

     you

     for

     considering

     me

     as

     an

     interview

     candidate

    .

     Let

     me

     know

     if

     you

     would

     like

     me

     to

     elaborate

     on

     my

     qualifications

     or

     provide

     more

     information

     about

     my

     career

    .

     [

    Name

    ]

     [

    Title

    ]

     [

    Number

     of

     Years

    ]

     [

    Number

     of

     Accom

    pl

    ishments

    ]

     [

    Job

     Title

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     country

    ,

     and

     it

     serves

     as

     the

     political

    ,

     economic

    ,

     cultural

    ,

     and

     historical

     center

     of

     France

    .

     Paris

     is

     renowned

     for

     its

     history

    ,

     art

    ,

     architecture

    ,

     food

    ,

     fashion

    ,

     and

     music

    .

     The

     city

     is

     also

     known

     for

     its

     cosm

    opolitan

     culture

     and

     its

     status

     as

     the

     birth

    place

     of

     many

     famous

     people

    ,

     including

     Napoleon

     Bon

    ap

    arte

     and

     Marie

     Cur

    ie

    .

     
    


    Paris

     is

     home

     to

     a

     rich

     tape

    stry

     of

     historical

     and

     architectural

     landmarks

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

     the

     Arc

     de

     Tri

    omp

    he

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     multi

    -f

    ac

    eted

     and

     diverse

    ,

     driven

     by

     a

     range

     of

     emerging

     trends

     and

     technologies

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     AI

     landscape

     in

     the

     years

     ahead

    :
    


    1

    .

     Increased

     Human

    -A

    I

     Collaboration

    :

     Human

    -A

    I

     collaboration

     could

     become

     more

     common

    ,

     as

     AI

     systems

     become

     more

     capable

     of

     learning

     and

     adapting

     to

     the

     human

     needs

     and

     behaviors

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     AI

     systems

     that

     can

     work

     alongside

     human

     professionals

     in

     a

     variety

     of

     domains

    .
    


    2

    .

     Autonomous

     and

     Semi

    -A

    ut

    onomous

     Robots

    :

     The

     development

     of

     autonomous

     and

     semi

    -aut

    onomous

     robots

     could

     lead

     to

     a

     new

     era

     of

     workplace

     automation

     and

     increased

     efficiency

    .

     These

     robots

     could

     perform

    



```python
llm.shutdown()
```
