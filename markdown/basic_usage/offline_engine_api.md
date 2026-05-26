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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]

    Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.80it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.80it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.80it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.80it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.80it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.80it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]

    Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:04<00:08,  5.49it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:03, 10.26it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 17.78it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 17.78it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 17.78it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 17.78it/s]

    Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 17.78it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 17.78it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 17.78it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:01, 17.78it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:01, 17.78it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:01, 17.78it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 26.21it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 26.21it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 26.21it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 26.21it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 26.21it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 26.21it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 26.21it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 26.21it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 26.21it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 26.21it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 35.37it/s]

    Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 35.37it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 46.40it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 46.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=62.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=62.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=62.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=62.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=62.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=62.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=62.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=62.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=62.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=62.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=62.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=62.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=62.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=62.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=62.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.52it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=62.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=62.07 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=62.07 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=62.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.55 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.54 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.03it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=61.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.52 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=960 avail_mem=61.53 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.03it/s] Capturing num tokens (num_tokens=960 avail_mem=61.53 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.83it/s]Capturing num tokens (num_tokens=896 avail_mem=61.53 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.83it/s]Capturing num tokens (num_tokens=832 avail_mem=61.52 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.83it/s]Capturing num tokens (num_tokens=768 avail_mem=61.52 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.83it/s]Capturing num tokens (num_tokens=704 avail_mem=61.52 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.83it/s]Capturing num tokens (num_tokens=640 avail_mem=61.51 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.83it/s]Capturing num tokens (num_tokens=640 avail_mem=61.51 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=576 avail_mem=61.51 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.23it/s]

    Capturing num tokens (num_tokens=512 avail_mem=61.50 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=480 avail_mem=61.51 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=448 avail_mem=61.51 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=416 avail_mem=61.51 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=416 avail_mem=61.51 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.53it/s]Capturing num tokens (num_tokens=384 avail_mem=61.51 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.53it/s]Capturing num tokens (num_tokens=352 avail_mem=61.50 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.53it/s]Capturing num tokens (num_tokens=320 avail_mem=61.50 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.53it/s]Capturing num tokens (num_tokens=288 avail_mem=61.50 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=256 avail_mem=61.49 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=256 avail_mem=61.49 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=240 avail_mem=61.49 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.21it/s]

    Capturing num tokens (num_tokens=224 avail_mem=61.49 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=208 avail_mem=61.48 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=192 avail_mem=61.48 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=176 avail_mem=61.48 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=176 avail_mem=61.48 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=160 avail_mem=61.48 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=144 avail_mem=61.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=128 avail_mem=61.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=112 avail_mem=61.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=96 avail_mem=61.46 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.43it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=61.46 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=80 avail_mem=61.46 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=64 avail_mem=61.46 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=48 avail_mem=61.45 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=32 avail_mem=61.45 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=28 avail_mem=61.44 GB):  81%|████████  | 47/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=28 avail_mem=61.44 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=24 avail_mem=61.44 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=20 avail_mem=61.44 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=16 avail_mem=61.44 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=12 avail_mem=61.43 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.91it/s]

    Capturing num tokens (num_tokens=8 avail_mem=61.43 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.91it/s] Capturing num tokens (num_tokens=8 avail_mem=61.43 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=4 avail_mem=61.43 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=4 avail_mem=61.43 GB): 100%|██████████| 58/58 [00:01<00:00, 37.72it/s]


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
    Generated text:  Lucy White and I am a writer. I have been writing for almost 15 years now. I am a freelance writer who writes for many different publications in the news, books, and magazines. I have been a reporter, journalist, and editor for many years.
    I have been writing since I was 15 years old, and I have written extensively on topics including politics, business, sports, and culture. I have also written for the BBC, The New York Times, The Guardian, and many other publications.
    I have had the pleasure of helping many people to express themselves in their writing. I have helped people to express themselves
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political leader. The president is the head of the executive branch of the U.S. government. He or she serves a two-year term. The president is the commander-in-chief of the armed forces. The president's term of office is the same as the term of a president of the United States. The president is the chairman of the Council of Governance. The president makes decisions about the country. They make decisions about national security, foreign policy, and national security. The president is the commander-in-chief of the armed forces. The president is the chairman of the Council of Governance. The president is the chairman of the Council of Governance.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. A city has 24 apartments and the number of apartments is the same as the number of hours in a day. How many hours is the city hours long?
    
    To determine the number of hours in a day, we need to understand the relationship between the number of apartments and the number of hours in a day. According to the problem, a city has 24 apartments and the number of apartments is the same as the number of hours in a day. This means that if we divide the number of apartments by the number of hours in a day, we should get the number of hours in a day.
    
    Let's denote the
    ===============================
    Prompt: The future of AI is
    Generated text:  a complex and rapidly evolving field with a lot to be excited about. In the past few years, there has been significant growth in the use of AI for various tasks such as image recognition, natural language processing, and even autonomous driving. However, there is also a growing concern about the impact of AI on the environment and social justice.
    
    In this article, we will explore the potential challenges and benefits of AI, and how they can be mitigated or addressed to ensure that AI is used in a sustainable and ethical way. We will also discuss some key ethical considerations that need to be taken into account when developing AI systems.
    
    One of the most


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm passionate about [What I Love About My Profession]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Personality Trait] person who is [What I Like About My Personality]. I'm always ready to learn and grow, and I'm excited to share my knowledge and experience with others. I'm a [What I Aim to Achieve] person who is [What I Aim to Achieve]. I'm a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major financial center and home to many world-renowned museums, art galleries, and restaurants. The city is known for its vibrant nightlife and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city that has played a significant role in French history and continues to be a major cultural and economic center in the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced privacy and security: As AI becomes more prevalent, there will be a growing need for measures to protect the privacy and security of personal data. This may include measures such as data encryption, access controls, and regular audits of AI systems.
    
    3. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a growing need for ethical considerations to be taken into account.
    


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
    Generated text:  [Name], and I'm a [Occupation]! I've always been fascinated by the concept of the unknown, and I'm always eager to uncover the secrets hidden in plain sight. From [list any specific skills or interests that differentiate you] to my desire to solve complex problems with a keen eye for detail, I'm always on the lookout for new opportunities to make a difference. What other aspects of your character do you want to highlight? A brief personal statement or profile is appreciated, but don't hesitate to introduce yourself in a way that captures your personality and interests. Let's see where this story takes us! Hello, my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    This statement provides a clear and concise overview of the capital city of France, which is Paris. To elaborate further, Paris is the largest and most populous city in France, located on the banks of the Seine River. It is the capital of France and is known for its rich history, art, and culture, as well as its iconic landmarks such as the Eiffel Tower and the Louvre Museum. Paris is also home to the Eiffel Tower, the Palace of Versailles, and the Notre-Dame Cathedral, among other historical and cultural sites. With its mix of modern and medieval architecture, Paris is a world
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be a combination of both positive and negative developments, with some areas of focus expected to see significant advancements. Here are some potential trends in AI that could shape the industry in the coming years:
    
    Positive trends:
    
    1. Improved privacy and security: As AI becomes more powerful, there will be greater concerns about its use in surveillance and data collection. This could lead to the development of new privacy protection technologies and more stringent regulations.
    
    2. Increased efficiency and productivity: AI has the potential to automate tasks that were previously done manually, leading to increased efficiency and productivity. This could result in significant cost savings for businesses and organizations.
    
    3. Enhanced


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

    /an

     [

    age

    ]

     year

     old

     [

    gender

    ],

     [

    Name

    ]

     from

     [

    Location

    ].

     I

     have

     a

    /an

     [

    career

    ]

     and

     I

     work

     in

     [

    occupation

    ].

     I

     have

     a

    /an

     [

    occupation

    ]

     and

     I

     work

     [

    at

    ]

     [

    company

    ].

     I

     love

     [

    job

    ],

     and

     I

     enjoy

     [

    job

    ]

     and

     I

     want

     to

     be

     [

    job

    ].

     I

    'm

     [

    job

    ].

     I

    'm

     passionate

     about

     [

    job

    ]

     and

     I

     believe

     [

    job

    ]

     is

     [

    job

    's

     main

     reason

    ].

     I

    'm

     [

    job

    ]

     and

     I

    'm

     [

    job

    's

     advisor

    ].

     I

    'm

     [

    job

    ].

     I

    'm

     always

     [

    job

    ]

     and

     I

    'm

     always

     [

    job

    's

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     the

     third

    -largest

     city

     in

     the

     world

    .

     The

     city

     is

     home

     to

     many

     iconic

     landmarks

     and

     museums

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     It

     is

     also

     a

     popular

     tourist

     destination

     known

     for

     its

     architecture

    ,

     cuisine

    ,

     and

     lively

     atmosphere

    .

     France

    's

     capital

     city

     is

     also

     home

     to

     the

     country

    's

     political

     and

     cultural

     center

    .

     The

     city

     is

     important

     in

     the

     world

     stage

     and

     plays

     an

     important

     role

     in

     the

     country

    's

     economy

     and

     society

    .

     Paris

     has

     a

     rich

     and

     diverse

     culture

     and

     a

     long

     and

     stor

    ied

     history

    ,

     making

     it

     a

     city

     that

     continues

     to

     thrive

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     potential

    ,

     and

     there

     are

     many

     possibilities

     for

     what

     the

     future

     holds

     for

     this

     rapidly

     evolving

     technology

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     ai

     intelligence

    :

     With

     the

     advancements

     in

     machine

     learning

    ,

     AI

     will

     continue

     to

     become

     more

     sophisticated

     and

     intelligent

    .

     AI

     will

     become

     more

     capable

     of

     recognizing

     patterns

     and

     making

     decisions

     that

     were

     once

     difficult

     for

     humans

     to

     accomplish

    .
    


    2

    .

     AI

     in

     healthcare

    :

     AI

     will

     play

     an

     increasingly

     important

     role

     in

     healthcare

    ,

     with

     doctors

     using

     AI

     to

     assist

     in

     diagnosis

     and

     treatment

     planning

    .

     AI

     will

     also

     be

     used

     to

     analyze

     large

     medical

     data

     sets

    ,

     identify

     trends

    ,

     and

     improve

     diagnostic

     procedures

    .
    


    3

    .

     AI

     in

     the

     environment

    :

     AI

    



```python
llm.shutdown()
```
