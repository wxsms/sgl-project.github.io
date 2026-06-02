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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.57it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.84it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.84it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.84it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.84it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.84it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.84it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.84it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.84it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.84it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]

    Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  8.00it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 12.42it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 12.42it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 12.42it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 12.42it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 12.42it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 12.42it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 12.42it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 12.42it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:05<00:02, 12.42it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:05<00:02, 12.42it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 19.54it/s]

    Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=128):  60%|██████    | 35/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s]Compiling num tokens (num_tokens=12):  78%|███████▊  | 45/58 [00:05<00:00, 28.81it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 39.97it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 39.97it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 39.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 14.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 14.14it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 14.14it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:03, 14.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:03, 14.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   7%|▋         | 4/58 [00:00<00:03, 14.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):  10%|█         | 6/58 [00:00<00:03, 15.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):  10%|█         | 6/58 [00:00<00:03, 15.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  10%|█         | 6/58 [00:00<00:03, 15.61it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  10%|█         | 6/58 [00:00<00:03, 15.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.77it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 31.41it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.08it/s]

    Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.49it/s]Capturing num tokens (num_tokens=512 avail_mem=72.47 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.49it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.49it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.26it/s]

    Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s]

    Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s]Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.70it/s]

    Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.70it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 46.28it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 36.24it/s]


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
    Generated text:  Samantha and I'm a 22 year old English girl. I'm from a big city in the UK called Edinburgh, Scotland. I'm a popular social media influencer. I do many activities to help people on a daily basis. I'm very passionate about making friends with people from all around the world. I help people to have a great time. I help people to find ways to live a happy life, and to share interesting things with them. I have a certain amount of money, I don't have a lot of things. My dream is to travel around the world and visit all of the places I have always wanted to
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she has a lot of responsibilities and duties to do. It is not easy for a president to make decisions and try to do his or her best to serve the country. President Obama has made many decisions throughout his or her life. He made some of his decisions in the mid-2000s. During that time, the world was going through many events. The world was going through a crisis. The world was going through economic recession. There was a major terrorist attack in the United States. He did not allow the United States to be a member of the United Nations. The world was going
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the north of the country. The capital of the United Kingdom is in the south of the country. Which one of the following options correctly ranks the capital of France from the capital of the United Kingdom from north to south?
    A: London, London
    B: Paris, London
    C: London, Paris
    D: Paris, London To determine the correct ranking of the capital cities of France and the United Kingdom from north to south, we need to know the capitals of each country. Let's go through this step by step.
    
    1. Identify the capital cities of France and the United Kingdom.
       - The capital city of France
    ===============================
    Prompt: The future of AI is
    Generated text:  fully dependent on the speed of AI technology. The best AI technology is the one that is able to learn from the data it receives. Unfortunately, in the current world, the majority of AI algorithms are too slow and require a significant amount of time and resources to process. This results in many companies hiring experts who specialize in developing AI algorithms, but they are not always able to generate high-quality results within a given timeframe.
    In order to overcome this problem, there is a new kind of AI called XAI (Natural Language Understanding). XAI is a new field of AI that focuses on the ability of AI algorithms to understand natural language.
    The


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


    Generated text:  Paris, also known as "La Ville de Paris" and the "City of Light". It is the largest city in France and the second-largest city in the European Union. Paris is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. It is also known for its beautiful architecture, vibrant culture, and annual festivals. Paris is a popular tourist destination and a major economic center in France. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its food and wine, as well as its fashion and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into our daily lives, from manufacturing to healthcare. This will lead to increased automation in many industries, as machines can perform tasks that were previously done by humans.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be increased concerns about privacy and security. This will lead to more stringent regulations and standards for AI development and deployment.
    
    3. Greater reliance on AI for decision
    


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
    Generated text:  [Your Name], and I am a [Your Profession or Industry]. I’m currently [Your Age], and I have [Your Relevant Education or Training] degree in [Your Relevant Field of Study or Technology]. I have always been fascinated by [Your Passion or Interest], and I have always wanted to learn more about it. I’m a hard worker, and I enjoy working with people, especially those who are interested in [Your Interests/Subjects/Profession]. I'm excited about the prospect of [Your Long-term Goals/Responsibilities] with [Your Company/Company/Team]. Feel free to ask me any questions, and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the west of the country and is the largest city in the country. It has a population of around 2 million people and is known for its rich history, architecture, and culture. Paris is home to many famous landmarks and attractions, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a popular tourist destination, attracting millions of visitors every year. With its diverse culture, cuisine, and history, Paris is a must-visit destination for those interested in France and the world.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some possible trends that may shape the landscape of the technology:
    
    1. AI will become more personalized: As AI learns to better understand and interpret human behavior and preferences, it may become more personalized, delivering more accurate and relevant results.
    
    2. AI will incorporate more ethical considerations: As more ethical concerns come to the forefront, we may see AI being developed with more ethical guidelines, to ensure that it serves the greater good of society.
    
    3. AI will become more collaborative: With the rise of cloud computing and the internet, AI is likely to become more collaborative, allowing for greater sharing of data and insights among multiple


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

    ]

     and

     I

     am

     a

    /an

     [

    insert

     the

     profession

     or

     title

     of

     your

     character

    ].

     I

    'm

     a

    /an

     [

    insert

     the

     age

     range

     of

     your

     character

    ].

     I

    'm

     a

    /an

     [

    insert

     the

     race

     of

     your

     character

    ].

     I

     was

     born

     in

     [

    insert

     date

     of

     birth

    ].

     I

     grew

     up

     in

     [

    insert

     hometown

    ].

     I

     was

     raised

     by

     my

     [

    insert

     the

     name

     of

     your

     character

    's

     parent

    ]

     or

     [

    insert

     the

     name

     of

     your

     character

    's

     guardian

    ].

     I

     was

     [

    insert

     the

     number

     of

     years

     of

     service

     or

     career

    ]

     years

     old

     when

     I

     met

     [

    insert

     the

     name

     of

     the

     character

     you

     are

     trying

     to

     introduce

     yourself

     to

    ],

     who

     has

     inspired

     me

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     central

     part

     of

     the

     country

     and

     has

     a

     population

     of

     around

     

    2

    .

    1

     million

     people

    .

     It

     is

     the

     capital

     of

     France

     and

     is

     also

     the

     second

    -largest

     city

     in

     the

     country

    .

     Paris

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

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

    ,

     and

     is

     a

     popular

     tourist

     destination

     for

     French

     and

     international

     tourists

     alike

    .

     The

     city

     is

     also

     known

     for

     its

     cuisine

    ,

     art

    ,

     and

     culture

    ,

     which

     have

     made

     it

     a

     significant

     part

     of

     the

     French

     identity

    .

     Paris

     is

     considered

     the

     cultural

     capital

     of

     the

     world

     and

     has

     a

     rich

     history

    ,

     including

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     constantly

     evolving

    ,

     with

     new

     advancements

     being

     made

     every

     day

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

     integration

     of

     AI

     into

     everyday

     life

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     it

     is

     becoming

     more

     and

     more

     integrated

     into

     everyday

     life

    .

     For

     example

    ,

     AI

    -powered

     chat

    bots

     are

     becoming

     more

     prevalent

    ,

     and

     they

     are

     being

     used

     for

     everything

     from

     customer

     service

     to

     HR

     support

    .
    


    2

    .

     Improved

     ethical

     considerations

    :

     As

     AI

     technology

     becomes

     more

     advanced

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     ensuring

     that

     it

     is

     used

     eth

    ically

    .

     This

     will

     involve

     developing

     new

     ethical

     guidelines

     and

     standards

     for

     AI

    ,

     as

     well

     as

     continuing

     to

     educate

     the

     public

     about

     the

     implications

     of

    



```python
llm.shutdown()
```
