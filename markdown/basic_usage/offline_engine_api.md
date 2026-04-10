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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.49it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.47it/s]


    2026-04-10 05:40:12,665 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 05:40:12] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.94it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.94it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.94it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.94it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.94it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.94it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.94it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.94it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.99it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.99it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.99it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.99it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.99it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.99it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.99it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.99it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.98it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.98it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.98it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.98it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.98it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.98it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.98it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.26it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.26it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.26it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.26it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.26it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.26it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.26it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.26it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.53it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.53it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.53it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  21%|██        | 12/58 [00:00<00:01, 28.55it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.55it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.55it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.55it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.60 GB):  21%|██        | 12/58 [00:00<00:01, 28.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.10it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.10it/s]

    Capturing num tokens (num_tokens=960 avail_mem=131.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.10it/s] Capturing num tokens (num_tokens=960 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.35it/s]Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.35it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.35it/s]Capturing num tokens (num_tokens=768 avail_mem=131.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.35it/s]Capturing num tokens (num_tokens=704 avail_mem=131.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.35it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.35it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=480 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.55it/s]

    Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.37it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.37it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.37it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.37it/s]Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.37it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.37it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=224 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.10it/s]

    Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.12it/s]Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.12it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.12it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.12it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.12it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.12it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  81%|████████  | 47/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  81%|████████  | 47/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=64 avail_mem=131.51 GB):  81%|████████  | 47/58 [00:01<00:00, 37.68it/s]

    Capturing num tokens (num_tokens=48 avail_mem=131.51 GB):  81%|████████  | 47/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  81%|████████  | 47/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  88%|████████▊ | 51/58 [00:01<00:00, 24.60it/s]Capturing num tokens (num_tokens=28 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 24.60it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 24.60it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 24.60it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 24.60it/s]

    Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  95%|█████████▍| 55/58 [00:01<00:00, 26.18it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  95%|█████████▍| 55/58 [00:01<00:00, 26.18it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  95%|█████████▍| 55/58 [00:01<00:00, 26.18it/s] Capturing num tokens (num_tokens=4 avail_mem=131.48 GB):  95%|█████████▍| 55/58 [00:01<00:00, 26.18it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 31.93it/s]


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
    Generated text:  Charlie and I am a chemistry teacher from South Africa. And I was born in Johannesburg, South Africa and I have always been fascinated by chemistry and I decided to pursue a career in teaching, particularly chemistry. I did not go to university, I went to school where I excelled in science and I started teaching when I was just a teenager. I knew that I was very passionate about chemistry and I want to share this love with all the people who are curious to know about the wonders that chemistry is. Chemistry is a very fascinating and interesting subject, and it is one of the many subjects that are taught in schools around the world. It
    ===============================
    Prompt: The president of the United States is
    Generated text:  scheduled to visit the United Kingdom for 20 days. If he takes a 4-day vacation each week, how many weeks will it take for him to complete his 20-day visit? To determine how many weeks it will take for the president of the United States to complete his 20-day visit to the United Kingdom, given that he takes a 4-day vacation each week, we can follow these steps:
    
    1. Calculate the total number of days he will spend in the United Kingdom without considering the vacation period.
    2. Determine how many weeks it will take to cover the remaining days after accounting for the vacation.
    
    First
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It's located in the south of the country. It's about 325 miles (525 kilometers) from the Atlantic Ocean. It's surrounded by the North Sea, the English Channel and the Mediterranean Sea. Paris is the most populous city in France. The city has a population of over 2 million people. The city has the population of the world's third largest city. 
    What is the population of Paris? The population of Paris is over 2 million people. The city has a population of over 2 million people. The city has the population of the world's third largest city. 
    
    So,
    ===============================
    Prompt: The future of AI is
    Generated text:  already here. The technology has already been in use for years and the cost of the technology is dropping. But is it all good or does it have a lot of potential drawbacks? AI has been a product of our society for a long time now. It is already being used to make our lives easier and faster, and it is a part of our daily lives. But now, we are starting to look at its future. It is a technology that will continue to develop and evolve, and we can expect to see even more advanced and powerful AI in the future.
    There are several potential benefits of AI, including the ability to automate tasks,


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Vehicle] [Vehicle Name]. I have been driving for [Number of Years] years and have [Number of Miles] miles driven. I am a [Favorite Hobby] [Hobby Name]. I have a [Favorite Color] [Favorite Food] [Favorite Book] [Favorite Movie]. I am [Favorite Quote] [Quote]. I am [Favorite Quote] [Quote]. I am [Favorite Quote] [Quote]. I am [Favorite Quote] [Quote]. I am [Favorite Quote] [Quote]. I am [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also a major financial center and a major tourist destination. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a popular destination for tourists and locals alike, and is considered one of the most beautiful cities in the world. It is also a major center for the arts and culture, with many museums, theaters, and galleries. Paris is a city that is constantly evolving and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could emerge in the coming years:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more sophisticated, we can expect to see even more widespread use of AI in healthcare, with more personalized and accurate diagnoses and treatments.
    
    2. AI in finance: AI is already being used in finance to improve risk management and fraud detection. As AI technology continues to improve, we can expect to see even more widespread
    


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
    Generated text:  John. I'm a successful businessman with a passion for entrepreneurship. I love exploring new opportunities and learning from my mistakes. I have a strong work ethic and adept at problem-solving, and I thrive in fast-paced environments. I love meeting new people and helping them achieve their dreams. I'm excited to share my experiences and learn from you. [Your answer should be brief and informative, staying neutral and objective] John, a successful businessman with a passion for entrepreneurship, loves exploring new opportunities, learning from mistakes, and is a fast-paced person with a strong work ethic and problem-solving skills. John is a passionate and eager entrepreneur with a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city in the country.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  anticipated to be marked by continued technological advancement, with more complex algorithms, machine learning, and artificial neural networks emerging. Some of the key trends in AI include:
    
    1. Increased automation: AI is likely to become more autonomous and self-directed, with the ability to perform tasks with greater ease and efficiency. This could lead to the development of new applications such as autonomous vehicles, robotic surgery, and predictive maintenance.
    
    2. Integration with existing technologies: AI is likely to become more integrated with other technologies, such as sensors, sensors, and data storage. This could lead to the development of more sophisticated and robust AI systems that can operate effectively in the


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

     writer

     and

     screen

    writer

     with

     experience

     in

     [

    fill

     in

     any

     relevant

     details

     here

    ].

     My

     interests

     include

     [

    mention

     any

     specific

     hobbies

     or

     passions

    ],

     and

     I

     love

     [

    mention

     a

     particular

     genre

     of

     writing

     or

     storytelling

    ].


    Ex

    perienced

    ,

     versatile

    ,

     and

     with

     a

     keen

     eye

     for

     detail

    ,

     I

     craft

     content

     that

     reson

    ates

     with

     readers

     across

     all

     ages

     and

     backgrounds

    .

     My

     work

     has

     been

     featured

     in

     numerous

     literary

     magazines

     and

     on

     major

     streaming

     platforms

    .

     I

    'm

     always

     looking

     for

     ways

     to

     push

     the

     boundaries

     of

     storytelling

     and

     bring

     new

     narratives

     to

     life

    .

     Can

     you

     please

     tell

     me

     more

     about

     your

     genre

     of

     writing

     or

     storytelling

    ?

     I

    'm

     really

     interested

     in

     exploring

     more

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     and

     the

     largest

     metropolitan

     area

     in

     the

     European

     Union

     by

     population

    ,

     and

     the

     second

     most

     populous

     city

     in

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

     a

     major

     cultural

    ,

     economic

    ,

     and

     political

     center

    .

     It

     is

     also

     known

     for

     its

     unique

     art

     and

     architecture

    ,

     including

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

    .

     The

     city

     is

     also

     home

     to

     many

     historic

     and

     architectural

     landmarks

    ,

     including

     the

     Ch

    amps

    -

    É

    lys

    ées

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

     Paris

     is

     often

     described

     as

     the

     "

    City

     of

     Love

    "

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     advancements

     and

     significant

     changes

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

     automation

     and

     robotics

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

     more

     automation

     and

     robotics

     to

     become

     more

     prevalent

    .

     This

     could

     lead

     to

     more

     efficient

     work

     processes

     and

     potentially

     reduce

     human

     error

    .
    


    2

    .

     Increased

     transparency

     and

     accountability

    :

     AI

     systems

     will

     become

     more

     transparent

     and

     accountable

    ,

     with

     greater

     emphasis

     on

     explaining

     their

     decisions

     and

     outputs

    .

     This

     will

     help

     to

     create

     a

     more

     trust

    -based

     approach

     to

     AI

     use

    .
    


    3

    .

     AI

     for

     healthcare

     and

     personalized

     medicine

    :

     AI

     will

     be

     used

     in

     healthcare

     to

     help

     diagnose

     and

     treat

     diseases

     more

     accurately

    ,

     with

     potential

     benefits

     for

    



```python
llm.shutdown()
```
