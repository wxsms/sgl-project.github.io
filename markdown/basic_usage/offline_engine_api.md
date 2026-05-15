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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.04it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.03it/s]


    2026-05-15 19:08:56,168 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 19:08:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:05,  7.10it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:05,  7.10it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:05,  7.10it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:05,  7.10it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:05,  7.10it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:05,  7.10it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:04<00:05,  7.10it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:04<00:05,  7.10it/s]

    Compiling num tokens (num_tokens=832):  28%|██▊       | 16/58 [00:04<00:05,  7.10it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:04<00:02, 12.62it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:04<00:02, 12.62it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:04<00:02, 12.62it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:04<00:02, 12.62it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:04<00:02, 12.62it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:04<00:02, 12.62it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:04<00:02, 12.62it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:04<00:02, 12.62it/s]Compiling num tokens (num_tokens=416):  41%|████▏     | 24/58 [00:04<00:02, 12.62it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 19.17it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 19.17it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 19.17it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 19.17it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:01, 19.17it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:04<00:01, 19.17it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:04<00:01, 19.17it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:04<00:01, 19.17it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:04<00:01, 19.17it/s]

    Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:05<00:01, 19.17it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 27.47it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 27.47it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 36.57it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 36.57it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 36.57it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 36.57it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 36.57it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 36.57it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 36.57it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 36.57it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 36.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.14 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   9%|▊         | 5/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.12 GB):   9%|▊         | 5/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.11 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.10 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.10 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.09 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.09 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.08 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.08 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.08 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.06 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.49it/s]Capturing num tokens (num_tokens=960 avail_mem=72.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.49it/s] Capturing num tokens (num_tokens=896 avail_mem=72.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.49it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.06 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.49it/s]Capturing num tokens (num_tokens=832 avail_mem=72.06 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=768 avail_mem=72.06 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=704 avail_mem=72.06 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=640 avail_mem=72.05 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=576 avail_mem=72.05 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=512 avail_mem=72.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=512 avail_mem=72.04 GB):  50%|█████     | 29/58 [00:00<00:00, 42.20it/s]Capturing num tokens (num_tokens=480 avail_mem=72.05 GB):  50%|█████     | 29/58 [00:00<00:00, 42.20it/s]Capturing num tokens (num_tokens=448 avail_mem=72.05 GB):  50%|█████     | 29/58 [00:00<00:00, 42.20it/s]Capturing num tokens (num_tokens=416 avail_mem=72.05 GB):  50%|█████     | 29/58 [00:00<00:00, 42.20it/s]Capturing num tokens (num_tokens=384 avail_mem=72.05 GB):  50%|█████     | 29/58 [00:00<00:00, 42.20it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.04 GB):  50%|█████     | 29/58 [00:00<00:00, 42.20it/s]Capturing num tokens (num_tokens=352 avail_mem=72.04 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=320 avail_mem=72.03 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=288 avail_mem=72.03 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=256 avail_mem=72.03 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=240 avail_mem=72.03 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=224 avail_mem=72.02 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=224 avail_mem=72.02 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=208 avail_mem=72.02 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=192 avail_mem=72.02 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=176 avail_mem=72.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=160 avail_mem=72.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.41it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=144 avail_mem=72.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=128 avail_mem=72.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=112 avail_mem=72.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=96 avail_mem=72.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.16it/s] Capturing num tokens (num_tokens=80 avail_mem=72.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=64 avail_mem=71.99 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=64 avail_mem=71.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=48 avail_mem=71.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=32 avail_mem=71.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=28 avail_mem=71.98 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=24 avail_mem=71.98 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.00it/s]

    Capturing num tokens (num_tokens=20 avail_mem=71.98 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=20 avail_mem=71.98 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.48it/s]Capturing num tokens (num_tokens=16 avail_mem=71.98 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.48it/s]Capturing num tokens (num_tokens=12 avail_mem=71.97 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.48it/s]Capturing num tokens (num_tokens=8 avail_mem=71.97 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.48it/s] Capturing num tokens (num_tokens=4 avail_mem=71.96 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.48it/s]Capturing num tokens (num_tokens=4 avail_mem=71.96 GB): 100%|██████████| 58/58 [00:01<00:00, 40.84it/s]


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
    Generated text:  Sina. I am a 15-year-old girl from Shanghai. I have many new friends in my class. They are all very friendly and kind. We often play games together. One day, my friend suggested that we play a game of chess together. We had a great time. We won the game and everyone was very happy. I was so excited that I wanted to tell my parents and my friends. My parents and my friends were busy at work that day. It was too late for me to tell them. I told my mother and I also told my friends that day. They asked me to do the homework at home
    ===============================
    Prompt: The president of the United States is
    Generated text:  scheduled to travel to Europe next week. To ensure a smooth travel experience, the president wants to ensure that he gets to see different countries and landmarks. To do this, the president decides to visit at least two cities from the United States, and he also wants to visit at least one landmark from each of these cities.
    
    1. The president plans to visit two cities in the United States. What are the possible pairs of cities he could visit?
    2. The president wants to visit at least one landmark from each of the two cities he visited in the first part of his journey. Given the cities he plans to visit in the first part,
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Lyon
    C. Marseille
    D. Geneva
    The capital of France is Paris. Paris is the capital city of France, located on the north bank of the Seine River, in the Île de la Cité (City Island) and the Île de la Cité, and its Île de Provence. It is the largest city in France and the fourth-largest city in the European Union. It is a major cultural and economic center, known for its museums, theaters, opera, and theater. Paris is the birthplace of the French Revolution, the French Revolution, Napoleon Bonaparte
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but these new technologies will have a huge impact on us. If we don't take steps now, we'll be left behind.
    
    That's the message from the CEO of the Start with AI Foundation, who told a packed room at the Annual Conference of the Association for Computing Machinery (ACM) about the need to keep up with the latest AI technologies.
    
    "As we move forward into the next few years, we must be able to identify which technologies we want to pursue and what they are capable of," said the founder and CEO, Elon Musk. "We need to be able to decide what we want to create and what technologies will


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Character] who has a [Favorite Hobby] that I enjoy [Briefly describe your hobby]. I'm a [Type of Character] who has a [Favorite Hobby] that I enjoy [Briefly describe your hobby]. I'm a [Type of Character] who has a [Favorite Hobby] that I enjoy [Briefly describe your hobby]. I'm a [Type of Character] who has a [Favorite Hobby] that I enjoy [Briefly describe your hobby]. I'm a [Type of Character] who has a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is known for its rich history, art, and culture, and is a major tourist destination. Paris is also home to many important institutions such as the French Academy of Sciences and the French Parliament. The city is a major economic center and plays a significant role in French culture and politics. Paris is a popular destination for tourists and locals alike, and is considered one of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to more sophisticated and adaptive AI systems that can learn from human behavior and adapt to new situations.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be increased concerns about privacy and security. There will be a need for more robust privacy and security measures to protect the data and information that is generated and used by AI systems.
    
    3. Greater reliance on AI for decision-making: AI is likely to become more
    


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
    Generated text:  [Name] and I'm a/an [occupation] who has been working in [field of work] for [number] years. My [number] year stint in the field of [field of work] started on [start date] and ended on [end date]. I've always been passionate about [occupations' or fields of work], and I've been dedicated to helping people improve their lives and make the world a better place. What's your profession, interests, and how have you been navigating the challenges of your career? How would you like to assist someone in their journey of self-improvement? Here's an example
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Explanation: Paris is the largest city in France and the second-largest city in the European Union. It is the seat of the French government and the country's cultural, economic, and political capital. The city's main landmarks include the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Luxembourg Gardens. Paris is known for its romantic architecture, cuisine, and wine culture, as well as its many historical and cultural sites. The city is also home to many notable artists, writers, and musicians. It is a UNESCO World Heritage site and one of the most visited cities in the world. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising and it is expected to continue to evolve and improve. Here are some possible future trends in artificial intelligence:
    
    1. Increased automation: AI will continue to automate repetitive tasks, freeing up human beings to work on more complex, creative, and creative jobs.
    
    2. AI for healthcare: AI will play a significant role in healthcare, with more accurate diagnoses, personalized treatment plans, and predictive analytics.
    
    3. AI for personalization: AI will enable better understanding of human behavior and preferences, leading to more personalized and relevant marketing and personalization.
    
    4. AI for decision-making: AI will help make better decisions by analyzing large datasets and providing


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

     first

     name

    ]

     and

     I

    'm

     an

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

     have

     a

     passion

     for

     [

    insert

     hobby

     or

     interest

    ]

     and

     I

     enjoy

     [

    insert

     something

     you

     do

     for

     fun

    ].

     I

    'm

     [

    insert

     age

    ]

     years

     old

    ,

     and

     I

     was

     born

     in

     [

    insert

     birth

    place

    ].

     I

     believe

     in

     [

    insert

     belief

     or

     value

    ]

     and

     I

     believe

     that

     everyone

     has

     a

     unique

     talent

     or

     skill

     that

     can

     be

     used

     to

     make

     the

     world

     a

     better

     place

    .

     If

     you

     had

     the

     chance

     to

     meet

     me

    ,

     what

     would

     you

     want

     to

     talk

     about

    ?

     And

     what

     advice

     would

     you

     give

     to

     someone

     new

     to

     the

     world

     of

     self

    -im

    pro

    vement

    ?

     


    [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     in

     the

     western

     part

     of

     the

     country

     and

     served

     as

     the

     largest

     city

     in

     Europe

     for

     centuries

    .

     It

     is

     home

     to

     many

     iconic

     landmarks

     and

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

    .

     The

     city

     is

     known

     for

     its

     beautiful

     architecture

    ,

     art

    ,

     and

     cuisine

    ,

     and

     is

     also

     home

     to

     important

     institutions

     such

     as

     the

     Lou

    vre

     and

     the

     Paris

     Opera

    .

     Paris

     has

     a

     diverse

     population

     and

     is

     a

     major

     cultural

    ,

     economic

    ,

     and

     political

     hub

     of

     France

    .

     The

     city

     is

     also

     known

     for

     its

     food

     and

     drink

     culture

    ,

     with

     its

     famous

     gastr

    onomy

     and

     wine

     producing

     regions

    .

     Paris

     is

     one

     of

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     currently

     facing

     some

     extreme

     and

     ambitious

     trends

    ,

     including

    :
    


    1

    .

     Enhanced

     AI

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     may

     see

     more

     powerful

     and

     capable

     machines

    ,

     such

     as

     machines

     that

     can

     perform

     complex

     tasks

     such

     as

     logical

     reasoning

    ,

     creativity

    ,

     and

     decision

    -making

    .
    


    2

    .

     Universal

     AI

    :

     Another

     trend

     is

     the

     idea

     of

     AI

     that

     can

     understand

     and

     generate

     human

    -like

     language

    ,

     thought

    ,

     and

     emotions

    ,

     and

     can

     even

     interpret

     and

     respond

     to

     human

     emotions

     and

     intentions

    .
    


    3

    .

     AI

     for

     Health

    :

     There

     is

     a

     growing

     trend

     of

     using

     AI

     to

     improve

     the

     health

     of

     humans

    .

     This

     could

     include

     developing

     AI

    -driven

     diagnostic

     tools

    ,

     personalized

     health

     plans

    ,

     and

     even

     creating

     AI

    -powered

     robots

     to

    



```python
llm.shutdown()
```
