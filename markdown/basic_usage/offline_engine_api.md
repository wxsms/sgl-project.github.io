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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.62it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.61it/s]


    2026-05-06 15:40:22,634 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 15:40:22] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.56it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.56it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.56it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.56it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.56it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.56it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.56it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.56it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.56it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.56it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.56it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 32.90it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.90it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.90it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.90it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.11it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  21%|██        | 12/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.16it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.05it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.05it/s]Capturing num tokens (num_tokens=832 avail_mem=76.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.05it/s]Capturing num tokens (num_tokens=768 avail_mem=76.68 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.05it/s]Capturing num tokens (num_tokens=704 avail_mem=76.68 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.05it/s]Capturing num tokens (num_tokens=640 avail_mem=76.68 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.05it/s]Capturing num tokens (num_tokens=640 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.98it/s]Capturing num tokens (num_tokens=576 avail_mem=76.67 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.98it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.17 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.98it/s]Capturing num tokens (num_tokens=480 avail_mem=76.18 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.98it/s]Capturing num tokens (num_tokens=448 avail_mem=76.18 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.98it/s]Capturing num tokens (num_tokens=416 avail_mem=76.18 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.98it/s]Capturing num tokens (num_tokens=416 avail_mem=76.18 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=384 avail_mem=76.08 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=320 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.11it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.00it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.00it/s]

    Capturing num tokens (num_tokens=224 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.00it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.00it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.00it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.00it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=160 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.08it/s] Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 44.51it/s]

    Capturing num tokens (num_tokens=64 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  81%|████████  | 47/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s] Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.43it/s]Capturing num tokens (num_tokens=4 avail_mem=75.93 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.43it/s]

    Capturing num tokens (num_tokens=4 avail_mem=75.93 GB): 100%|██████████| 58/58 [00:01<00:00, 39.43it/s]


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
    Generated text:  Maxine. I'm a 15-year-old girl who has always dreamed of visiting the moon, but I was always in a dilemma about whether I should go to space or not. I know that going to space would be a lot of work and you wouldn't get to see the world like I do, but I've always wanted to go there. Can you tell me about your childhood and how you became interested in space exploration? 
    
    I'm looking for a topic for my essay that will show why space exploration is important and how it could potentially benefit society. I want to use my experience as a 15-year-old girl
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. He will be re-elected if he has at least 50% of the popular vote. He received 245 electoral votes, and there were 149 candidates running for president. However, for every 2 votes required to win, the winning candidate receives 2.5% more votes than the second-place candidate. What is the minimum number of votes the candidate who finished in second place needs to win?
    
    To determine the minimum number of votes the candidate who finished in second place needs to win, we start by understanding the electoral system and the given conditions.
    
    First, we calculate the total
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the seat of the French government and the country’s capital. It is located on the River Seine in the Paris Basin. It was founded in 843 AD. There are many churches in the city.
    Most of Paris was destroyed during the French Revolution. In 1793, the Bastille was stormed and captured by the royalists. It was stormed again on May 14, 1799 and was stormed in 1870 by French troops and captured by the French General Charles de Gaulle. The cathedral of Notre Dame de Paris, the only remaining medieval building in
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but so is the danger it brings. The fates of the 10 most influential companies are now in the hands of the very same people who control the 5, 000 largest banks in the world. And these are the people who are responsible for the 500 largest patents, the 10,000 most expensive AI research papers, and the 10, 000 most dangerous patents. The first article in this series will examine the dynamics of the race to control AI patents, and how this race may affect innovation, economic growth, and our health and safety. The


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm passionate about [What I Love to Do] and I'm always looking for ways to [What I Want to Improve]. I'm [What I Want to Do] and I'm always looking for ways to [What I Want to Improve]. I'm [What I Want to Do] and I'm always looking for ways to [What I Want to Improve]. I'm [What I Want to Do] and I'm always looking for ways
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of many famous French artists, writers, and composers. Paris is a bustling metropolis with a rich cultural heritage and a diverse population of over 2 million people. The city is known for its fashion, art, and cuisine, and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant atmosphere. The city is also home to many international organizations and institutions, including the French Academy of Sciences
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency, cost savings, and job displacement for some workers.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more personalized and accurate healthcare solutions.
    
    3. AI-powered
    


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
    Generated text:  Sarah and I'm a friendly and kind-hearted person who is always looking for new ways to help others. I'm confident and always ready to lend a helping hand, and I'm passionate about being a mentor and providing guidance to those who need it the most. My goal is to inspire others to reach their full potential and achieve their dreams, and I believe that education and personal development are the keys to success. I'm always seeking opportunities to share my knowledge and experience, and I'm excited to be able to help someone in any way I can. Thank you for considering me as a potential mentor. How can I best engage with you?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital and largest city of France. It is a historical and cultural center of the country. Paris is known for its art, architecture, fashion, and cuisine. It is also known as the "City of Light" and the "City of Lights" due to the large number of nightclubs and nightlife in the city. Paris is the third largest metropolitan area in the world. It is also the sixth most populous city in the world. 
    
    Paris is located on the River Seine, which separates the city of Paris from the rest of France. The city is home to the Eiffel Tower, the Lou
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of different trends, including:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into various aspects of society, there will be increased scrutiny of its potential negative impacts on human well-being. This will lead to more rigorous ethical standards and regulatory frameworks for AI development and deployment.
    
    2. Greater integration with other technologies: As AI becomes more integrated with other technologies like robotics, machine learning, and cloud computing, there will be increased potential for AI to expand its capabilities and reach new markets. This could lead to new business models and new industries that rely on AI-powered technologies.
    
    3. Larger deployment of AI:


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

     am

     a

     [

    job

     title

    ]

     who

     specializes

     in

     [

    job

     title

    ]

     roles

    .

     I

     bring

     a

     wealth

     of

     experience

     in

     [

    mention

     your

     expertise

     or

     area

     of

     focus

    ].

     My

     passion

     is

     [

    mention

     what

     you

     enjoy

     most

     about

     your

     job

    ],

     and

     I

     thrive

     in

     the

     dynamic

     and

     fast

    -paced

     environment

     of

     [

    mention

     your

     company

     or

     industry

    ].

     I

     am

     a

     team

     player

     who

     is

     always

     looking

     to

     learn

     and

     grow

    ,

     and

     I

     am

     always

     on

     the

     lookout

     for

     opportunities

     to

     contribute

     to

     our

     company

    's

     success

    .

     I

     am

     a

     [

    mention

     what

     you

     believe

     makes

     you

     a

     good

     fit

     for

     the

     role

    ]

     and

     I

     am

     excited

     to

     work

     with

     you

     to

     achieve

     your

     goals

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

     and

     is

     known

     for

     its

     rich

     history

     and

     bustling

     urban

     life

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

    :
    


    1

    .

     Growth

     in

     the

     number

     of

     AI

     algorithms

     and

     models

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

     continued

     increase

     in

     the

     number

     of

     algorithms

     and

     models

     that

     can

     be

     trained

     and

     deployed

    .

     This

     will

     require

     more

     resources

     and

     investment

     in

     research

     and

     development

    .
    


    2

    .

     Increased

     focus

     on

     ethical

     AI

    :

     With

     the

     increasing

     awareness

     of

     the

     potential

     risks

     of

     AI

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     ethical

     AI

     practices

    .

     This

     will

     include

     considerations

     of

     fairness

    ,

     transparency

    ,

     accountability

    ,

     and

     the

     responsible

     development

     of

     AI

     systems

    .
    


    3

    .

     Rise

     of

     multi

    -

    architecture

     AI

    :

     The

     development

     of

     multi

    -

    architecture

     AI

     is

    



```python
llm.shutdown()
```
