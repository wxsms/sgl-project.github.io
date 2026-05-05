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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.19it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.18it/s]


    2026-05-05 11:31:42,283 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 11:31:42] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.27it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.26it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.26it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.26it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.26it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.26it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.26it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.26it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.26it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.26it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.26it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.26it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.45it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.45it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.45it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.45it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.45it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.45it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 23.45it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 23.45it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.45it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.45it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.45it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.61it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.33it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.77 GB):   9%|▊         | 5/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.76 GB):   9%|▊         | 5/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.75 GB):   9%|▊         | 5/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.75 GB):   9%|▊         | 5/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.75 GB):   9%|▊         | 5/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.93it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.31 GB):  22%|██▏       | 13/58 [00:00<00:01, 25.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 25.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 25.42it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 25.42it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  22%|██▏       | 13/58 [00:00<00:01, 25.42it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  22%|██▏       | 13/58 [00:00<00:01, 25.42it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  31%|███       | 18/58 [00:00<00:01, 28.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.29 GB):  31%|███       | 18/58 [00:00<00:01, 28.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 28.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  31%|███       | 18/58 [00:00<00:01, 28.11it/s]Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 28.11it/s] Capturing num tokens (num_tokens=896 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 28.11it/s]Capturing num tokens (num_tokens=896 avail_mem=72.28 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=704 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.38it/s]

    Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:00<00:00, 32.92it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:00<00:00, 32.92it/s]Capturing num tokens (num_tokens=512 avail_mem=72.25 GB):  47%|████▋     | 27/58 [00:00<00:00, 32.92it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:00<00:00, 32.92it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:00<00:00, 32.92it/s]Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.92it/s]Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.18it/s]Capturing num tokens (num_tokens=384 avail_mem=72.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.18it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.18it/s]

    Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.18it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.18it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 33.50it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 33.50it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 33.50it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 33.50it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 33.50it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 33.50it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  71%|███████   | 41/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 37.54it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.83it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.83it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.83it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.83it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.83it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.08it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.08it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.08it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.08it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.08it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 30.88it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 30.88it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 30.88it/s]

    Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 30.88it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 30.88it/s] Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  98%|█████████▊| 57/58 [00:01<00:00, 27.65it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  98%|█████████▊| 57/58 [00:01<00:00, 27.65it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 30.23it/s]


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
    Generated text:  Changye. This is my first day at New York University, a university I have never attended before. I will be studying an MBA in business administration. I will major in the management of a corporation, and I will also study the management of a healthcare organization. I am very enthusiastic and optimistic. I believe in myself and my abilities, and I believe that I can make my dreams come true. I am looking forward to my first day here. What did you find out about the university? When did you decide to start your first day? What was your experience at the university? I will thank you for your time. Goodbye.
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a new term in 2021. The current president is Joe Biden, and the previous president was Donald Trump. If there are 41 other candidates running for president in the election, in how many ways can the president be re-elected?
    To determine the number of ways the president can be re-elected, we need to consider that the president is being re-elected twice, once for the first term and once for the second term. This means we need to calculate the number of ways to choose a president for the first term and then subtract the number of ways to choose a president for the second term.
    
    1. **Calculate
    ===============================
    Prompt: The capital of France is
    Generated text:  the capital of the country, not the country. While France is a country, its capital is Paris.
    
    Paris is a city in France, the country. Paris is the capital of France.
    
    There are no capital cities in the world other than the United States of America, Mexico, and New Zealand. The United States, Mexico, and New Zealand are not capital cities.
    
    Davidsen is the capital city of Denmark. Davidsen is not the capital city of Denmark.
    
    The United Kingdom is the capital of the United Kingdom, not the United States of America.
    
    1. Is Paris the capital of France?
    2. Is Davids
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s a far more transformative and beneficial force than we ever imagined. AI is the next big thing, and the world is currently on the cusp of a new era of innovation. From the future of manufacturing and healthcare to customer service and transportation, AI is taking the world by storm.
    AI, or Artificial Intelligence, is a powerful technology that is transforming the way we work and live. It’s used to automate processes, improve efficiency, and provide insights that were once impossible to obtain. AI is a rapidly evolving field, and there are many exciting opportunities for those who want to use it.
    One of the most promising areas


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm passionate about [What I Love to Do], and I'm always looking for ways to [What I Want to Improve]. I'm a [What I Do Best], and I'm always ready to [What I Can Do to Help]. I'm [What I'm Looking for in a Job], and I'm always eager to [What I Want to Learn]. I'm [What I'm Looking for in a Job], and I'm always eager to [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is also the seat of the French government and the country's cultural and political capital. Paris is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It is also home to many famous museums, including the Musée d'Orsay and the Musée d'Orsay. Paris is a popular tourist destination and is known for its rich history, art, and cuisine. It is a major economic and financial center in Europe and plays a significant role in the country's
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical AI: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations, such as privacy, fairness, and accountability. This could lead to more robust AI systems that are designed to be transparent, accountable, and responsible.
    
    3. Increased use of AI in healthcare: AI is already
    


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
    Generated text:  [Name], and I am a [age] year-old. I have always been passionate about writing and creativity, and I love spending time outdoors. I enjoy taking photos, creating art, and exploring new cultures. I believe in being true to myself and my own identity, and I am proud to call myself [Name]! 🌍✨
    
    Feel free to add any additional information or context you feel is relevant to the character's personality or interests. Let me know if you have any other questions about me or the character. 👨‍💼🎨
    
    I'm really interested in knowing more about you! Can you tell me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris, the iconic capital city of France, is renowned for its stunning architecture, vibrant culture, and historical significance, making it a beacon of modernity and historical preservation in the heart of the world's largest city. The city has a rich cultural heritage with numerous iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, each housing unique and fascinating historical and artistic treasures. Paris is also home to some of the world's most renowned museums, including the Musée d'Orsay, the Musée Rodin, and the Musée de l'Orangerie, all of which
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and varied, with several trends that are poised to shape the technology and its applications. Here are some potential areas of focus:
    
    1. Increased focus on ethical AI: As the tech industry becomes more aware of the potential consequences of AI development, there is a push to ensure that AI is developed and used ethically. This includes considerations of privacy, bias, and the impact on society. Companies and governments will need to work together to develop guidelines and standards for AI that promote ethical practices.
    
    2. Integration of AI with other technologies: AI is already being used in a wide range of applications, from self-driving cars to personalized education. As


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

    Your

     Name

    ],

     and

     I

    'm

     an

     experienced

     [

    specific

     occupation

     or

     field

    ]

     professional

    .

     I

     have

     a

     passion

     for

     helping

     people

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     more

     about

     the

     world

     around

     me

    .

     Whether

     it

    's

     through

     reading

    ,

     watching

    ,

     or

     just

     hanging

     out

     with

     people

     who

     share

     my

     interests

    ,

     I

     enjoy

     exploring

     new

     things

     and

     staying

     curious

    .

     I

    'm

     always

     looking

     for

     opportunities

     to

     expand

     my

     skills

     and

     knowledge

    ,

     and

     I

    'm

     excited

     to

     meet

     new

     people

     and

     share

     my

     experiences

     with

     them

    .

     Thank

     you

     for

     taking

     the

     time

     to

     learn

     about

     me

    .

     Let

    's

     connect

    !

     

    🛋

    ️

    🔍

    
    


    Hey

     there

    ,

     [

    Name

     of

     person

     you

    're

     writing

     about

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     most

     populous

     city

     in

     France

     and

     the

     second

     most

     populous

     city

     in

     the

     European

     Union

     after

     Rome

    .

     The

     city

     is

     home

     to

     many

     of

     France

    ’s

     most

     famous

     landmarks

     and

     attractions

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Vol

    taire

     House

    .

     The

     city

     is

     also

     known

     for

     its

     gastr

    onomy

    ,

     particularly

     its

     seafood

     and

     its

     cuisine

    ,

     which

     is

     heavily

     influenced

     by

     French

     culture

     and

     traditions

    .

     Paris

     is

     a

     major

     cultural

     and

     economic

     center

     and

     is

     often

     referred

     to

     as

     the

     "

    Paris

     of

     the

     World

    ."

     The

     French

     capital

     is

     located

     in

     the

     Î

    le

    -de

    -F

    rance

     region

    ,

     which

     includes

     the

     cities

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     constantly

     evolving

    ,

     with

     new

     breakthrough

    s

     and

     applications

     emerging

     at

     a

     rapid

     pace

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Enhanced

     Intelligence

    :

     As

     AI

     continues

     to

     improve

     in

     its

     ability

     to

     process

     and

     understand

     complex

     data

    ,

     its

     intelligence

     will

     continue

     to

     grow

     and

     evolve

    .

     This

     will

     lead

     to

     more

     intelligent

    ,

     self

    -aware

    ,

     and

     adaptable

     machines

     that

     can

     handle

     a

     wider

     range

     of

     tasks

    .
    


    2

    .

     Autonomous

     Systems

    :

     Autonomous

     systems

     are

     becoming

     increasingly

     common

     as

     AI

     technology

     improves

    .

     These

     systems

     are

     designed

     to

     operate

     without

     human

     intervention

    ,

     and

     can

     make

     decisions

     based

     on

     complex

     data

     analysis

    .
    


    3

    .

     Virtual

     and

     Aug

    mented

     Reality

    :

     AI

     is

     also

     becoming

     more

     advanced

    



```python
llm.shutdown()
```
