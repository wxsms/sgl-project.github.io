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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.53it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.52it/s]


    2026-05-08 02:49:00,888 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 02:49:00] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.42it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:09<00:03,  9.90it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:09<00:09,  3.18it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:09<00:09,  3.18it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:09<00:09,  3.18it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:09<00:09,  3.18it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:09<00:09,  3.18it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:09<00:09,  3.18it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:09<00:09,  3.18it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:09<00:09,  3.18it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:09<00:09,  3.18it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:09<00:04,  4.96it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:09<00:04,  4.96it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:09<00:04,  4.96it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:09<00:04,  4.96it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:09<00:04,  4.96it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:09<00:04,  4.96it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:09<00:04,  4.96it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:09<00:04,  4.96it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:09<00:04,  4.96it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:09<00:04,  4.96it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:09<00:01,  7.69it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:09<00:01,  7.69it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:09<00:01,  7.69it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:09<00:01,  7.69it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:09<00:01,  7.69it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:09<00:01,  7.69it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:09<00:01,  7.69it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:09<00:01,  7.69it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:09<00:01,  7.69it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:09<00:01,  7.69it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:09<00:00, 11.30it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:09<00:00, 11.30it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:09<00:00, 11.30it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:09<00:00, 11.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  5.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   2%|▏         | 1/58 [00:00<00:08,  6.51it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   2%|▏         | 1/58 [00:00<00:08,  6.51it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   2%|▏         | 1/58 [00:00<00:08,  6.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   2%|▏         | 1/58 [00:00<00:08,  6.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 14.93it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.49it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.49it/s]

    Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.24it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.24it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.24it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.24it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.24it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.24it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.32it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.32it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.32it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.32it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  71%|███████   | 41/58 [00:01<00:00, 44.25it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 44.25it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 44.25it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 44.25it/s]

    Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 44.25it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 44.25it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.26it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.26it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.26it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.26it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.26it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.26it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]

    Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.79it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 37.02it/s]


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
    Generated text:  Kira and I am 20 years old. I am from New York City, but I travel a lot and live in Toronto, Canada. I work at an online bookstore called RedMelee and I enjoy writing about my adventures in the world of books and making connections with people through my writing. My favorite book is "The Giver" by Lois Lowry. I also love reading travel books and spending time in nature. If you are interested in meeting me, please contact me at [insert email address] or [insert phone number]. I hope to meet you soon. Is this description of what I am trying to describe accurate
    ===============================
    Prompt: The president of the United States is
    Generated text:  3 feet tall. The vice president is 1.5 feet tall. If it takes 2 minutes to walk from one president to the other, how many minutes will it take for the vice president to walk from the first floor to the second floor?
    
    To determine how long it will take for the vice president to walk from the first floor to the second floor, we need to follow these steps:
    
    1. Identify the height difference between the vice president and the first president.
    2. Determine how long it takes to walk this height difference.
    
    The height difference between the vice president and the first president is:
    \[ 1.5 \
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and its biggest park is the Arc de Triomphe, built in 1793. Here, we will learn about a park, the Arc de Triomphe, and the story of the famous ship that was wrecked here in 1793, the RMS Titanic.
    
    The Arc de Triomphe is a monumental arch located in the center of Paris. It was built to commemorate the military victories won by the French during the French Revolutionary Wars. The arch has a diameter of 120 meters, and it stands at an elevation of 55 meters above sea level. It was originally built in
    ===============================
    Prompt: The future of AI is
    Generated text:  not yet here, but some companies are working to revolutionize the technology and create a more efficient, data-driven future. Here’s how tech companies are bringing AI to the real world to improve productivity, reduce costs, and make better decisions.
    1. Amazon: Amazon has been a pioneer in AI since the early 2000s and has been a leading player in the data analytics space. Amazon’s AI platform, Amazon Comprehend, is a language translation and speech recognition system that can convert human speech into text, and vice versa. By utilizing its own data, Amazon has developed a sophisticated approach to analyzing and understanding customer behavior


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third-largest city in the world by population. It is also the seat of government, the capital of the French Republic, and the largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its cuisine, fashion, and music. It is a popular tourist destination and a cultural hub in Europe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the development of AI in the coming years:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the impact of AI on society.
    
    2. Development of more advanced models: As AI technology continues to advance, there will be a greater focus on developing more advanced models that can handle complex and diverse data sets
    


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
    Generated text:  [Name], and I'm an AI language model. I'm here to help you with any questions or concerns you may have, whether they are related to technology, science, or anything else. And don't forget to check out my capabilities and personality to see if I could be the best choice for you! Let's connect! 🚀📚✨ #AI #LanguageModeler
    Your response is clear and concise. Can you add some more details about your personality and how you work? I'm really interested in learning more about you! #AI #LanguageModeler
    My personality is friendly and approachable, always ready to help
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A concise factual statement about France’s capital city is that it is the capital of France and one of the world's most famous cities. 
    
    A more concise statement would be: "Paris is the capital of France and one of the most influential cities in the world." 
    
    A more precise statement would be: "Paris is the capital of France and one of the world's most important cities." 
    
    A somewhat less precise statement would be: "Paris is the capital of France and one of the largest cities in the world." 
    
    The second most accurate statement would be: "Paris is the capital of France and one of the most
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and promising, with many potential trends emerging. Here are some of the most promising trends:
    
    1. Personalized AI: As AI continues to improve, we are likely to see an increase in personalized AI solutions. This will involve using algorithms to tailor the AI to the specific needs of each individual user, resulting in more efficient and effective services.
    
    2. Ethical AI: With the potential of AI to make life better, ethical considerations are becoming more important. Governments, businesses, and organizations are beginning to take a more active role in ensuring that AI is used ethically and responsibly.
    
    3. Autonomous AI: Autonomous AI will become more prevalent


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

     __

    ________

    _

     and

     I

    ’m

     a

    (n

    )

     __

    ________

    _.

     
    


    My

     primary

     role

     is

     __

    ________

    _.

     In

     my

     free

     time

    ,

     I

     enjoy

     __

    ________

    _.

     I

    'm

     a

     (

    main

    ly

    )

     __

    ________

    _

     who

     enjoys

     learning

     new

     things

     and

     making

     friends.

     I

    'm

     passionate

     about

     __________

    _.

     I

    'm

     also

     a

     member

     of

     __

    ________

    _.

     I

    've

     been

     a

     member

     of

     __

    ________

    _

     for

     __

    ________

    _

     years

    ,

     and

     I

     can

     attest

     to

     the

     fact

     that

     it

     has

     been

     a

     journey

     of

     __

    ________

    _.

     


    What

     do

     you

     know

     about

     me

    ?

     What

     can

     you

     tell

     me

     about

     your

     character

    ?

     
    


    You

     can

     say

     something

     like

     this

    :


    -

     I

    'm

     an

     __

    ________

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Its

     iconic

     landmarks

     include

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

     Arc

     de

     Tri

    omp

    he

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     cuisine

    ,

     as

     well

     as

     its

     vibrant

     nightlife

     and

     music

     scene

    .

     It

     is

     a

     major

     center

     of

     education

    ,

     science

    ,

     and

     culture

    .

     Paris

     is

     the

     second

    -largest

     city

     in

     France

     by

     population

     and

     is

     considered

     one

     of

     the

     most

     important

     cities

     in

     the

     world

     by

     media

     outlets

    .

     The

     city

     is

     home

     to

     the

     European

     Parliament

     and

     the

     United

     Nations

    .

     Its

     major

     transport

     hub

     is

     the

     Ch

    amps

    -

    É

    lys

    ées

     area

    ,

     which

     includes

     the

     E

    iff

    el

     Tower

     and

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     poised

     to

     continue

     to

     evolve

     and

     change

     rapidly

    ,

     with

     many

     potential

     trends

     in

     the

     years

     ahead

    .

     Here

     are

     some

     of

     the

     most

     promising

     trends

     to

     watch

    :
    


    1

    .

     Improved

     accuracy

     and

     efficiency

    :

     One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     the

     continued

     improvement

     of

     algorithms

     and

     techniques

     to

     enhance

     accuracy

     and

     efficiency

    .

     This

     will

     likely

     lead

     to

     more

     accurate

     predictions

     and

     more

     efficient

     use

     of

     resources

    ,

     which

     could

     lead

     to

     significant

     economic

     and

     societal

     benefits

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     various

     industries

    ,

     including

     healthcare

    ,

     but

     there

     is

     a

     lot

     of

     potential

     for

     growth

     in

     the

     coming

     years

    .

     AI

     could

     be

     used

     to

     improve

     patient

     care

    ,

     reduce

    



```python
llm.shutdown()
```
