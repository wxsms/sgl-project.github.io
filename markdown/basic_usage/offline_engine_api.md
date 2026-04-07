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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.31it/s]


    2026-04-07 03:39:42,907 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 03:39:42] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.75it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.75it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.75it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.75it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.75it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.75it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.75it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.75it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.05it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.05it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.05it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.05it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.05it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.05it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.05it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.82it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.82it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.82it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.82it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.82it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.82it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.82it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.82it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.82it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=119.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.31it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.31it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.31it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=119.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.00 GB):  21%|██        | 12/58 [00:00<00:01, 24.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.99 GB):  21%|██        | 12/58 [00:00<00:01, 24.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.99 GB):  21%|██        | 12/58 [00:00<00:01, 24.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.99 GB):  21%|██        | 12/58 [00:00<00:01, 24.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.98 GB):  21%|██        | 12/58 [00:00<00:01, 24.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.86it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.86it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.86it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=118.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.86it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.37it/s]Capturing num tokens (num_tokens=960 avail_mem=118.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.37it/s] Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.37it/s]Capturing num tokens (num_tokens=832 avail_mem=118.66 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.37it/s]Capturing num tokens (num_tokens=768 avail_mem=118.66 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.37it/s]

    Capturing num tokens (num_tokens=768 avail_mem=118.66 GB):  43%|████▎     | 25/58 [00:00<00:01, 28.25it/s]Capturing num tokens (num_tokens=704 avail_mem=118.66 GB):  43%|████▎     | 25/58 [00:00<00:01, 28.25it/s]Capturing num tokens (num_tokens=640 avail_mem=118.65 GB):  43%|████▎     | 25/58 [00:00<00:01, 28.25it/s]Capturing num tokens (num_tokens=576 avail_mem=118.65 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.25it/s]Capturing num tokens (num_tokens=576 avail_mem=118.65 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.83it/s]Capturing num tokens (num_tokens=512 avail_mem=118.64 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.83it/s]Capturing num tokens (num_tokens=480 avail_mem=118.66 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.83it/s]Capturing num tokens (num_tokens=448 avail_mem=118.65 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.83it/s]

    Capturing num tokens (num_tokens=416 avail_mem=118.65 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.83it/s]Capturing num tokens (num_tokens=416 avail_mem=118.65 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.28it/s]Capturing num tokens (num_tokens=384 avail_mem=118.65 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.28it/s]Capturing num tokens (num_tokens=352 avail_mem=118.64 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.28it/s]Capturing num tokens (num_tokens=320 avail_mem=118.64 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.28it/s]Capturing num tokens (num_tokens=288 avail_mem=118.64 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.28it/s]Capturing num tokens (num_tokens=256 avail_mem=118.63 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.28it/s]Capturing num tokens (num_tokens=256 avail_mem=118.63 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=240 avail_mem=118.63 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=224 avail_mem=118.63 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=208 avail_mem=118.62 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.90it/s]

    Capturing num tokens (num_tokens=192 avail_mem=118.62 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=176 avail_mem=118.62 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=176 avail_mem=118.62 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=160 avail_mem=118.61 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=144 avail_mem=118.61 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=128 avail_mem=118.61 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=112 avail_mem=118.60 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=112 avail_mem=118.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.77it/s]Capturing num tokens (num_tokens=96 avail_mem=118.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.77it/s] Capturing num tokens (num_tokens=80 avail_mem=118.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.77it/s]Capturing num tokens (num_tokens=64 avail_mem=118.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.77it/s]

    Capturing num tokens (num_tokens=48 avail_mem=118.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.77it/s]Capturing num tokens (num_tokens=32 avail_mem=118.58 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.77it/s]Capturing num tokens (num_tokens=32 avail_mem=118.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.37it/s]Capturing num tokens (num_tokens=28 avail_mem=118.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.37it/s]Capturing num tokens (num_tokens=24 avail_mem=118.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.37it/s]Capturing num tokens (num_tokens=20 avail_mem=118.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.37it/s]Capturing num tokens (num_tokens=16 avail_mem=118.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.37it/s]Capturing num tokens (num_tokens=12 avail_mem=118.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.37it/s]Capturing num tokens (num_tokens=12 avail_mem=118.57 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.88it/s]Capturing num tokens (num_tokens=8 avail_mem=118.57 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.88it/s] Capturing num tokens (num_tokens=4 avail_mem=118.56 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.88it/s]

    Capturing num tokens (num_tokens=4 avail_mem=118.56 GB): 100%|██████████| 58/58 [00:01<00:00, 32.26it/s]


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
    Generated text:  Ali, and I'm a full-stack developer who works on a team of 13 people. I'm an expert in multiple tech languages including JavaScript, HTML, CSS, and React, along with my favorite frameworks including React, Node.js, and Docker. As I develop software, I'm always looking for ways to improve performance and scalability.
    I'm passionate about building products that are easy to use and enjoyable to use. I'm constantly looking for ways to enhance the user experience of my apps and websites, and I'm always eager to improve my skills and knowledge in my field.
    I'm available for onboarding, mentoring, and collaborating
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person.
    
    No, the president of the United States is an elected official. The president of the United States is a head of state, a member of Congress, and a symbol of the country. The president serves a four-year term, and the party with the majority in Congress can nominate a successor to fill the position. The president is not a person.
    
    Let's analyze the given options:
    
    A) The president is a person.
    
    B) The president is an elected official.
    
    C) The president is a symbol of the country.
    
    D) The president is not a person.
    
    E) The president is not an elected official.
    
    Which option
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. If the population of Paris was 436,000 people in 2012, and this number is expected to grow to 800,000 by 2020, what will be the approximate percentage increase in population?
    
    To find the approximate percentage increase in population, we need to follow these steps:
    
    1. Determine the increase in population.
    2. Calculate the percentage increase based on the initial population.
    
    First, let's find the increase in population. The initial population in 2012 was 436,000 people, and the population is expected
    ===============================
    Prompt: The future of AI is
    Generated text:  still uncertain. But what are the benefits that this technology brings to the field of education? As a part of this, we will look at how AI can revolutionize the way we teach and learn in the future.
    The digital age has transformed the way we live, work, and learn. The technology revolution has changed the way we consume information, communicate, and interact with the world around us. But the technology that is changing the way we learn has not yet been fully explored. While educational technology has been around for a long time, it has been a tool that has been used in education for a long time. But now, with the


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name]: Hello! I'm [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name]: Of course! I'm here to learn more about your career and to get to know you better. What can I expect from our conversation? [Name]: Of course! I'm here to learn more about your career and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Blanche" (The White City). It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, art, and culture, and is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also known for its fashion industry, with many famous fashion designers and boutiques located in the city. Paris is a popular tourist destination, with millions of visitors annually. It is also home to many important institutions, including the French Academy of Sciences and the French National Library.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the development of AI in the coming years:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in various industries, there is a growing need for ethical considerations to be taken into account. This will likely lead to more stringent regulations and guidelines for the development and use of AI.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, such as self-driving cars, smart homes, and virtual assistants
    


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
    Generated text:  [insert character's name]. I am a talented writer and illustrator, with a passion for capturing the essence of everyday life through vivid and memorable characters and settings. I've been working in the creative industry for several years now, and I'm passionate about using art as a tool for personal and professional growth. I've won several awards for my work and am always on the lookout for new opportunities to push my creative boundaries. Thank you for taking the time to meet me! 😊✨
    
    This is a great start, but can you add some more information about your work? I'm really interested in learning more about you and your craft.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the French capital. It is known for its iconic Eiffel Tower and the Notre-Dame Cathedral, which is one of the oldest and most famous in the world. Other famous landmarks include the Louvre Museum and the Palace of Versailles, which was the former residence of the French monarchy. Paris has a rich cultural history dating back to the Roman Empire and is known for its art, cuisine, and fashion. It is a major transportation hub and a popular tourist destination. Paris has a diverse population with a mix of French, Italian, and other European languages. The city is also home to numerous museums, galleries, and theaters
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, with new and exciting trends shaping how the technology will be used and developed. Here are some possible trends that could influence the development of AI in the coming years:
    
    1. Increased focus on ethical AI: The development of AI will be increasingly influenced by ethical considerations, such as privacy, bias, and accountability. The increase in regulation and scrutiny of AI systems will drive innovation and help ensure that AI is used ethically.
    
    2. Advancements in machine learning and deep learning: The field of machine learning is advancing rapidly, and deep learning is becoming increasingly popular for tasks such as image and speech recognition, natural language processing, and predictive


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

     an

     [

    Age

    ],

     [

    Gender

    ],

     [

    H

    ometown

    ],

     [

    Special

    ity

    ]

     professional

    .

     I

     enjoy

     [

    occupation

    -related

     activity

    ],

     [

    interest

    ],

     and

     [

    aff

    ection

    ate

     term

     for

     a

     friend

     or

     partner

    ].

     I

     strive

     to

     be

     [

    amb

    itious

    ,

     modest

    ,

     or

     confident

    ],

     and

     I

     am

     always

     [

    exc

    ited

    ,

     knowledgeable

    ,

     or

     curious

    ].

     How

     can

     someone

     best

     approach

     interacting

     with

     me

    ?

     People

     will

     be

     [

    friendly

    ,

     cautious

    ,

     or

     neutral

    ].

     I

     will

     always

     [

    open

    ,

     reserved

    ,

     or

     empath

    etic

    ].

     How

     would

     you

     describe

     the

     tone

     of

     your

     interactions

    ?

     I

     strive

     to

     maintain

     a

     [

    cal

    m

    ,

     friendly,

     or

     empath

    etic

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Is

     the

     following

     a

     factual

     statement?

     "

    The

     capital

     of

     France is

     Paris

    ."

     Is

     the

     statement

     true

    ?

     Yes

    ,

     the

     statement

     is

     true

    .

     
    


    The

     statement

     "

    The

     capital

     of

     France

     is

     Paris

    "

     is

     accurate

     and

     factual

    ,

     as

     the

     capital

     of

     France

     is

     indeed

     Paris

    .

     This

     is

     a

     well

    -known

     fact

     about

     the

     French

     capital

     city

     and

     is

     widely

     recognized

     and

     accepted

     by

     many

     people

     worldwide

    .

     Paris

     is

     the

     most

     populous

     city

     in

     France

    ,

     with

     a

     population of

     approximately

     

    2

    .

     

    4

     million

     people

    ,

     making

     it

     the

     largest

     city

     in

     Europe

     by

     area

    .

     It

     is

     also

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    ,

     with

     millions

     of

     tourists

     visiting each

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     hardware

     and

     software

    ,

     increased

     demand

     for

     AI

     solutions

    ,

     and

     ongoing

     technological

     advancements

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

     AI

     inclus

    ivity

    :

     As

     more AI

     systems

     become

     more

     accessible

     and

     accessible

     to

     a

     wider

     audience

    ,

     there

     may

     be

     a

     growing

     trend

     towards

     AI

     that

     is

     more

     inclusive

     and

     equitable

    .

     This

     could

     involve

     better

     targeting

     and

     personalized

     AI

     solutions

    ,

     as

     well

     as

     more

     diverse

     and

     representative

     AI

     algorithms

    .
    


    2

    .

     Greater

     emphasis

     on

     ethical

     AI

    :

     As

     concerns

     about

     AI

    's

     potential

     impact

     on

     society

     and

     the

     environment

     continue

     to

     grow

    ,

     there

     may

     be

     a

     greater

     focus

     on

     ethical AI

    .

     This

    



```python
llm.shutdown()
```
