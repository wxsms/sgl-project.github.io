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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.76it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.75it/s]


    2026-04-14 15:13:44,026 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 15:13:44] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.09it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.09it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.09it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.09it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.09it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  6.09it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.09it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.09it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.09it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.31it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.31it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.31it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.31it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.31it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.31it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.31it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.31it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.80it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.55it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.55it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.55it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.55it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.55it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.55it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.55it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.58it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.44it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.44it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]

    Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:00<00:01, 23.97it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:00<00:01, 23.97it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:00<00:01, 23.97it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:00<00:01, 23.97it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  43%|████▎     | 25/58 [00:01<00:01, 23.97it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 23.97it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.88it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.88it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.88it/s]

    Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.88it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.88it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.88it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  60%|██████    | 35/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  60%|██████    | 35/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.88it/s]

    Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.17it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.17it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.17it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.17it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.17it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.17it/s]

    Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.07it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.07it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.07it/s] Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.07it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 34.35it/s]


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
    Generated text:  John. I am the CEO of a company that manufactures and sells high-tech medical devices. I am a technology enthusiast and I use the term "the long tail" to describe the phenomenon where a small amount of product sells for a large sum. I would like to share my beliefs about the long tail with you.
    
    1. The long tail is a metaphor for technology that involves the creation of many small products that perform a specialized function or serve a specific purpose. Each small product is unique and valuable, but as a whole, they can be purchased for significant amounts of money. For example, Apple's iPhone and MacBook computers are examples of a
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 35 years older than the president of Brazil, and the president of Brazil is 25 years younger than the president of France. If the president of France is currently 35 years old, how old will the president of Brazil be in 10 years?
    To solve the problem, we need to determine the current ages of the presidents of the United States, Brazil, and France, and then calculate their ages in 10 years.
    
    1. **Determine the current age of the president of the United States:**
       - The president of the United States is currently 35 years older than the president of
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the ____.
    A. Alps
    B. Pyrenees
    C. Alps
    D. Pyrenees
    Answer:
    
    C
    
    The capital of France is located in the ____.
    A. Alps
    B. Pyrenees
    C. Alps
    D. Pyrenees
    Answer:
    
    C
    
    Which of the following is NOT a manifestation of the conflict between modern democracy and tradition in China? A. Legalism B. Confucianism C. Confucianism D. Legalism
    Answer:
    
    A
    
    According to the first paragraph, which of the following is NOT true? A. The theme of the film
    ===============================
    Prompt: The future of AI is
    Generated text:  on the horizon and many different companies are investing in AI systems. The major companies like Amazon, Google, Facebook and Microsoft are the biggest champions of AI and are working with other companies to create amazing AI tools. However, it is difficult to keep track of all the companies and their intentions. In this post, we will take a closer look at how AI and data science are impacting businesses today.
    The main purpose of AI is to help businesses understand their customers and predict future trends. Data science and machine learning are helping companies create accurate models and give them the ability to make data-driven decisions. In this article, we will discuss how AI and


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm [Gender] and [Age] years old. I'm a [Skill] with [Number] years of experience in [Field]. I'm [Gender] and [Age] years old. I'm a [Skill] with [Number] years of experience in [Field]. I'm [Gender] and [Age] years old. I'm a [Skill] with [Number] years of experience in [Field]. I'm [Gender]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in Europe by population. It is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its rich cultural heritage, including its art museums, theaters, and music venues. The city is a major center for business, finance, and tourism, and is a popular tourist destination. Paris is a vibrant and diverse city with a rich history and a strong sense of French identity. Its status as the capital of France is recognized
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is expected to become more prevalent in various industries, with automation becoming more prevalent in manufacturing, transportation, and customer service. This will lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI will become more integrated with other technologies: AI will become more integrated with other technologies, such as the Internet of Things (IoT), blockchain, and quantum computing. This will create new opportunities for AI to be used in new ways, such as in healthcare, finance, and energy.
    
    3. AI will become more ethical and
    


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
    Generated text:  [Name] and I'm a [Field of Work] with [Years of Experience]. I have a passion for [Job Title] and enjoy [Reason for Job]. I'm always looking for ways to [Opportunity] and what more could I do? If you're interested, feel free to reach out! 
    [Name] is a (Field of Work) with  (Years of Experience), and my primary role is to [Your Job Title] and I'm passionate about [Reason for Job]. I thrive on [Opportunity] and have a unique skill set that allows me to [Opportunity]. If you're interested in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the region of Paris. It is the largest and most populous city in France and is the heart of the country’s culture, politics, and economy. Paris is famous for its iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral, as well as its rich history dating back over 2,000 years. The city also has a diverse population of around 2.7 million residents, with many of them speaking French, as well as other languages like Italian, German, and Arabic. Paris is a major center of art, music, and literature, and plays an important role in European and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly dynamic and driven by emerging technologies and trends. Here are some possible trends in AI that may emerge in the coming years:
    
    1. Increased Integration of AI into Other Fields: AI is already being integrated into various fields, including healthcare, finance, manufacturing, and transportation. We may see even greater integration in the future, with AI becoming more integrated into other industries as well.
    
    2. Advances in Natural Language Processing: The ability to process natural language will allow AI to understand and interact with humans in a more natural way. This could lead to more efficient and effective communication, as well as more accurate and personalized information sharing.
    
    3


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

     [

    Age

    ]

     year

     old

    .

     I

    'm

     [

    Gender

    ]

     and

     I

    'm

     a

     [

    Occup

    ation

    ].

     I

     love

     [

    Why

     it

    's

     important

     to

     you

    ].

     I

    'm

     passionate

     about

     [

    My

     hobby

     or

     interest

    ].

     What

     about

     you

    ?

     Do

     you

     have

     a

     hobby

     or

     interest

     that

     you

    're

     passionate

     about

    ?

     What

     does

     it

     mean

     to

     you

    ?

     How

     do

     you

     spend

     your

     free

     time

    ?

     Lastly

    ,

     what

    's

     your

     dream

     job

    ,

     and

     how

     do

     you

     think

     you

    'll

     achieve

     it

    ?

     Let

     me

     know

    ,

     and

     I

    'll

     be

     here

     to

     chat

    .

     [

    Name

    ]

     [

    Phone

     number

    ]

     [

    Email

     address

    ]

     [

    Address

    ]

     [

    City

    ,

     State

    ,

     Zip

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    ,

     and

     is

     a

     cosm

    opolitan

     and

     diverse

     city

    .

     It

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

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     Roman

     times

    .

     The

     city

     is

     known

     for

     its

     museums

    ,

     art

     galleries

    ,

     and

     fashion

     industry

    ,

     and

     is

     home

     to

     several

     major

     educational

     and

     research

     institutions

    .

     The

     city

     is

     also

     home

     to

     the

     iconic

     Lou

    vre

     Museum

    ,

     which

     features

     over

     

    6

    ,

    0

    0

    0

     works

     of

     art

     from

     around

     the

     world

    .

     Overall

    ,

     Paris

     is

     a

     vibrant

     and

     vibrant

     city

     that

     is

     a

     melting

     pot

     of

     cultures

     and

     ideas

    .

     What

     is

     your

     favorite

     part

     of

     the

     French

     capital

    ?

     As

     an

     AI

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     trends

     that

     are

     expected

     to

     shape

     its

     development

     and

     impact

    .

     Some

     of

     the

     possible

     future

     trends

     in

     artificial

     intelligence

     include

    :
    


    1

    .

     Increased

     sophistication

     and

     diversity

     of

     AI

    :

     As

     more

     data

     and

     resources

     become

     available

    ,

     AI

     systems

     are

     likely

     to

     become

     more

     sophisticated

     and

     diverse

    .

     This

     means

     that

     AI

     will

     be

     able

     to

     learn

     from

     more

     complex

     examples

     and

     make

     better

     predictions

    .
    


    2

    .

     Enhanced

     machine

     learning

     and

     computer

     vision

    :

     Machine

     learning

     will

     be

     more

     sophisticated

     and

     capable

    ,

     with

     the

     ability

     to

     learn

     from

     large

     amounts

     of

     data

     and

     make

     more

     accurate

     predictions

    .

     Computer

     vision

     will

     also

     be

     more

     advanced

    ,

     with

     the

     ability

     to

     see

     beyond

     the

     surface

     level

     and

     interpret

     complex

    



```python
llm.shutdown()
```
