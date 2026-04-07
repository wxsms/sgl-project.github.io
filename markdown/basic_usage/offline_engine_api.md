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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.85it/s]


    2026-04-07 07:04:31,274 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 07:04:31] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  6.10it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.10it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.10it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.10it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.37it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.99it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.99it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.99it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.99it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.99it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.99it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.99it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.05it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.05it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.05it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.05it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.05it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.05it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.05it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.69it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.69it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.85it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.85it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.85it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.85it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.85it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.85it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.85it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=121.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=121.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=121.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.98 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.67it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.67it/s] Capturing num tokens (num_tokens=896 avail_mem=120.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=832 avail_mem=120.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=832 avail_mem=120.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=768 avail_mem=120.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=704 avail_mem=120.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=640 avail_mem=120.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=576 avail_mem=120.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=512 avail_mem=120.94 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=512 avail_mem=120.94 GB):  50%|█████     | 29/58 [00:00<00:00, 40.22it/s]Capturing num tokens (num_tokens=480 avail_mem=120.95 GB):  50%|█████     | 29/58 [00:00<00:00, 40.22it/s]

    Capturing num tokens (num_tokens=448 avail_mem=120.95 GB):  50%|█████     | 29/58 [00:00<00:00, 40.22it/s]Capturing num tokens (num_tokens=416 avail_mem=120.95 GB):  50%|█████     | 29/58 [00:00<00:00, 40.22it/s]Capturing num tokens (num_tokens=384 avail_mem=120.95 GB):  50%|█████     | 29/58 [00:00<00:00, 40.22it/s]Capturing num tokens (num_tokens=352 avail_mem=120.94 GB):  50%|█████     | 29/58 [00:00<00:00, 40.22it/s]Capturing num tokens (num_tokens=352 avail_mem=120.94 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=320 avail_mem=120.94 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=288 avail_mem=120.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=256 avail_mem=120.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=240 avail_mem=120.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=224 avail_mem=120.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.38it/s]

    Capturing num tokens (num_tokens=224 avail_mem=120.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.53it/s]Capturing num tokens (num_tokens=208 avail_mem=120.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.53it/s]Capturing num tokens (num_tokens=192 avail_mem=120.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.53it/s]Capturing num tokens (num_tokens=176 avail_mem=120.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.53it/s]Capturing num tokens (num_tokens=160 avail_mem=120.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.53it/s]Capturing num tokens (num_tokens=144 avail_mem=120.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.53it/s]Capturing num tokens (num_tokens=144 avail_mem=120.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=128 avail_mem=120.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=112 avail_mem=120.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=96 avail_mem=120.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.62it/s] Capturing num tokens (num_tokens=80 avail_mem=120.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.62it/s]

    Capturing num tokens (num_tokens=64 avail_mem=120.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=64 avail_mem=120.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.66it/s]Capturing num tokens (num_tokens=48 avail_mem=120.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.66it/s]Capturing num tokens (num_tokens=32 avail_mem=120.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.66it/s]Capturing num tokens (num_tokens=28 avail_mem=120.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.66it/s]Capturing num tokens (num_tokens=24 avail_mem=120.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.66it/s]Capturing num tokens (num_tokens=20 avail_mem=120.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.66it/s]Capturing num tokens (num_tokens=20 avail_mem=120.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=16 avail_mem=120.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=12 avail_mem=120.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=8 avail_mem=120.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.05it/s] Capturing num tokens (num_tokens=4 avail_mem=120.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.05it/s]

    Capturing num tokens (num_tokens=4 avail_mem=120.87 GB): 100%|██████████| 58/58 [00:01<00:00, 39.45it/s]


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
    Generated text:  Mandy, I am a 17 year old school girl. I'm very good at drawing but I'm not very good at writing. I think I can become a writer if I read a lot of books and practice. Could you please tell me how to improve my writing skills? Also, I am very good at drawing, but I'm not good at writing. Is there a way to improve my writing skills if I read a lot of books and practice? And how can I become a writer if I read a lot of books? Mandy is a 17-year-old high school student who is good at drawing but not
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a third term, which means he is seeking reelection to office in 2025. The first term of office is 4 years. The president must be 70 years old in 2025 to run for a third term. How old is the president now if he is 70 years old in 1975 and the president is 60 years old in 1985?
    
    To determine the current age of the president, we need to first find out how many years have passed since the president's first term, and then use that information to calculate his current age in 
    ===============================
    Prompt: The capital of France is
    Generated text: : [ ].
    A. Paris
    B. Lyon
    C. Nice
    D. Dijon
    Answer: A
    
    Which of the following is NOT a common function of databases? ____.
    A. Real-time monitoring
    B. Data querying
    C. Data modification
    D. Data analysis
    Answer: A
    
    The number of binary bits in an IP address is: [ ]
    A. 10
    B. 20
    C. 30
    D. 40
    Answer: A
    
    The main physical network used in the Internet is the ____.
    A. Local Area Network
    B. Metropolitan Area
    ===============================
    Prompt: The future of AI is
    Generated text:  now. In 2022, AI is still in its infancy but already it is here and will play a significant role in the future. According to a report, the global AI market size is expected to reach $267.3 billion by 2027. This figure is expected to grow at a CAGR of 25.5% between 2022 and 2027. As a result, the demand for AI engineers is increasing and the entry level jobs in the AI field are in high demand.
    In the academic field, the demand for AI engineering is high. With the


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm passionate about [What I love to do]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Personality Type] with [Strengths/Weaknesses]. I'm [What I'm Known For]. I'm [What I'm Looking For in a Partner]. I'm [What I'm Looking for in a Job]. I'm [What I'm Looking for in a Relationship]. I'm [What I'm Looking for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and music, and is a major tourist destination. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. It is the largest city in France and the second-largest city in the world by population. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more rigorous testing and evaluation of AI systems, as well as greater transparency and
    


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
    Generated text:  [Name]. I am a [Occupation] who have always been passionate about [Occupation]. I strive to make the world a better place by using my skills and experience to help people in need. I'm always looking for new opportunities to learn and grow in this field. Thanks for having me. [Name] looks up. (Dress in neutral and professional attire as appropriate for the character's occupation and setting.) Hey there! 🌟 I'm [Name], a [Occupation] who's always passionate about [Occupation]. As someone who's always trying to make the world a better place, I'm always looking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the city of love, and it is one of the most popular cities in the world. The city is located on the Seine River and is home to many iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its cultural scene, including the annual Eiffel Tower Tower Fair and the famous street food, the Galerie du Marche. The city is also home to a diverse population, with over 3 million people residing in its metro area. Overall, Paris is a vibrant, exciting, and popular city that is widely recognized around the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a rapid increase in its applications and development, as well as a further improvement in its efficiency and accuracy. Some possible future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more advanced and complex, there is a growing emphasis on ethical considerations and ensuring that they are used in a responsible and unbiased manner.
    
    2. More specialized AI: AI will continue to be developed and refined in areas such as natural language processing, robotics, and computer vision, but it is likely that there will be a greater emphasis on developing more specialized and domain-specific AI models for tasks that are more complex or require


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

    ]

     and

     I

    'm

     an

     [

    age

    ]

     year

     old

     [

    gender

    ]

     with

     [

    sp

    elling

     correct

     name

    ]

     born

     on

     [

    birth

     date

    ].

     I

    'm

     the

     daughter

     of

     [

    father

    's

     name

    ]

     and

     [

    mother

    's

     name

    ].

     I

     have

     a

     happy

     and

     loving

     family

    ,

     where

     we

     enjoy

     spending

     time

     together

    .

     What

     can

     you

     tell

     me

     about

     yourself

     and

     your

     family

    ?

     How

     do

     you

     stay

     healthy

     and

     happy

    ?

     You

     have

     a

     kind

     heart

    ,

     a

     big

     heart

    ,

     and

     a

     big

     family

    .

     What

     do

     you

     do

     in

     your

     free

     time

    ?

     I

     enjoy

     playing

     board

     games

     and

     reading

    .

     How

     do

     you

     feel

     about

     the

     future

    ?

     I

     am

     currently

     in

     high

     school

     and

     am

     looking

     forward

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Petite

     Ville

    ",

     and

     is

     the

     largest

     city

     in

     the

     country

     by

     area

    .

     It

     is

     also

     the

     seat

     of

     the

     French

     government

    ,

     the

     head

     of

     state

    ,

     and

     the

     capital

     of

     the

     overseas

     departments

    .

     The

     city

     is

     home

     to

     many

     renowned

     cultural

     institutions

    ,

     such

     as

     the

     Lou

    vre

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     known

     as

     the

     "

    City

     of

     Light

    "

     for

     its

     vibrant

     and

     colorful

     streets

     and

     festivals

     throughout

     the

     year

    .

     It

     is

     home

     to

     many

     iconic

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Mont

    mart

    re

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Overall

    ,

     Paris

     is

     a

     fascinating

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     progress

     and

     innovation

    .

     Here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     emerge

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     AI

     will

     continue

     to

     become

     more

     personal

    ,

     with

     the

     ability

     to

     learn

     and

     adapt

     to

     individual

     users

    '

     needs

     and

     preferences

    .

     This

     will

     enable

     more

     personalized

     experiences

    ,

     such

     as

     speech

     recognition

    ,

     image

     recognition

    ,

     and

     predictive

     maintenance

    .
    


    2

    .

     Autonomous

     vehicles

    :

     AI

     will

     play

     an

     increasingly

     important

     role

     in

     autonomous

     vehicles

    ,

     with

     the

     ability

     to

     learn

     from

     data

    ,

     make

     decisions

    ,

     and

     navigate

     roads

     safely

    .

     This

     will

     require

     advanced

     algorithms

     and

     machine

     learning

     techniques

    .
    


    3

    .

     Smart

     cities

    :

     AI

     will

     be

     used

     to

     optimize

     city

     services

    



```python
llm.shutdown()
```
