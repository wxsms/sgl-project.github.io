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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.28it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.27it/s]


    2026-04-13 04:29:07,468 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 04:29:07] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:03, 12.27it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:03, 12.27it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:02<00:03, 12.27it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.27it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.27it/s]

    Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 17.51it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 17.51it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 17.51it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 17.51it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 17.51it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 17.51it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:01, 17.51it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:03<00:01, 17.51it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:03<00:01, 24.58it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 31.83it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 31.83it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 31.83it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 31.83it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 31.83it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 31.83it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 31.83it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 31.83it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 38.87it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 38.87it/s]

    Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 45.35it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.07 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.07 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.07 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.07 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.07 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.06 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.07 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.06 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.06 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.03it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.03it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=61.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.84it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s]Capturing num tokens (num_tokens=960 avail_mem=61.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=61.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s]Capturing num tokens (num_tokens=832 avail_mem=61.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s]Capturing num tokens (num_tokens=832 avail_mem=61.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.92it/s]Capturing num tokens (num_tokens=768 avail_mem=61.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.92it/s]Capturing num tokens (num_tokens=704 avail_mem=61.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.92it/s]Capturing num tokens (num_tokens=640 avail_mem=61.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.92it/s]Capturing num tokens (num_tokens=576 avail_mem=61.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.92it/s]Capturing num tokens (num_tokens=512 avail_mem=60.99 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.92it/s]Capturing num tokens (num_tokens=512 avail_mem=60.99 GB):  50%|█████     | 29/58 [00:00<00:00, 36.99it/s]Capturing num tokens (num_tokens=480 avail_mem=61.00 GB):  50%|█████     | 29/58 [00:00<00:00, 36.99it/s]

    Capturing num tokens (num_tokens=448 avail_mem=61.00 GB):  50%|█████     | 29/58 [00:00<00:00, 36.99it/s]Capturing num tokens (num_tokens=416 avail_mem=61.00 GB):  50%|█████     | 29/58 [00:00<00:00, 36.99it/s]Capturing num tokens (num_tokens=384 avail_mem=61.00 GB):  50%|█████     | 29/58 [00:00<00:00, 36.99it/s]Capturing num tokens (num_tokens=384 avail_mem=61.00 GB):  57%|█████▋    | 33/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=352 avail_mem=60.99 GB):  57%|█████▋    | 33/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=320 avail_mem=60.99 GB):  57%|█████▋    | 33/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=288 avail_mem=60.98 GB):  57%|█████▋    | 33/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=256 avail_mem=60.98 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=240 avail_mem=60.98 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=224 avail_mem=60.98 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=224 avail_mem=60.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=208 avail_mem=60.97 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.71it/s]

    Capturing num tokens (num_tokens=192 avail_mem=60.97 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=176 avail_mem=60.97 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=160 avail_mem=60.97 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=144 avail_mem=60.96 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=128 avail_mem=60.96 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=128 avail_mem=60.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=112 avail_mem=60.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=96 avail_mem=60.95 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.52it/s] Capturing num tokens (num_tokens=80 avail_mem=60.95 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=64 avail_mem=60.95 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=48 avail_mem=60.94 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=32 avail_mem=60.94 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.52it/s]

    Capturing num tokens (num_tokens=32 avail_mem=60.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=28 avail_mem=60.93 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=24 avail_mem=60.93 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=20 avail_mem=60.93 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=16 avail_mem=60.93 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=12 avail_mem=60.92 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=8 avail_mem=60.92 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.05it/s] Capturing num tokens (num_tokens=8 avail_mem=60.92 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.74it/s]Capturing num tokens (num_tokens=4 avail_mem=60.92 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.74it/s]Capturing num tokens (num_tokens=4 avail_mem=60.92 GB): 100%|██████████| 58/58 [00:01<00:00, 40.06it/s]


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
    Generated text:  Tom. My parents are both teachers in our school. I like my parents because they are kind to me. But I don't like my parents because they are strict. Tom: Do you like your parents? They are kind to me. Tom: But I don't like my parents because they are strict. How do you think they are? Parents are important people. Some people think parents are bad because they never tell you the truth, but I don't think that's true. I like my parents very much. Parents love me very much and they teach me to be kind to people. And my parents also teach me to be polite
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing a new tax on individuals based on the income and wealth of their families. The tax would be collected through a complex and cumbersome system of deductions and credits. To understand the impact on individuals, a survey was conducted among 1000 randomly selected individuals, with a sample mean income of $120,000 and a sample standard deviation of $20,000. The population standard deviation for income is $50,000. 
    
    Calculate the margin of error for the estimate of the population mean income for this tax proposal, assuming a 95% confidence level. Also, determine the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, located in the center of the island of France. It is a historic and modern city that is the third largest city in France, and it is a major cultural and economic center in Europe.
    The 2022 FIFA World Cup will be the first time that the capital of France has hosted the tournament. It is the third time that the capital of France has hosted the World Cup. The first time that the capital of France hosted the World Cup was in 1938.
    The following table shows the World Cup that was hosted in Paris. In order to have it in the table, it must be known that the
    ===============================
    Prompt: The future of AI is
    Generated text:  so promising. But it also is incredibly risky. In the race to harness AI, there are three types of AI that must be looked at: the true AI, the fake AI, and the defective AI.
    The true AI is the most advanced AI, that is, the AI that is truly able to understand the world and make decisions. It is the most advanced AI that can understand and operate without human input or guidance. The defective AI is the AI that is not truly advanced, but is still capable of making decisions.
    As AI becomes more advanced, the chance of AI becoming defective becomes greater. The true AI is the one that is


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Person] who enjoys [Favorite Activity] and [Favorite Hobby]. I'm also a [Skill or Talent] who can [Describe a Skill or Talent]. I'm [Describe Your Personality] and I'm always [Describe Your Attitude]. I'm [Describe Your Goals] and I'm always [Describe Your Motivation]. I'm [Describe Your Values] and I'm always [Describe Your Ethics]. I'm [Describe Your Interests] and I'm always [Describe Your Interests]. I'm [Describe Your Skills]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" (floating city) due to its floating population of people. It is the largest city in Europe by population and is the seat of the French government and the country's cultural, political, and economic center. Paris is famous for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also known for its rich history, art, and cuisine. Paris is a popular tourist destination and a cultural hub, attracting millions of visitors each year. The city is home to many museums, theaters, and other
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations. This could lead to more efficient and effective decision-making, as well as more personalized experiences.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, such as ensuring that AI is used in a way that is fair and transparent. This could lead
    


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
    Generated text:  [Name]. I'm a/an [Title] who has been working at [Company/Position] for [Number] years. I bring a wealth of experience and expertise in [Job Title]. I'm confident and I believe in [Motto/Value]. [Name] started [Start Date] at [Company Name], and I've always been passionate about [Career Objective]. I'm always looking to learn and grow, always trying to improve myself. I'm a [Skill/Attitude] person and I'm always open to new experiences and ideas. If you need help with [Skill or Area], I'm here to help.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Its unique cultural and historical significance makes it a cultural and tourist destination for many visitors, attracting millions of tourists each year. Paris has a rich history dating back to ancient times, and today it is a bustling metropolis with a diverse range of neighborhoods and restaurants catering to various tastes and budgets. The city has been a global center of politics, art, and literature for centuries, and continues to play a vital role in French and European society. It is also known for its iconic landmarks, including the Eiffel Tower and the Louvre museum. As a result of its unique status, Paris is often referred to as "the city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be incredibly diverse, and there are numerous emerging trends and developments in this rapidly evolving field. Here are some of the key trends to watch for in the coming years:
    
    1. Personalized AI: As AI becomes more sophisticated, developers will be able to create systems that can learn from the data and preferences of individual users, offering tailored experiences and recommendations. This could lead to a new era of personalization in AI-powered services and products.
    
    2. AI for healthcare: The use of AI in healthcare will likely continue to expand, with more sophisticated algorithms being developed to improve the accuracy and efficiency of diagnosis and treatment. AI may also be


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

    name

    ]

     and

     I

    'm

     a

     [

    location

    ]

     citizen

    .

     I

    'm

     here

     to

     help

     you

     with

     any

     questions

     or

     concerns

     you

     might

     have

    .
    


    I

    'm

     always

     here

     to

     assist

     you

     with

     any

     questions

     you

     may

     have

    ,

     and

     I

    'm

     here

     to

     provide

     you

     with

     the

     information

     you

     need

     to

     make

     the

     best

     decision

     for

     your

     situation

    .

     Whether

     you

    're

     looking

     for

     advice

     on

     [

    topic

    ],

     or

     just

     need

     some

     help

     figuring

     out

     [

    something

    ],

     I

    'm

     here

     to

     help

    .

     
    


    So

    ,

     if

     you

     have

     any

     questions

     or

     concerns

    ,

     feel

     free

     to

     ask

    ,

     and

     I

    'll

     do

     my

     best

     to

     answer

     them

    .

     Let

    's

     get

     to

     know

     each

     other

     and

     have

     a

     chat

     about

     [

    topic

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     is

     accurate

     and

     brief

    ,

     summar

    izing

     the

     location

     of

     the

     main

     city

     in

     France

    .

     The

     statement

     includes

     only

     the

     key

     details

     required

     to

     convey

     the

     information

    ,

     without

     unnecessary

     details

     or

     elabor

    ation

    .

     It

     provides

     a

     clear

     and

     concise

     overview

     of

     the

     capital

     city

    ,

     which

     is

     the

     primary

     focus

     of

     the

     given

     question

    .

     The

     statement

     is

     appropriate

     for

     use

     in

     educational

     contexts

     or

     when

     providing

     information

     about

     the

     French

     capital

    .

     It

     is

     structured

     in

     a

     way

     that

     is

     easy

     to

     read

     and

     understand

    ,

     making

     it

     suitable

     for

     a

     wide

     audience

    .

     The

     statement

     is

     also

     likely

     to

     be

     seen

     as

     factual

     and

     accurate

     as

     it

     does

     not

     contain

     any

     assumptions

     or

     subjective

     elements

    .

     The

     statement

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     an

     exciting

     and

     rapidly

     evolving

     field

     with

     many

     potential

     areas

     of

     development

     and

     application

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

     algorithm

    ic

     decision

    -making

    :

     As

     more

     and

     more

     jobs

     are

     automated

    ,

     we

     may

     see

     an

     increase

     in

     the

     use

     of

     AI

    -powered

     automation

     in

     various

     industries

    .

     This

     could

     lead

     to

     more

     efficient

     and

     cost

    -effective

     processes

    ,

     but

     it

     could

     also

     lead

     to

     job

     displacement

     for

     humans

    .
    


    2

    .

     Enhanced

     natural

     language

     processing

    :

     As

     AI

     becomes

     more

     capable

     of

     understanding

     natural

     language

    ,

     we

     may

     see

     more

     applications

     of

     AI

     in

     areas

     like

     customer

     service

    ,

     chat

    bots

    ,

     and

     virtual

     assistants

    .
    


    3

    .

     Improved

     privacy

     and

     security

    :

     As

    



```python
llm.shutdown()
```
