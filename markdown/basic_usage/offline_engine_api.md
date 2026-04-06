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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.91it/s]


    2026-04-06 04:44:44,713 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 04:44:44] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.21it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.21it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:23,  2.21it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:23,  2.21it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:23,  2.21it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:23,  2.21it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.21it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:08,  5.84it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:08,  5.84it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:08,  5.84it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:08,  5.84it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:08,  5.84it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:08,  5.84it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:08,  5.84it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:08,  5.84it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:08,  5.84it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 11.87it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 11.87it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.14it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.14it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.14it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.14it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.14it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.14it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.14it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 33.96it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 33.96it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 33.96it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 33.96it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 33.96it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 33.96it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 33.96it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.09it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.09it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.09it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.09it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.09it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.09it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.09it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.71 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=121.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.70 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.42 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.23it/s]Capturing num tokens (num_tokens=3584 avail_mem=121.41 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=121.11 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.23it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.70 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.70 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.70 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.37it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.70 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.37it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.70 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.69 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.69 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.69 GB):  31%|███       | 18/58 [00:00<00:01, 34.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.69 GB):  31%|███       | 18/58 [00:00<00:01, 34.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.68 GB):  31%|███       | 18/58 [00:00<00:01, 34.98it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.66 GB):  31%|███       | 18/58 [00:00<00:01, 34.98it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.68 GB):  31%|███       | 18/58 [00:00<00:01, 34.98it/s] Capturing num tokens (num_tokens=896 avail_mem=120.67 GB):  31%|███       | 18/58 [00:00<00:01, 34.98it/s]Capturing num tokens (num_tokens=896 avail_mem=120.67 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=832 avail_mem=120.67 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=768 avail_mem=120.67 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=704 avail_mem=120.66 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=640 avail_mem=120.66 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=576 avail_mem=120.66 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=576 avail_mem=120.66 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=512 avail_mem=120.65 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=480 avail_mem=120.66 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.86it/s]

    Capturing num tokens (num_tokens=448 avail_mem=120.66 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=416 avail_mem=120.66 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=384 avail_mem=120.66 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=384 avail_mem=120.66 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=352 avail_mem=120.32 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=256 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.32it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.08it/s]

    Capturing num tokens (num_tokens=208 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=192 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=96 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.77it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.77it/s]

    Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=48 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=28 avail_mem=120.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=24 avail_mem=120.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=24 avail_mem=120.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=12 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.40it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.40it/s]

    Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 38.92it/s]


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
    Generated text:  Jon, and I have never had a problem that I could not solve. I enjoy trying to understand the causes of human failure and inventing solutions to overcome them.
    
    What is my probability of winning the lottery?
    
    It's impossible to win the lottery. It's a game of chance, and no one has ever claimed that they have the ability to predict the future with any degree of accuracy.
    
    This does not make it a lottery, but it does mean that the notion of a lottery is a fiction that has no basis in reality. People like to think that the outcome of their lottery draws is certain, but it's clear that they cannot trust
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking advice on a proposal he has been working on for several months. The proposal involves a program that will help the president learn new languages, particularly Mandarin and Japanese, which are currently not spoken in the United States. The president has asked for your advice on the feasibility of this program and the impact it will have on the nation. Can you provide any insights or suggestions on how the program could be developed and implemented?
    As an AI language model, I cannot provide specific details on how the program will be developed and implemented, as it would depend on various factors such as budget, timeline, resources, and expertise of the project team. However,
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. New York
    C. London
    D. Berlin
    Answer:
    
    A
    
    The characteristic of the patient's oral mucosa is ____
    A. Deep and moist
    B. Red and congested
    C. Red and swollen
    D. Pus-filled
    E. White and edematous
    Answer:
    
    E
    
    For a 35-year-old woman with primary dysmenorrhea, which of the following should be chosen?
    A. Norethisterone
    B. Medroxyprogesterone
    C. Diethylstilbestrol
    D. Doxycycline
    ===============================
    Prompt: The future of AI is
    Generated text:  more complex than ever before, and the answers to these questions are already shaping the way that we live today. This article will discuss the current state of AI and how it is transforming the way we live, work, and interact with technology. It will also provide a summary of the key areas of AI research and development that are currently shaping the future of AI, including the areas of robotics, natural language processing, machine learning, and computer vision.
    What is AI and how does it work?
    AI (Artificial Intelligence) is a field that is focused on creating machines that can learn, reason, and make decisions similar to humans. It involves


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that serves as the political, cultural, and economic center of the country. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its rich history, including the influence of French colonialism in the Americas and the impact of the French Revolution. The city is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée d'Art Moderne. Paris is a vibrant and diverse city with a rich cultural scene, and it is a popular tourist destination. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As the technology continues to advance, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient monitoring.
    
    2. AI in manufacturing: AI is already being used in manufacturing to improve efficiency and reduce costs. As the technology continues to evolve, we can expect to see even more widespread use of AI in manufacturing,
    


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
    Generated text:  [Name] and I'm a young software engineer who graduated from [University/Colleges/Programs]. I have a strong passion for [field of interest]. I'm always on the lookout for opportunities to learn and grow, and I'm eager to contribute to the field. I'm patient, persistent and have a strong work ethic. I enjoy problem-solving, and I'm always looking for innovative solutions to complex issues. I'm excited to meet and learn from new people and to help build a team. I'm passionate about taking on new challenges and trying new things. I'm confident in my abilities and look forward to being a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris, also known as "la cité de l'amour," is the capital city of France. It's a charming and historical city located in the northwestern region of France, known for its beautiful architecture, vibrant culture, and rich history. Visitors can explore the iconic Eiffel Tower, see the Louvre Museum, explore the Notre-Dame Cathedral, and experience the city's diverse neighborhoods and cuisine. Paris is an important cultural and economic center of France, and it plays a significant role in French politics, economics, and daily life. The city has a fascinating history dating back to the 6
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking very promising and many possible trends are predicted. Here are a few:
    
    1. Integration with human emotions: AI is starting to be used more in areas where it can learn to understand and respond to human emotions. This includes natural language processing, machine learning, and speech recognition.
    
    2. Real-time interaction: AI is becoming more and more capable of real-time interaction, allowing it to respond to users in real-time, regardless of their distance. This is making it easier to provide personalized and interactive experiences.
    
    3. Universal AI: AI is becoming more capable of replicating human intelligence, making it possible for it to understand and respond to a


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

     name

    ],

     and

     I

    'm

     a

     computer

     scientist

     from

     [

    insert

     location

    ].

     I

    'm

     currently

     a

     [

    insert

     profession

     or

     field

     of

     study

    ]

     at

     [

    insert

     institution

     or

     company

    ],

     and

     I

    've

     been

     working

     on

     [

    insert

     research

     or

     project

    ]

     for

     [

    insert

     duration

     of

     time

    ].

     I

    'm

     always

     eager

     to

     learn

     and

     to

     explore

     new

     ideas

    ,

     and

     I

     love

     [

    insert

     reason

     for

     love

     or

     interest

    ].

     I

    'm

     a

     [

    insert

     personality

     trait

     or

     personality

     type

    ]

     and

     enjoy

     [

    insert

     hobby

     or

     activity

    ].

     I

    'm

     always

     on

     the

     lookout

     for

     new

     challenges

     and

     opportunities

    ,

     and

     I

    'm

     excited

     to

     see

     what

     the

     future

     holds

     for

     me

     in

     this

     field

    .

     Thank

     you

     for

     asking

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .
    


    Here

    's

     a

     concise

     factual

     statement

     about

     France

    's

     capital

     city

    :
    


    The

     capital

     of

     France

     is

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

     like

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     
    


    This

     statement

     captures

     the

     essence

     of

     Paris

    '

     reputation

     as

     a

     cultural

     and

     historical

     capital

     while

     highlighting

     the

     notable

     features

     of

     the

     city

    's

     most

     iconic

     landmarks

    .

     The

     key

     points

     are

    :


    1

    .

     Paris

     is

     the

     capital

     city

     of

     France

    .


    2

    .

     The

     capital

     is

     Paris

    ,

     France

    .


    3

    .

     Paris

     is

     known

     for

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     factors

    ,

     including

     advances

     in

     computing

     power

    ,

     increased

     data

     availability

    ,

     and

     the

     development

     of

     new

     technologies

     and

     algorithms

    .

     Some

     possible

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     automation

    :

     With

     the

     rise

     of

     automation

    ,

     AI

     will

     become

     more

     efficient

     and

     powerful

    ,

     freeing

     up

     human

     workers

     to

     focus

     on

     more

     complex

     tasks

    .
    


    2

    .

     Improved

     accessibility

    :

     AI

     will

     be

     more

     accessible

     to

     everyone

    ,

     with

     fewer

     barriers

     to

     entry

    ,

     making

     it

     easier

     for

     individuals

     to

     gain

     access

     to

     advanced

     AI

     technologies

    .
    


    3

    .

     Enhanced

     privacy

     and

     security

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     there

     will

     be

     an

     increased

     need

     for

     robust

     privacy

     and

     security

     measures

     to

     protect

     against

     misuse

     and

    



```python
llm.shutdown()
```
