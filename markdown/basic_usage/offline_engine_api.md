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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.07it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.07it/s]


    2026-05-06 12:38:36,365 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 12:38:36] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.08it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.08it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.66it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.66it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.66it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.66it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.66it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.66it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.66it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.66it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.66it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.71it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.71it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.71it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.71it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.71it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.71it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.71it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.71it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.71it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.71it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.54it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.54it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.54it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.54it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.54it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.54it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.54it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.54it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.54it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.54it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.27it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 37.98it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 37.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 34.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 34.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 34.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 34.81it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 34.81it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 34.81it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.86it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.86it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.86it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.86it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.86it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.86it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.07it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.07it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.07it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.07it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.07it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.07it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.35it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.35it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.35it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.35it/s]

    Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.77it/s]

    Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.77it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.98it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.98it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.98it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.98it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.98it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.98it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.19it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.19it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.19it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.19it/s]

    Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.19it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.19it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.02it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.02it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.02it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 34.41it/s]


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
    Generated text:  Anna, I'm 15 years old and I live in the United States. I like to eat vegetables and fruit, I like to read books, I like to ride a bike. I have a good friend named Mike and he likes to play football. My favorite color is yellow. I can't play the piano but I like to draw. I am also interested in other sports like swimming and tennis. I have a bicycle and I love to ride it. My favorite day is Friday and my favorite subject is English because it's my favorite. My favorite subject is science because I like to look at things and learn about them. My
    ===============================
    Prompt: The president of the United States is
    Generated text:  a male. The next president will also be a male.
    Does this mean that the president of the United States has always been a male?
    Pick your answer from:
    a). yes;
    b). no;
    b). no;
    ===============================
    Prompt: The capital of France is
    Generated text:  called ____.
    A. Paris
    B. Rome
    C. Berlin
    D. Moscow
    Answer:
    
    A
    
    Which of the following is the most basic method to determine if a certain individual is a spy or not? ____ 
    A. Discharge phone calls
    B. Physical examination
    C. Court trial
    D. Review of electronic communications
    Answer:
    
    D
    
    ____ is the best way to understand the true image of the real world.
    A. Observation
    B. Inquiry
    C. Perspective
    D. Evaluation
    Answer:
    
    A
    
    The most suitable type of research for understanding the true image of the real world is ____.
    A
    ===============================
    Prompt: The future of AI is
    Generated text:  here
    
    This week, I was invited to speak at the Q&A session of the World Economic Forum in Davos. In my presentation, I thought of a particularly interesting article in The Wall Street Journal. The article, titled “Why the AI revolution is about to begin,” argues that we are entering a new era of artificial intelligence, where AI will become more complex and more capable, but will also be more sensitive, and therefore more dangerous.
    
    There are several reasons why AI is so complex and capable, but for me, the reasons are most compelling and why the future of AI is about to begin are


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Skill] who has been [Number of Years] years in the field of [Field of Interest]. I have always been [Positive Trait] and have always been [Positive Trait]. I am [Positive Trait] and I am always [Positive Trait]. I am [Positive Trait] and I am always [Positive Trait]. I am [Positive Trait] and I am always [Positive Trait]. I am [Positive Trait] and I am always [Positive Trait]. I am [Positive Trait] and I am always [Positive Trait]. I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also known for its cuisine, including French cuisine, and is home to many famous French artists and writers. The city is also known for its fashion industry, with many famous fashion designers and boutiques. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased focus on privacy and security: As AI becomes more integrated into our daily lives, there will be a greater focus on privacy and security, with concerns about data privacy, cybersecurity, and the potential for AI to
    


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
    Generated text:  [Your Name], and I am a [Role] in our fictional organization. I'm here to assist [Role] in the following ways:
    1. [Briefly describe one or two actions you have done as [Role]].
    2. [Briefly describe one or two things you would like to learn about or about [Role]].
    3. [Briefly describe one or two things you would like to contribute to [Role] in the future].
    4. [Briefly describe one or two qualities you believe [Role] should have in order to succeed]. 
    5. [Briefly describe one or two skills you have used
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a bustling metropolis with a rich history and diverse culture. It is located in the Île-de-France region and is the most populous city in the country. Paris is known for its vibrant arts and culture scene, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major transportation hub for Europe, with many important transportation systems connecting the city to other parts of France and the world. Paris is a popular tourist destination, attracting millions of visitors each year for its beautiful architecture, delicious cuisine, and world-class museums and landmarks. Despite its size, Paris has a strong sense of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid progress and significant advancements in the areas of machine learning, natural language processing, and autonomous systems. Some potential future trends in AI include:
    
    1. More advanced machine learning algorithms: As AI technology continues to evolve, more sophisticated algorithms will be developed to improve accuracy, efficiency, and adaptability. This will be particularly important in areas such as healthcare, finance, and transportation.
    
    2. Personalization and interaction: AI will continue to improve in terms of personalization and interaction, allowing machines to better understand and respond to the needs of their users. This will enable more seamless and intuitive interactions between humans and AI systems.
    
    


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

    ].

     I

    'm

     a

     [

    Your

     profession

    /

    role

    ].

     I

     have

     [

    Your

     skills

     or

     unique

     qualities

    ].

     I

     am

     dedicated

     to

     [

    Your

     goal

     or

     mission

    ].

     I

     work

     hard

     to

     [

    Your

     actions

     or

     achievements

    ].

     I

     strive

     to

     [

    Your

     values

     or

     beliefs

    ].

     I

     am

     [

    Your

     personality

     traits

     or

     personality

     type

    ].

     I

     am

     always

     [

    Your

     attitude

     or

     manner

    ].

     I

     am

     [

    Your

     ability

     to

     handle

     various

     situations

    ].

     I

     am

     always

     [

    Your

     energy

     level

    ].

     I

     am

     [

    Your

     ideal

     level

     of

     work

    -life

     balance

    ].

     I

     am

     [

    Your

     social

     media

     presence

    ].

     I

     am

     [

    Your

     personal

     hobbies

     or

     interests

    ].

     I

     am

     [

    Your

     general

     disposition

    ].


    My

     name

     is

     Sarah

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     French

     capital

     city

     is

     Paris

    .

     The

     capital

     of

     France

     is

     Paris

    .

     This

     answer

     is

     correct

     as

     it

     accurately

     identifies

     Paris

     as

     the

     capital

     of

     France

    ,

     providing

     a

     concise

     factual

     statement

    .

     The

     capital

     of

     France

    ,

     also

     known

     as

     Paris

    ,

     is

     a

     large

    ,

     historic

     city

     located

     in

     the

     south

     of

     the

     country

    .

     It

     is

     a

     major

     center

     of

     culture

    ,

     politics

    ,

     and

     business

     in

     France

    .

     Paris

     is

     home

     to

     many

     famous

     landmarks

     and

     attractions

    ,

     such

     as

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     known

     for

     its

     rich

     cultural

     heritage

    ,

     including

     the

     Roman

    es

    que

    ,

     Gothic

    ,

     and

     Renaissance

     architectural

     styles

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     significant

     growth

     in

     both

     the

     breadth

     and

     depth

     of

     its

     applications

    .

     Here

     are

     some

     of

     the

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Accessibility

    :

     As

     AI

     technology

     improves

    ,

     the

     cost

     of

     using

     AI

     solutions

     will

     decrease

    ,

     making

     it

     more

     accessible

     to

     businesses

     and

     organizations

     of

     all

     sizes

    .

     This

     will

     lead

     to

     an

     increase

     in

     the

     number

     of

     AI

    -based

     applications

    ,

     with

     more

     people

     being

     able

     to

     access

     AI

     tools

     and

     services

    .
    


    2

    .

     Improved

     Security

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     the

     risk

     of

     security

     vulnerabilities

     will

     also

     increase

    .

     To

     address

     this

    ,

     there

     will

     be

     an

     increased

     focus

     on

     developing

     more

     secure

     and

     robust

     AI

     systems

     that

     can

     protect

     against

     attacks

     and

    



```python
llm.shutdown()
```
