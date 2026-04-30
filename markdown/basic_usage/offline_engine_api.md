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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.56it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.56it/s]


    2026-04-30 22:31:17,444 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 22:31:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.20it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.43it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.25it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.25it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.25it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.25it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.25it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.25it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.25it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.25it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.25it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.25it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.25it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.20it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.20it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.20it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.20it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.20it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.20it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.20it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.20it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.20it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.20it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.20it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.18it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.18it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.18it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.18it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.18it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.18it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.18it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.32 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.32 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.32 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.32 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.32 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.32 GB):   9%|▊         | 5/58 [00:00<00:02, 22.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.31 GB):   9%|▊         | 5/58 [00:00<00:02, 22.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.30 GB):   9%|▊         | 5/58 [00:00<00:02, 22.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.30 GB):   9%|▊         | 5/58 [00:00<00:02, 22.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.30 GB):   9%|▊         | 5/58 [00:00<00:02, 22.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.30 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.20it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=56.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.28 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.27 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.27 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.27 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.26 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.26 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.26 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.26 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.24 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.54it/s]Capturing num tokens (num_tokens=960 avail_mem=56.25 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.54it/s] Capturing num tokens (num_tokens=896 avail_mem=56.25 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.54it/s]Capturing num tokens (num_tokens=832 avail_mem=56.25 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.54it/s]

    Capturing num tokens (num_tokens=768 avail_mem=56.24 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.54it/s]Capturing num tokens (num_tokens=704 avail_mem=56.24 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.54it/s]Capturing num tokens (num_tokens=704 avail_mem=56.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=640 avail_mem=56.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=576 avail_mem=56.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=512 avail_mem=56.22 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=480 avail_mem=56.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=448 avail_mem=56.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=416 avail_mem=56.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=416 avail_mem=56.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.36it/s]Capturing num tokens (num_tokens=384 avail_mem=56.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.36it/s]Capturing num tokens (num_tokens=352 avail_mem=56.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.36it/s]Capturing num tokens (num_tokens=320 avail_mem=56.22 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.36it/s]

    Capturing num tokens (num_tokens=288 avail_mem=56.22 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.36it/s]Capturing num tokens (num_tokens=256 avail_mem=56.22 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.36it/s]Capturing num tokens (num_tokens=240 avail_mem=56.21 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.36it/s]Capturing num tokens (num_tokens=240 avail_mem=56.21 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.02it/s]Capturing num tokens (num_tokens=224 avail_mem=56.21 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.02it/s]Capturing num tokens (num_tokens=208 avail_mem=56.21 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.02it/s]Capturing num tokens (num_tokens=192 avail_mem=56.21 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.02it/s]Capturing num tokens (num_tokens=176 avail_mem=56.20 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.02it/s]Capturing num tokens (num_tokens=160 avail_mem=56.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 48.02it/s]Capturing num tokens (num_tokens=144 avail_mem=56.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 48.02it/s]Capturing num tokens (num_tokens=144 avail_mem=56.20 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.13it/s]Capturing num tokens (num_tokens=128 avail_mem=56.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.13it/s]Capturing num tokens (num_tokens=112 avail_mem=56.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.13it/s]

    Capturing num tokens (num_tokens=96 avail_mem=56.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.13it/s] Capturing num tokens (num_tokens=80 avail_mem=56.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.13it/s]Capturing num tokens (num_tokens=64 avail_mem=56.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.13it/s]Capturing num tokens (num_tokens=64 avail_mem=56.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.36it/s]Capturing num tokens (num_tokens=48 avail_mem=56.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.36it/s]Capturing num tokens (num_tokens=32 avail_mem=56.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.36it/s]Capturing num tokens (num_tokens=28 avail_mem=56.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.36it/s]Capturing num tokens (num_tokens=24 avail_mem=56.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.36it/s]Capturing num tokens (num_tokens=20 avail_mem=56.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.36it/s]Capturing num tokens (num_tokens=16 avail_mem=56.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.36it/s]Capturing num tokens (num_tokens=16 avail_mem=56.16 GB):  95%|█████████▍| 55/58 [00:01<00:00, 49.84it/s]Capturing num tokens (num_tokens=12 avail_mem=56.16 GB):  95%|█████████▍| 55/58 [00:01<00:00, 49.84it/s]Capturing num tokens (num_tokens=8 avail_mem=56.15 GB):  95%|█████████▍| 55/58 [00:01<00:00, 49.84it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=56.15 GB):  95%|█████████▍| 55/58 [00:01<00:00, 49.84it/s]Capturing num tokens (num_tokens=4 avail_mem=56.15 GB): 100%|██████████| 58/58 [00:01<00:00, 43.67it/s]


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
    Generated text:  Evgeny.
    
    I graduated from the University of Natal, and was a student at the University of South Africa, where I received a Bachelor of Commerce degree with an emphasis in Economics from 2007 to 2009.
    
    Currently, I am working as an Analyst at PwC. My work mainly involves consulting for companies, financial institutions, and other entities. I work primarily on financial analysis, but also perform some research and consulting on a range of topics including market analysis, financial modeling, and financial planning and forecasting.
    
    I have a degree in Economics from the University of Natal, a Bachelor of Commerce from the University
    ===============================
    Prompt: The president of the United States is
    Generated text:  240 cm tall. If the standard deviation of the heights of presidents is 6 cm, what is the probability that a randomly selected president is taller than 280 cm? To determine the probability that a randomly selected president is taller than 280 cm, we need to use the properties of the normal distribution. Here's the step-by-step process:
    
    1. **Identify the given values:**
       - The height of the president, \( X \), is normally distributed with mean \( \mu = 240 \) cm and standard deviation \( \sigma = 6 \) cm.
      
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Many people who live in Paris would like to live in a different city. They want to live in the city where the streets are wide and the buildings are tall and the weather is nice. There are many cities like that. But which one is the best? Let's take a look at some cities in Europe. We'll use maps and pictures of the cities. We'll try to find the best cities in Europe. For example, we might visit Rome or Berlin. But we will use other cities to compare. We'll look for cities that are located in different places on the map. For example, we might look for a
    ===============================
    Prompt: The future of AI is
    Generated text:  in the data. The focus of this online workshop will be to introduce participants to the challenges and opportunities of AI driven by data. The workshop will provide an overview of the current state of AI and data and how these two fields are currently impacting the design of data pipelines and AI models. This workshop will also provide an overview of the challenges and opportunities of AI and data driven machine learning solutions.
    This training will provide the participants with an understanding of the concepts and techniques that will enable them to create a data pipeline for data driven machine learning solutions that will improve their models and business models.
    This online training will include practical hands on sessions to help participants


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about your personality or skills]. I enjoy [insert a short, positive, enthusiastic statement about your hobbies or interests]. I'm always looking for ways to [insert a short, positive, enthusiastic statement about your goals or aspirations]. I'm a [insert a short, positive, enthusiastic statement about your attitude or personality]. I'm always eager to learn and grow, and I'm always looking for new
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also home to many world-renowned museums, including the Louvre and the Musée d'Orsay. Paris is a cultural and economic hub, with a diverse population and a rich history dating back to the Roman Empire. It is a popular tourist destination and a major center of politics, business, and culture in Europe. The city is also known for its cuisine, including
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is expected to become more prevalent in various industries, including manufacturing, healthcare, and transportation. Automation will likely lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be increased concerns about privacy and security. There will likely be more regulations and standards to ensure that AI is used responsibly and ethically.
    
    3. AI-driven innovation: AI will continue to drive innovation in many fields, from healthcare to finance to transportation. This will lead to
    


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
    Generated text:  [Your Name], and I'm an [insert the profession or title of your character] with a background in [insert your character's educational background or significant contributions]. What can you tell me about yourself? I've always been passionate about learning and understanding the world around me, whether it's through my work, my hobbies, or even just my own interests. I'm always looking for new knowledge and ways to expand my horizons. And when it comes to building relationships, I'm always looking for new people who share my interests and values. What are some of your hobbies or interests outside of work? I'm always eager to learn new
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located on the Mediterranean coast and the largest city in Europe.
    
    Sure! Paris is the capital of France, located on the Mediterranean coast. It is the largest city in Europe and is known for its rich history, beautiful architecture, and numerous cultural institutions. The city is also a major transportation hub and a major financial center. Paris has a long history dating back to the ancient Greeks and Romans, and it has played a significant role in French history and culture throughout its history. The city's landmarks, such as Notre Dame Cathedral and the Eiffel Tower, are some of France's most famous attractions. In recent years, Paris has
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, and there is no one definitive answer to what the future holds. However, some potential future trends in AI include:
    
    1. Increased reliance on AI in healthcare: AI will become more integrated into the healthcare system, allowing for more personalized and accurate diagnoses and treatments. This could lead to significant cost savings and improved patient outcomes.
    
    2. Advancements in natural language processing: AI will continue to improve in its ability to understand and generate natural language, leading to applications in areas such as chatbots, virtual assistants, and language translation.
    
    3. Increased integration of AI in consumer goods: AI will continue to be integrated into consumer goods,


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

     of

     the

     character

    ].

     I

    'm

     a

     [

    insert

     profession

     or

     occupation

    ],

     and

     I

    've

     always

     been

     passionate

     about

     [

    insert

     something

     that

     interests

     you

    ,

     like

     sports

    ,

     travel

    ,

     or

     technology

    ].

     I

     believe

     in

     [

    insert

     something

     important

     to

     you

    ,

     like

     kindness

    ,

     honesty

    ,

     or

     creativity

    ].

     I

    'm

     constantly

     learning

     and

     evolving

    ,

     and

     I

     enjoy

     [

    insert

     something

     that

     reflects

     this

    ,

     like

     trying

     new

     things

    ,

     exploring

     new

     cultures

    ,

     or

     keeping

     up

     with

     the

     latest

     trends

     in

     technology

    ].

     I

    'm

     always

     looking

     for

     ways

     to

     make

     a

     positive

     impact

     in

     the

     world

     and

     I

     strive

     to

     do

     so

     through

     my

     work

     and

     my

     personal

     actions

    .

     And

     last

    ly

    ,

     I

    'm

     a

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     architecture

    ,

     cultural

     attractions

    ,

     and

     rich

     history

    .

     Paris

     is

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

     and

     is

     the

     largest

     city

     in

     the

     European

     Union

    .

     The

     city

     has

     a

     rich

     tradition

     of

     art

    ,

     literature

    ,

     and

     cuisine

    ,

     and

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

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

     for

     its

     fashion

     industry

    ,

     and

     the

     city

     is

     a

     popular

     tourist

     destination

     throughout

     the

     year

    .

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

     architecture

    ,

     cultural

     attractions

    ,

     and

     rich

     history

    .

     It

     is

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     several

     trends

     are

     likely

     to

     shape

     it

     in

     the

     coming

     years

    .

     Some

     of

     the

     most

     significant

     trends

     include

    :
    


    1

    .

     Improved

     accuracy

    :

     AI

     systems

     are

     becoming

     more

     accurate

     and

     detailed

    ,

     but

     there

     are

     still

     many

     aspects

     of

     the

     world

     that

     AI

     cannot

     yet

     fully

     understand

     or

     predict

    .
    


    2

    .

     Personal

    ization

    :

     AI

     is

     becoming

     more

     personalized

    ,

     with

     systems

     that

     learn

     from

     user

     data

     to

     provide

     tailored

     experiences

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     use

     of

     resources

    .
    


    3

    .

     Ethics

     and

     compliance

    :

     AI

     systems

     are

     increasingly

     being

     used

     for

     applications

     that

     involve

     complex

     ethical

     and

     legal

     issues

    ,

     such

     as

     autonomous

     vehicles

    ,

     medicine

    ,

     and

     financial

     services

    .

     There

     is

     a

     need

    



```python
llm.shutdown()
```
