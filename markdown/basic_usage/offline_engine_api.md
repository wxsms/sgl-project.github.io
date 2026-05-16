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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.68it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.67it/s]


    2026-05-16 03:51:22,693 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 03:51:22] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.60it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.81it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.81it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.81it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.81it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.81it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.81it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.81it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.81it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.81it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.81it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.81it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.98it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.98it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.98it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.98it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.98it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.98it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.98it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 37.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 37.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 37.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  31%|███       | 18/58 [00:00<00:01, 37.12it/s]Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  31%|███       | 18/58 [00:00<00:01, 37.12it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  31%|███       | 18/58 [00:00<00:01, 37.12it/s]Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=832 avail_mem=71.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=640 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=480 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=448 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.12it/s]

    Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.94it/s]Capturing num tokens (num_tokens=352 avail_mem=71.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.94it/s]Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.94it/s]Capturing num tokens (num_tokens=288 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.94it/s]Capturing num tokens (num_tokens=256 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.94it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.94it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=224 avail_mem=71.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=208 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=176 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.09it/s]

    Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.89it/s]Capturing num tokens (num_tokens=144 avail_mem=71.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.89it/s]Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.89it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.89it/s]Capturing num tokens (num_tokens=96 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.89it/s] Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.89it/s]Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.93it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.93it/s]Capturing num tokens (num_tokens=48 avail_mem=71.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.93it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.93it/s]Capturing num tokens (num_tokens=28 avail_mem=71.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.93it/s]

    Capturing num tokens (num_tokens=24 avail_mem=71.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.93it/s]Capturing num tokens (num_tokens=24 avail_mem=71.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.00it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.00it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.00it/s]Capturing num tokens (num_tokens=12 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.00it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.00it/s] Capturing num tokens (num_tokens=4 avail_mem=71.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.00it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 47.38it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 41.79it/s]


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
    Generated text:  John and I am a professional professional gamer. I am not a gaming expert, but I do own a PlayStation 4, which I play in the PlayStation Network (PSN) server. I like to make cool graphics and designs with different levels of complexity. I would like to do a tutorial on this topic and would appreciate it if you would write an explanation for me. 
    
    What is High-Definition Video (HD)?
    
    What is Ultra HD?
    
    Are High-Definition Video (HD) and Ultra HD the same thing? (Show me an example of both types of video)
    
    Are HD and Ultra HD the same thing? (Show me
    ===============================
    Prompt: The president of the United States is
    Generated text:  a six-letter word. What is the word?
    The word "president" is six letters long. 
    
    To confirm, let's break it down:
    - "pres-" is a five-letter word, which completes the six-letter word.
    - "osition" is a five-letter word, which further completes the six-letter word.
    - "ator" is a five-letter word, which completes the six-letter word.
    
    Therefore, the six-letter word is "president". 
    
    To ensure accuracy, let's consider a hypothetical example:
    - If "pres-" is a five-letter word, it completes the six-letter word.
    - If "osition
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Marseille C. Bordeaux D. Lille
    
    The capital of France is Paris. Therefore, the correct answer is: A. Paris. 
    
    Please note that "Bordeaux" is the capital of France, but it's not a specific capital building or district; the capital of France is Paris. The other cities are not capitals of France. 
    
    Option C (Bordeaux) is not a capital city of France. The other options (Paris, Marseille, Lille) are all capital cities of France. 
    
    To clarify, Paris is the capital of France, and its name is French. 
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about translating digital data into a usable format. It also depends on the people and technology that are responsible for its development, deployment, and sustainability. Here are some of the top technologies and companies that have been at the forefront of developing AI, and what they are currently doing:
    1. Google: Google is one of the most advanced companies in AI development and is known for its popular Google Assistant, which is the world's first fully AI-powered assistant. Google is also working on creating more advanced AI-powered devices and applications, such as Google Translate and Google Maps.
    2. Amazon: Amazon is another company that has been leading the way


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [industry]. I'm a [job title] at [company name], and I enjoy [job title] work. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is also the birthplace of many famous French artists, writers, and composers. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. The city is home to many museums, theaters, and landmarks, including the Louvre and Notre-Dame Cathedral. It is a popular tourist destination and a major economic center in Europe. Paris is known for its cuisine, fashion, and art, and is a major center for international business and diplomacy. The city is also home to many international organizations and institutions,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI, there is likely to be a greater emphasis on ethical AI. This could include developing more transparent and accountable AI systems, as well as creating guidelines for how AI should be used.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could include developing AI that can understand and interpret human
    


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
    Generated text:  [Name]. I am a [major]. I have always been [relatively] [positive] about my [cultural background]. I am [experience]. I am [career]. I am [personality]. I am [role]. Thank you. That sounds great, but I feel like I'm not being completely honest. What can I do to improve my self-introduction so that it sounds more natural and less awkward? You can add your cultural background and experience, and be honest about your career and personality. Also, you can add your role, if that makes sense. And you can try to make it sound more informal or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historic and modern city with a rich history, home to numerous museums, monuments, and landmarks. The city is known for its vibrant culture, festivals, and cuisine. It is also home to a diverse population of people of many nationalities and ethnic backgrounds. Paris has become a global city and is a major economic and cultural center. It is the most visited city in the world and is a hub of creativity, fashion, and food. The city offers a unique and exciting experience for visitors from around the world. Therefore, it is a city of contrasts and diversity. Paris is a cultural and intellectual melting pot, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several trends and developments. Here are some of the possible future trends:
    
    1. Increased automation and AI: As more and more tasks become automated, we may see an increase in the automation of AI. This could lead to the creation of more efficient and accurate AI systems that can perform tasks more quickly and accurately than humans.
    
    2. Integration of AI into everyday life: AI is becoming more and more integrated into our daily lives, from personal assistants like Siri and Alexa to self-driving cars and robotics. This trend is likely to continue as AI becomes more sophisticated and integrated into our everyday technology.
    
    3. Ethics and privacy concerns


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

    ].

     I

    'm

     a

     professional

     programmer

    ,

     with

     a

     passion

     for

     coding

     and

     problem

    -solving

    .

     I

    've

     been

     coding

     since

     I

     was

     

    8

     years

     old

     and

     have

     written

     over

     

    1

    0

    0

     scripts

     that

     have

     helped

     businesses

     solve

     real

    -world

     problems

    .

     I

    'm

     also

     a

     skilled

     communicator

     and

     have

     a

     knack

     for

     explaining

     complex

     ideas

     in

     a

     simple

     way

    .

     I

     enjoy

     learning

     new

     programming

     languages

     and

     constantly

     exploring

     new

     ways

     to

     improve

     my

     skills

    .

     I

     believe

     that

     my

     programming

     skills

     are

     a

     valuable

     asset

     in

     the

     tech

     industry

    ,

     and

     I

    'm

     excited

     to

     share

     my

     knowledge

     with

     others

    .

     Thank

     you

     for

     taking

     the

     time

     to

     meet

     me

    ,

     [

    Name

    ].

     
    


    Note

    :

     I

     have

     a

     background

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

     and

     one

     of

     its

     most

     iconic

     landmarks

     is

     the

     E

    iff

    el

     Tower

    ,

     a

     famous

     French

     Art

     Nou

    veau

     skys

    craper

     located

     in

     the

     center

     of

     the

     city

    .

     It

    's

     a

     UNESCO

     World

     Heritage

     site

     and

     one

     of

     the

     most

     recognizable

     structures

     in

     the

     world

    .

     The

     E

    iff

    el

     Tower

     has

     been

     a

     beloved

     landmark

     of

     Paris

     since

     its

     completion

     in

     

    1

    8

    8

    9

     and

     is

     now

     a

     UNESCO

     World

     Heritage

     site

    .

     It

    's

     a

     fantastic

     example

     of

     Art

     Nou

    veau

     architecture

     and

     continues

     to

     attract

     millions

     of

     tourists

     every

     year

    .

     The

     E

    iff

    el

     Tower

     is

     a

     true

     Paris

    ian

     icon

     and

     a

     must

    -

    see

     attraction

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     full

     of

     possibilities

    .

     Here

     are

     some

     possible

     trends

    :
    


    1

    .

     Greater

     integration

     of

     AI

     into

     everyday

     life

    :

     As

     more

     and

     more

     applications

     of

     AI

     are

     integrated

     into

     daily

     life

    ,

     we

     can

     expect

     to

     see

     more

     machine

     learning

     algorithms

     and

     natural

     language

     processing

     capabilities

     that

     are

     seamlessly

     integrated

     into

     our

     daily

     routines

    .

     This

     could

     lead

     to

     a

     more

     personalized

     and

     automated

     experience

     for

     users

    ,

     potentially

     leading

     to

     greater

     convenience

     and

     efficiency

    .
    


    2

    .

     AI

     becoming

     more

     autonomous

    :

     The

     integration

     of

     AI

     will

     continue

     to

     grow

    ,

     but

     it

    's

     unlikely

     to

     become

     completely

     autonomous

     in

     the

     near

     future

    .

     AI

     systems

     will

     likely

     continue

     to

     be

     programmed

     with

     specific

     goals

     and

     tasks

    ,

     but

     they

     will

     have

     limited

     autonomy

    



```python
llm.shutdown()
```
