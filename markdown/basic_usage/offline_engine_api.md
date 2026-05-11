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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]


    2026-05-11 05:05:17,316 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 05:05:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:45,  3.95s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:45,  3.95s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:45,  3.95s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:45,  3.95s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:45,  3.95s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.84it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.78it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.78it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.78it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.78it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.78it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.78it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.78it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.78it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.78it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.78it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.38it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.38it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.38it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.38it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.38it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.38it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.38it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 18.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.55 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 34.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 34.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 34.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  31%|███       | 18/58 [00:00<00:01, 34.89it/s]Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  31%|███       | 18/58 [00:00<00:01, 34.89it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  31%|███       | 18/58 [00:00<00:01, 34.89it/s]Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.24it/s]Capturing num tokens (num_tokens=832 avail_mem=72.53 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.24it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.24it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.24it/s]Capturing num tokens (num_tokens=640 avail_mem=72.52 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.24it/s]Capturing num tokens (num_tokens=576 avail_mem=72.52 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.24it/s]Capturing num tokens (num_tokens=576 avail_mem=72.52 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.68it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.68it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.68it/s]

    Capturing num tokens (num_tokens=448 avail_mem=72.52 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.68it/s]Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.68it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.68it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  57%|█████▋    | 33/58 [00:00<00:00, 35.71it/s]Capturing num tokens (num_tokens=352 avail_mem=72.51 GB):  57%|█████▋    | 33/58 [00:00<00:00, 35.71it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=256 avail_mem=72.50 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=240 avail_mem=72.49 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.71it/s]

    Capturing num tokens (num_tokens=240 avail_mem=72.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=192 avail_mem=72.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=176 avail_mem=72.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=144 avail_mem=72.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]

    Capturing num tokens (num_tokens=96 avail_mem=72.47 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s] Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.58it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.58it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.58it/s]Capturing num tokens (num_tokens=32 avail_mem=72.45 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.58it/s]Capturing num tokens (num_tokens=28 avail_mem=72.45 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.58it/s]Capturing num tokens (num_tokens=24 avail_mem=72.45 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.58it/s]Capturing num tokens (num_tokens=24 avail_mem=72.45 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=20 avail_mem=72.44 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=12 avail_mem=72.44 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s]

    Capturing num tokens (num_tokens=8 avail_mem=72.43 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s] Capturing num tokens (num_tokens=4 avail_mem=72.43 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 44.63it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 37.93it/s]


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
    Generated text:  Maria and I am the owner of a small bakery. Currently, I am opening my first gluten-free, almond flour bakery. I am trying to create an Instagram account to showcase the products I produce. I have a question regarding a certain type of pastry that I am baking.
    Could you please provide me with a recipe for a gluten-free almond flour pastry? I would appreciate it if you also included the ingredients, measurements, and step-by-step instructions. Additionally, I would like to know if there are any specific tips or tricks for making this pastry, such as the best equipment or techniques to use.
    Additionally, I would like to know if
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He/she is in charge of the country and makes important decisions for the country. President Bush of the United States is the youngest president in the history of the United States. What happened was that the president wanted to invite some of his friends to dinner party. He invited them from all over the country, but he forgot to mention some important things. When he was returning home, he went to a hotel to have dinner. When he was at the hotel, he met some of his friends at the restaurant and asked them to come to dinner. When he was going to the hotel, he went to the restaurant, but
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    
    B) Paris
    
    C) London
    
    D) Tokyo
    
    E) Berlin
    To determine the capital of France, we need to identify the capital city of the country. The capital of France is Paris. Let's break it down step by step:
    
    1. Identify the capital cities of the other countries mentioned: Spain, Italy, Portugal, and Greece.
    2. Check the capital city of each of these countries to see if it matches France.
    
    - Spain: Madrid
    - Italy: Rome
    - Portugal: Lisbon
    - Greece: Athens
    
    From the list above, the capital of France is Paris. Therefore,
    ===============================
    Prompt: The future of AI is
    Generated text:  always evolving. However, here are a few predictions about the future of AI.
    
      1. The human side of AI will increasingly be considered part of the future of AI.
      2. The work of AI will continue to get more complex, but less risky.
      3. The world of AI will become more decentralized.
      4. AI will be the key to solving the world's problems and reduce the risk of war.
    
    The AI model is named after the British scientist Alan Turing, who first developed the concept of artificial intelligence (AI) in 1950, and the model of the Turing machine in


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name], and I'm excited to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Parliament House. Paris is a bustling metropolis with a rich cultural heritage and is a popular tourist destination. The city is known for its fashion, art, and cuisine, and is a major center of politics, science, and culture in France. Paris is the capital of France and is the largest city in the European Union by population. It is also the oldest capital city in the world, having been founded in 
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and cost-effective solutions, but it could also lead to job displacement and changes in work patterns.
    
    2. Enhanced privacy and security: As AI technology becomes more advanced, there will be an increased need for privacy and security measures to protect personal data and prevent cyber attacks. This could lead to the
    


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
    Generated text:  [Your Name], and I am a [Your occupation or profession]. I have always had a passion for [Your hobby or interest], which has driven me to pursue a career in [Your career]. I am currently [Your age], and I am currently working as a [Your job title] at [Your company name]. I have always been dedicated to [Your personal goal or dream], and I am constantly seeking ways to achieve it. I am always looking for new challenges and opportunities to grow and succeed in my career. Thank you for asking! How can I assist you? What's your story? What are you most excited about at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower and diverse cultural and historical landmarks. France's capital city, Paris, is renowned for its iconic Eiffel Tower and a rich history dating back to the Middle Ages. The city is also known for its diverse cultural and historical landmarks, including the Louvre Museum and Notre-Dame Cathedral. Paris is a major hub of arts and entertainment, with numerous museums, theaters, and other attractions. The city has a long-standing tradition of intellectual and cultural pursuits, and it remains a vibrant and lively city today. The city has a population of approximately 2.2 million people, making it the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and unpredictable, but here are some possible trends that have the potential to shape it:
    
    1. Increased focus on ethical AI: There is a growing recognition that AI is not only enabling but also causing significant ethical problems, such as bias, transparency, and accountability. As a result, there is a push for greater focus on ethical AI and accountability in AI development and deployment.
    
    2. Increased reliance on machine learning: With the exponential growth of data availability, machine learning is becoming increasingly powerful and versatile. It can be used to automate a wide range of tasks, from customer service to healthcare, with greater efficiency and accuracy.
    
    3. Integration


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

     [

    character

    's

     profession

     or

     hobby

    ].

     I

    'm

     here

     to

     help

     you

     with

     any

     questions

     you

     may

     have

    .

     What

     can

     I

     do

     for

     you

     today

    ?

     Let

    's

     get

     started

    !

     What

    's

     your

     name

    ,

     and

     what

    's

     your

     profession

     or

     hobby

    ?

     What

     brings

     you

     to

     this

     city

    ?

     I

    'm

     really

     excited

     to

     meet

     you

     and

     help

     you

     in

     any

     way

     I

     can

    .

     What

     can

     I

     do

     for

     you

    ?

     I

    'm

     excited

     to

     meet

     you

     and

     help

     you

     in

     any

     way

     I

     can

    .

     When

     can

     I

     expect

     to

     see

     you

     again

    ?

     When

     can

     I

     expect

     to

     see

     you

     again

    ?

     I

    'm

     really

     excited

     to

     meet

     you

     and

     help

     you

     in

     any

     way

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     the

     fifth

    -largest

     city

     in

     the

     world

    .

     It

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     is

     considered

     one

     of

     the

     world

    's

     cultural

     cities

    .

     Paris

     is

     known

     for

     its

     historical

     landmarks

    ,

     fashion

     industry

    ,

     and

     art

     scene

    .

     It

     is

     also

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

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     has

     a

     diverse

     population

    ,

     including

     people

     of

     various

     national

    ities

     and

     cultures

    .

     It

     is

     known

     for

     its

     cultural

     heritage

    ,

     art

    ,

     and

     cuisine

    .

     The

     city

     is

     also

     home

     to

     the

     headquarters

     of

     many

     major

     European

     companies

     and

     the

     French

     Parliament

    .

     It

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     trends

     that

     are

     likely

     to

     shape

     the

     technology

    's

     development

     and

     impact

     on

     various

     industries

    .

     Some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     The

     ethical

     implications

     of

     AI

     are

     becoming

     increasingly

     important

    ,

     and

     governments

     and

     organizations

     will

     need

     to

     prioritize

     the

     development

     of

     AI

     systems

     that

     are

     designed

     to

     be

     fair

    ,

     transparent

    ,

     and

     accountable

    .

     This

     will

     require

     a

     deep

     understanding

     of

     human

     values

     and

     the

     potential

     impact

     of

     AI

     on

     society

    .
    


    2

    .

     Integration

     of

     AI

     with

     other

     technologies

    :

     AI

     is

     not

     just

     a

     tool

     for

     solving

     problems

    ,

     but

     also

     a

     component

     of

     other

     technologies

    .

     For

     example

    ,

     AI

     can

     be

    



```python
llm.shutdown()
```
