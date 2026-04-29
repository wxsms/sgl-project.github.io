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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.29it/s]


    2026-04-29 03:16:09,633 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 03:16:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.09it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.20it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.20it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.20it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.20it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.20it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.20it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.20it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.20it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.20it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.20it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.88it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.88it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.88it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.88it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.88it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.88it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.88it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.88it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.88it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.88it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 14.88it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.55it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.55it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 31.08it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 31.08it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 31.08it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 31.08it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 31.08it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 31.08it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 31.08it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 31.08it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 20.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 20.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 20.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 20.83it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.16it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.55 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.90it/s] Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=832 avail_mem=72.53 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.90it/s]

    Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.45it/s]Capturing num tokens (num_tokens=640 avail_mem=72.52 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.45it/s]Capturing num tokens (num_tokens=576 avail_mem=72.52 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.45it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.45it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.45it/s]Capturing num tokens (num_tokens=448 avail_mem=72.52 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.45it/s]Capturing num tokens (num_tokens=448 avail_mem=72.52 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.76it/s]Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.76it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.76it/s]Capturing num tokens (num_tokens=352 avail_mem=72.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.76it/s]

    Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=256 avail_mem=72.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=240 avail_mem=72.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=192 avail_mem=72.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=192 avail_mem=72.49 GB):  71%|███████   | 41/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=176 avail_mem=72.48 GB):  71%|███████   | 41/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  71%|███████   | 41/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=144 avail_mem=72.48 GB):  71%|███████   | 41/58 [00:01<00:00, 42.67it/s]

    Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  71%|███████   | 41/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  71%|███████   | 41/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=96 avail_mem=72.47 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.21it/s] Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=32 avail_mem=72.44 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.21it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.44 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=28 avail_mem=72.43 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=24 avail_mem=72.43 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=20 avail_mem=72.42 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=16 avail_mem=72.42 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=16 avail_mem=72.42 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.80it/s]Capturing num tokens (num_tokens=12 avail_mem=71.92 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.80it/s]Capturing num tokens (num_tokens=8 avail_mem=71.92 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.80it/s] Capturing num tokens (num_tokens=4 avail_mem=71.83 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.80it/s]Capturing num tokens (num_tokens=4 avail_mem=71.83 GB): 100%|██████████| 58/58 [00:01<00:00, 34.11it/s]


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
    Generated text:  Mary, and I'm 17 years old. I live in the United States. I have a great school, and I study hard every day. I can speak French and I can sing. I am very popular. I like all kinds of things. What can you tell me about Mary? We will answer questions like this, "What can you tell me about Mary?" Mary is a young girl in the United States. She has a great school and studies hard every day. She can speak French and can sing. She is very popular. She likes all kinds of things. We will answer your questions like this. We ask:
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 32 years older than the president of Peru. In 15 years, the president of Peru will be 45 years older than the president of the United States. How old is the president of the United States?
    To determine the age of the president of the United States, we start by defining the variables and setting up the equations based on the given information.
    
    Let \( U \) represent the age of the president of the United States.
    Let \( P \) represent the age of the president of Peru.
    
    According to the problem, the president of the United States is currently 32 years older than the president of
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. London C. Moscow D. Berlin
    
    The capital of France is Paris.
    
    Paris is the largest and most populous city in France, located in the North West of the country on the Île de la Cité. It is the capital city of the department of French Guiana, and is home to the headquarters of the French embassy in the United States, the French Embassy in Canada and the French Consulate in Panama. The city is known as the "City of Love" due to its romantic architecture and vibrant nightlife.
    
    Therefore, the correct answer is:
    A. Paris
    You are an AI assistant that
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to be much more flexible and open to the influence of human creativity. How can AI developers incorporate human creativity into their work to create better AI systems? One way to incorporate human creativity into AI development is through the use of machine learning techniques that allow for the learning and adaptation of human expertise. This can be achieved by using natural language processing and machine learning algorithms to understand and interpret human language and behavior. By integrating machine learning algorithms into AI development, developers can create systems that are more responsive and adaptable to changing conditions, as well as more human-like in their interactions with humans. 
    
    Another way to incorporate human creativity into AI development is through


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] who has always been [Positive Trait]. I'm passionate about [What I Love to Do]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Positive Trait] person who is always [Positive Trait]. I'm a [Positive Trait] person who is always [Positive Trait]. I'm a [Positive Trait] person who is always [Positive Trait]. I'm a [Positive Trait] person who is always [Positive Trait]. I'm a [Positive Trait] person who is always [Positive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and festivals. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, diverse culture, and vibrant nightlife. It is the largest city in France and a major economic and political center in Europe. Paris is also home to the French Parliament, the French Academy of Sciences, and the French National Library. The city is known for its cuisine, including its famous
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, and it has the potential to revolutionize the field. AI-powered diagnostic tools, such as AI-powered X-rays and AI-powered MRI scans, are already being used in hospitals around the world.
    
    2. AI in finance: AI is already being used in finance to automate trading, fraud detection, and risk management. As AI technology continues to improve, we can expect to see even
    


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
    Generated text:  [Name], and I'm here to tell you the story of [Your Character's Name]. I'm a [Age], [Title] [Your Character's Name], and I'm passionate about [Why I'm passionate about what I do]. I've always been inspired by [Person/Event] and I'm determined to make a [Positive Impact/Big Difference] in the world. I believe that I'm capable of achieving my goals and that I have the talent and skills to help people who need me. Thank you for having me, [Name]. Let's get to know each other better! What do you think your character's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum, as well as its rich cultural history, including the famous Paris March. 
    
    (Note: Please ensure that the answer is based on factual information about Paris, not on personal opinions or cultural references.) France's capital city, Paris, is renowned for its iconic landmarks like Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum, as well as its rich cultural history, including the famous Paris March. Paris is a melting pot of France's diverse regions, known for its contemporary fashion, art, and cuisine
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and exciting, with potential to revolutionize industries, transform how we live, work, and interact with technology. Here are some possible trends in AI in the coming years:
    
    1. Increased Human-Centered AI: As AI becomes more advanced and complex, we are likely to see more AI that is designed with human-centered goals in mind. This could include developing AI that is designed to assist humans in areas such as healthcare, education, and transportation.
    
    2. More Robust Ethics and Privacy: As AI systems become more advanced, there will likely be increased scrutiny of their ethical and privacy implications. This could lead to more stringent regulations on


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

     first

     name

    ]

     and

     I

    'm

     a

     [

    insert

     profession

    ]

     in

     [

    insert

     location

    ]

     (

    city

    ,

     country

    ).

     I

    've

     been

     working

     with

     this

     company

     for

     [

    insert

     number

     of

     years

    ]

     years

    ,

     and

     I

     have

     a

     lot

     of

     experience

     in

     [

    insert

     relevant

     field

    ,

     such

     as

     [

    insert

     relevant

     skills

    ,

     such

     as

     [

    insert

     relevant

     experience

    ,

     such

     as

     [

    insert

     relevant

     projects

    ,

     such

     as

     [

    insert

     relevant

     achievements

    ,

     such

     as

     [

    insert

     notable

     accomplishment

    ]]

    ]

    "]).

     I

    'm

     passionate

     about

     [

    insert

     reason

     why

     you

     are

     passionate

     about

     this

     work

    ,

     such

     as

     [

    insert

     specific

     hobby

    ,

     like

     [

    insert

     specific

     interest

    ,

     like

     [

    insert

     specific

     career

     objective

    ,

     such

     as

     [

    insert

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     of

     France

     and

     the

     country

    's

     largest

     city

    ,

     located

     on

     the

     Mediterranean

     coast

    ,

     along

     the

     banks

     of

     the

     Se

    ine

     river

    .

     It

     is

     known

     for

     its

     famous

     landmarks

    ,

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

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

     The

     city

     is

     also

     known

     for

     its

     rich

     cultural

     and

     artistic

     heritage

    ,

     with

     numerous

     museums

    ,

     theaters

    ,

     and

     other

     cultural

     institutions

    .

     As

     a

     major

     economic

     and

     political

     center

    ,

     Paris

     has

     played

     an

     important

     role

     in

     French

     history

     and

     continues

     to

     be

     a

     major

     center

     of

     culture

     and

     art

     today

    .

     According

     to

     the

     

    2

    0

    2

    1

     census

    ,

     the

     population

     of

     Paris

     was

     

    2

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     dynamic

    ,

     and

     there

     are

     several

     possible

     trends

     that

     we

     can

     expect

     to

     see

     in

     the

     coming

     years

    .

     Here

     are

     some

     potential

     developments

     that

     are

     currently

     being

     considered

    :
    


    1

    .

     Adv

    ancements

     in

     AI

     technology

    :

     Over

     the

     next

     few

     decades

    ,

     we

     will

     continue

     to

     see

     significant

     advancements

     in

     AI

     technology

    ,

     including

     improvements

     in

     computer

     vision

    ,

     natural

     language

     processing

    ,

     and

     machine

     learning

    .
    


    2

    .

     Integration

     of

     AI

     into

     new

     industries

    :

     AI

     is

     already

     becoming

     an

     integral

     part

     of

     a

     variety

     of

     industries

    ,

     from

     healthcare

     to

     transportation

     to

     retail

    .

     We

     may

     see

     a

     greater

     integration

     of

     AI

     in

     future

     industries

     as

     AI

     becomes

     more

     integrated

     into

     all

     aspects

     of

     our

     lives

    .
    


    3

    .

     Increased

     focus

     on

     ethical

    



```python
llm.shutdown()
```
