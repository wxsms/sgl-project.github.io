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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.02it/s]


    2026-05-06 03:25:26,977 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 03:25:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.45it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.45it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.65 GB):   3%|▎         | 2/58 [00:00<00:03, 18.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.51it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.61 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.61 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.61 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=53.61 GB):  21%|██        | 12/58 [00:00<00:01, 29.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.60 GB):  21%|██        | 12/58 [00:00<00:01, 29.20it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.60 GB):  21%|██        | 12/58 [00:00<00:01, 29.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.60 GB):  21%|██        | 12/58 [00:00<00:01, 29.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.59 GB):  21%|██        | 12/58 [00:00<00:01, 29.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.59 GB):  21%|██        | 12/58 [00:00<00:01, 29.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.77it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=53.53 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.53 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=960 avail_mem=53.55 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.11it/s] Capturing num tokens (num_tokens=896 avail_mem=53.54 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=832 avail_mem=53.54 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=768 avail_mem=53.54 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=704 avail_mem=53.53 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=704 avail_mem=53.53 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.80it/s]Capturing num tokens (num_tokens=640 avail_mem=53.53 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.80it/s]Capturing num tokens (num_tokens=576 avail_mem=53.53 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.80it/s]Capturing num tokens (num_tokens=512 avail_mem=53.51 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.80it/s]

    Capturing num tokens (num_tokens=480 avail_mem=53.53 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.80it/s]Capturing num tokens (num_tokens=448 avail_mem=53.53 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.80it/s]Capturing num tokens (num_tokens=448 avail_mem=53.53 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=416 avail_mem=53.53 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=384 avail_mem=53.52 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=352 avail_mem=53.52 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=320 avail_mem=53.51 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=288 avail_mem=53.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=288 avail_mem=53.51 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.95it/s]Capturing num tokens (num_tokens=256 avail_mem=53.51 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.95it/s]Capturing num tokens (num_tokens=240 avail_mem=53.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.95it/s]Capturing num tokens (num_tokens=224 avail_mem=53.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.95it/s]

    Capturing num tokens (num_tokens=208 avail_mem=53.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.95it/s]Capturing num tokens (num_tokens=192 avail_mem=53.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.95it/s]Capturing num tokens (num_tokens=192 avail_mem=53.50 GB):  71%|███████   | 41/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=176 avail_mem=53.49 GB):  71%|███████   | 41/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=160 avail_mem=53.49 GB):  71%|███████   | 41/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=144 avail_mem=53.49 GB):  71%|███████   | 41/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=128 avail_mem=53.49 GB):  71%|███████   | 41/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=112 avail_mem=53.48 GB):  71%|███████   | 41/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=112 avail_mem=53.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=96 avail_mem=53.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.89it/s] Capturing num tokens (num_tokens=80 avail_mem=53.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.89it/s]

    Capturing num tokens (num_tokens=64 avail_mem=53.47 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=48 avail_mem=53.47 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=32 avail_mem=53.47 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=32 avail_mem=53.47 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=28 avail_mem=53.46 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=24 avail_mem=53.46 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=20 avail_mem=53.45 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=16 avail_mem=53.45 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=12 avail_mem=53.45 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=12 avail_mem=53.45 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.44it/s]Capturing num tokens (num_tokens=8 avail_mem=53.45 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.44it/s] Capturing num tokens (num_tokens=4 avail_mem=53.44 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.44it/s]

    Capturing num tokens (num_tokens=4 avail_mem=53.44 GB): 100%|██████████| 58/58 [00:01<00:00, 37.69it/s]


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
    Generated text:  Samantha and I'm a full-time professional in the field of customer service and have a background in IT support, marketing, and data analysis. Currently, I work as a project manager for a digital marketing agency, where I lead a team of 25 passionate and motivated individuals dedicated to driving our digital campaigns to their goals. I'm looking to expand my horizons and try a new career path, so I'm considering a travel-based option that would allow me to meet new people, gain experience, and work in a flexible environment. Can you provide me with some information on travel-based options and the requirements for a travel-based position? Additionally
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political leader, and the United States is a political nation. This passage could be described as:
    A. A narrative sentence
    B. A declarative sentence
    C. An interrogative sentence
    D. An exclamatory sentence
    Answer:
    B
    
    Please select the most appropriate option to fill in the blank:
    1. The reason for the benefits of this medicine is _________.
    2. You _________ the matter carefully.
    3. The boy is so _________ that he's always _________.
    4. I will write a letter to the _________.
    5. It's very important _________.
    6. He __________
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    Paris is a UNESCO World Heritage Site.
    Paris is located in the north of France.
    Paris is an international city and is a very important city.
    The city was founded in 789 CE by Charlemagne.
    The city is known for its art, architecture, and cuisine.
    You can get to Paris by train, plane, or ferry.
    Paris is located in the 10th and 11th Hours of the Day.
    Paris is famous for its coffee.
    Paris is the world’s 10th most populous city.
    Paris is the French capital city.
    Paris is located in the 10th hour
    ===============================
    Prompt: The future of AI is
    Generated text:  difficult to predict, but it's clear that the more we learn about it, the more we will want to understand it.
    As the idea of AI grows in scope and complexity, so too does the challenge of creating it. More than 100 years ago, a team of scientists created the first AI system. Since then, AI systems have been made more complex and the range of applications has expanded. In this talk, we will explore the early history of AI, its current state, and our current understanding of AI.
    The future of AI is difficult to predict, but it's clear that the more we learn about it, the


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm passionate about [What I love to do]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Personality] person who is [What you like to do]. I'm always eager to learn and improve myself. I'm [What you do for a living]. I'm [What you do for a living]. I'm [What you do for a living]. I'm [What you do for a living]. I'm [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is the largest city in France and the third largest in the world by population. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is also known for its fashion, art, and cuisine. The city is home to many famous landmarks and attractions, including the Notre-Dame Cathedral, the Louvre Museum, and the Champs-Élysées. Paris is a popular tourist destination and a major economic center in France. It is also known for its annual Eiffel Tower Festival
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection, risk
    


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
    Generated text:  [name] and I am a [insert profession or position]. I'm excited to meet you and I hope that we can help you achieve something great. What can I help you with? [insert a brief opening statement to grab the attention of the reader]. Remember to be polite and to the point in your interactions with me. Good luck! [insert any additional information or details that might be relevant to your character]. Good luck! [insert any additional information or details that might be relevant to your character]. Good luck! [insert any additional information or details that might be relevant to your character]. Good luck! [insert any additional information
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as Louvain in German.
    
    What key details about the city of Paris are provided in the given information? Paris is the capital of France. Paris is a historical and cultural city known for its architecture, museums, and world-renowned fashion. The city is also known for its annual Summer Olympics, which last from July to early August. Paris is also a popular tourist destination. The city has a rich history and is home to many historic landmarks such as the Eiffel Tower and Notre-Dame Cathedral. It is also known for its cuisine and is home to many famous French restaurants and bars. Paris is also a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be one of continued innovation and growth, driven by the increasing complexity of the technology and the increasing demand for advanced algorithms and machine learning. Here are some possible future trends in AI:
    
    1. Increased efficiency and accuracy: AI is likely to become even more efficient and accurate in its use of data and algorithms. This may lead to breakthroughs in areas such as medicine, transportation, and energy.
    
    2. Autonomous systems: Self-driving cars, drones, and other autonomous vehicles are likely to become more common in the future, leading to significant reductions in human errors and accidents.
    
    3. Personalized learning: AI is likely to become even more


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

    ].

     I

    'm

     a

     [

    insert

     job

     or

     occupation

    ]

     with

     over

     [

    insert

     number

    ]

     years

     of

     experience

     in

     the

     industry

    .

     I

     specialize

     in

     [

    insert

     key

     skills

     or

     areas

     of

     expertise

    ].

     I

     enjoy

     [

    insert

     one

     or

     two

     hobbies

     or

     interests

    ].

     I

    'm

     passionate

     about

     [

    insert

     a

     personal

     passion

     or

     cause

     you

     care

     about

    ].

     Overall

    ,

     I

    'm

     [

    insert

     a

     summary

     of

     your

     personality

     traits

     or

     strengths

    ]

     and

     always

     looking

     for

     opportunities

     to

     learn

     and

     grow

    .

     What

     about

     you

    ?


    Hello

     there

    !

     My

     name

     is

     [

    insert

     name

    ],

     and

     I

    'm

     a

     professional

     with

     over

     [

    insert

     number

    ]

     years

     of

     experience

     in

     the

     industry

    .

     I

     specialize

     in

     [

    insert

     key

     skills

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     the

     iconic

     city

     with

     its

     medieval

     architecture

     and

     towering

     E

    iff

    el

     Tower

    ,

     serves

     as

     the

     political

    ,

     cultural

    ,

     and

     economic

     heart

     of

     France

    .

     It

    's

     a

     vibrant

     met

    ropolis

     with

     a

     rich

     history

     and

     modern

     influences

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

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

     Lou

    vre

     Museum

    ,

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     also

     hosts

     numerous

     cultural

     festivals

    ,

     including

     the

     Festival

     de

     la

     dan

    se

    ,

     a

     massive

     parade

     celebrating

     the

     music

     of

     France

    .

     Despite

     its

     fame

    ,

     Paris

     has

     a

     peaceful

     and

     friendly

     atmosphere

    ,

     making

     it

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     Its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     there

     are

     several

     possible

     trends

     that

     are

     expected

     to

     shape

     the

     development

     of

     the

     technology

     in

     the

     coming

     years

    .

     Some

     of

     the

     most

     important

     trends

     include

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     is

     becoming

     increasingly

     integrated

     with

     other

     technologies

     such

     as

     machine

     learning

    ,

     computer

     vision

    ,

     natural

     language

     processing

    ,

     and

     robotics

    ,

     creating

     a

     more

     comprehensive

     and

     versatile

     system

    .
    


    2

    .

     Greater emphasis

     on

     ethical

     considerations

    :

     As

     AI

     systems

     become

     more

     complex

     and

     sensitive

    ,

     there

     is

     a

     growing

     emphasis

     on

     ethical

     considerations

    ,

     such

     as

     bias

    ,

     accountability

    ,

     and

     transparency

    .

     AI

     researchers

     are

     working

     to

     develop

     more

     robust

     and

     transparent

     systems

     that

     can

     be

     trusted

     to

     make

     decisions

     that

     are

     fair

     and

    



```python
llm.shutdown()
```
