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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.45it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.45it/s]


    2026-05-18 01:17:54,115 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 01:17:54] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.57it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.57it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.57it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.57it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.57it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.57it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.57it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.57it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.57it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.57it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.57it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 18.10it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.75 GB):   7%|▋         | 4/58 [00:00<00:03, 13.76it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.75 GB):   7%|▋         | 4/58 [00:00<00:03, 13.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.25 GB):   7%|▋         | 4/58 [00:00<00:03, 13.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.25 GB):   7%|▋         | 4/58 [00:00<00:03, 13.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.25 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.24 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.29it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.72it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.72it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.68it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.68it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.68it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.82it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.31it/s]

    Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.34it/s]

    Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 45.07it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 45.07it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 45.07it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 45.07it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 45.07it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 45.07it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.94it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.94it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.94it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.94it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.94it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.94it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.17it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.17it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.17it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.66it/s]


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
    Generated text:  Abigaelt. I have a background in Marketing & Communications, and in the last few years have had the opportunity to work with a variety of clients, from high-profile brands to smaller independent brands and startups. My experience spans from strategy and marketing management to brand management and social media content creation, and I am always looking to learn about new platforms and opportunities.
    I have a passion for helping businesses in a variety of industries, from startups to established brands, to achieve their marketing goals. I have a strong understanding of the marketing funnel and the different stages that businesses can go through during the process of reaching their audience.
    What is the main focus
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. Most people would not be surprised to learn that the president of the United States is the leader of the federal government of the United States. The president of the United States has the power to make laws, appoint judges and members of Congress, and dissolve the Congress and the Executive Branch. When the president is not in office, he is a consultant to the president. The president's job is very important because the president has a great deal of power and often uses his power to try to influence the decisions of the federal government. The president has a lot of other important jobs that he does. He is often called upon to
    ===============================
    Prompt: The capital of France is
    Generated text:  a city and an administrative centre. It has a great economy and a large population, and it is considered a major city in the country. The capital of France is Paris. The city's name comes from the Latin word 'parvis', which means 'front door'. Paris is also known as the 'City of the Hills'. It is famous for its museums, landmarks, and theaters. Paris is one of the most visited cities in the world, and it attracts visitors from all over the world. It is also one of the most important cities in France, and it is considered a major city in the country. The capital of France is
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about the technology itself, but also about the ethics and impact on society. As artificial intelligence becomes more integrated into our daily lives, it is essential to consider the implications and responsibilities it may have on different groups of people.
    One of the key ethical concerns is the potential for AI to perpetuate and amplify inequality. As AI systems become more sophisticated, they are likely to be used in situations where people are at a disadvantage, such as in the workplace or in accessing services. If these systems are designed and programmed to perpetuate inequalities, then it could be unfair to all of us.
    Another ethical issue is the potential for AI to be


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and festivals. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, diverse culture, and vibrant nightlife. It is a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This will lead to the creation of more efficient and productive machines that can perform tasks that were previously done by humans.
    
    2. AI ethics and privacy: As AI technology becomes more advanced, there will be a need to address the ethical and privacy concerns associated with it. This will
    


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
    Generated text:  [Name], and I'm a computer scientist. I specialize in designing and building efficient algorithms for complex problems, and I enjoy working with humans to solve challenging issues. I'm passionate about learning new technologies and staying up-to-date with the latest advancements in computer science. I'm constantly looking for ways to improve my skills and contribute to the field. What's your name and what kind of work are you currently doing? [Optional - Feel free to share any interesting facts or hobbies you have] Welcome, my friend. It's nice to meet you. I'm [Name], a computer scientist who specializes in designing and building efficient algorithms for complex
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, one of the most beautiful and populous cities in the world. It is located on the River Seine and is the largest city in France. Paris is known for its art, culture, and cuisine, and is a popular tourist destination. The city is also home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and numerous other landmarks.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating and it is difficult to predict with certainty, but here are a few possible trends that could be explored:
    
    1. Integration of AI into all aspects of life: With the increasing use of AI in various sectors, it is likely that we will see a greater integration of AI into all aspects of life. This could include healthcare, finance, transportation, and more.
    
    2. Personalization and automation: AI is already capable of learning and improving on its own, and with the development of new technologies, this will become even more effective. Personalization will become a more important factor, as AI will be able to learn and adapt to the individual


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

     your

     name

    ]

     and

     I

    'm

     a

     [

    insert

     your

     profession

     or

     title

    ]

     at

     [

    insert

     your

     company

     name

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     about

     you

     as

     a

     potential

     customer

    .

     Can

     you

     tell

     me

     a

     little

     bit

     about

     yourself

    ?

     Certainly

    !

     I

    'm

     a

     [

    insert

     your

     role

     or

     expertise

    ]

     at

     [

    insert

     your

     company

     name

    ],

     and

     I

    'm

     always

     eager

     to

     learn

     more

     about

     your

     needs

     and

     goals

    .

     Can

     you

     tell

     me

     a

     bit

     about

     your

     company

     and

     what

     you

     do

     for

     a

     living

    ?

     Sure

    !

     I

     am

     a

     [

    insert

     your

     role

     or

     expertise

    ]

     at

     [

    insert

     your

     company

     name

    ],

     and

     I

     work

     to

     help

     businesses

     like

     yours

     succeed

     by

     providing

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    [

    Mark

     down

    ]


    The

     capital

     of

     France

     is

     Paris

    .

     This

     monumental

     city

     has

     a

     rich

     history

     dating

     back

     to

     Roman

     times

    ,

     and

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

     E

    iff

    el

     Tower

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     Paris

     is

     also

     home

     to

     the

     French

     Parliament

     and

     the

     E

    iff

    el

     Tower

    ,

     and

     is

     a

     major

     transportation

     hub

     for

     Europe

     and

     the

     world

    .

     Its

     vibrant

     culture

    ,

     delicious

     cuisine

    ,

     and

     stunning

     architecture

     make

     it

     a

     beloved

     city

     for

     many

     people

    ,

     and

     remains

     a

     significant

     economic

     and

     cultural

     center

     of

     France

    .

     The

     city

     is

     also

     home

     to

     many

     prestigious

     museums

    ,

     art

     galleries

    ,

     and

     festivals

    ,

     making

     it

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     subject

     to

     significant

     changes

     over

     time

    ,

     but

     there

     are

     several

     potential

     trends

     that

     seem

     likely

    :
    


    1

    .

     Increased

     automation

     and

     efficiency

    :

     As

     AI

     technologies

     continue

     to

     improve

    ,

     more

     tasks

     will

     be

     automated

    ,

     freeing

     up

     time

     for

     humans

     to

     focus

     on

     more

     complex

     and

     creative

     activities

    .

     This

     could

     lead

     to

     greater

     efficiency

     and

     productivity

     in

     many

     industries

    .
    


    2

    .

     Improved

     natural

     language

     processing

    :

     With

     the

     development

     of

     more

     advanced

     AI

     models

    ,

     it

     is

     possible

     that

     natural

     language

     processing

     will

     become

     even

     more

     sophisticated

    .

     This

     could

     lead

     to

     more

     natural

     and

     intuitive

     interactions

     with

     machines

     and

     humans

     alike

    ,

     which

     could

     have

     significant

     implications

     for

     the

     way

     we

     communicate

     and

     interact

     with

     technology

    .
    


    3

    .

     AI

    



```python
llm.shutdown()
```
