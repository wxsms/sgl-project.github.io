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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.00it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:45,  3.96s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:45,  3.96s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:45,  3.96s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:45,  3.96s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:45,  3.96s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.84it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.83it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 23.83it/s]

    Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 23.83it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 33.64it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 33.64it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 33.64it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 33.64it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 33.64it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 33.64it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 33.64it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 33.64it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 33.64it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 33.64it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.47 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.44 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.43 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.43 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.43 GB):   3%|▎         | 2/58 [00:00<00:02, 19.05it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.43 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.42 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.14 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.42 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.42 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.42 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.53it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.42 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.41 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.41 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.53it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.41 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.53it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.40 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.53it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.40 GB):  31%|███       | 18/58 [00:00<00:01, 37.02it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.40 GB):  31%|███       | 18/58 [00:00<00:01, 37.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.40 GB):  31%|███       | 18/58 [00:00<00:01, 37.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.38 GB):  31%|███       | 18/58 [00:00<00:01, 37.02it/s]Capturing num tokens (num_tokens=960 avail_mem=74.39 GB):  31%|███       | 18/58 [00:00<00:01, 37.02it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.39 GB):  31%|███       | 18/58 [00:00<00:01, 37.02it/s]Capturing num tokens (num_tokens=896 avail_mem=74.39 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.52it/s]Capturing num tokens (num_tokens=832 avail_mem=74.39 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.52it/s]Capturing num tokens (num_tokens=768 avail_mem=74.38 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.52it/s]Capturing num tokens (num_tokens=704 avail_mem=74.38 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.52it/s]Capturing num tokens (num_tokens=640 avail_mem=74.38 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.52it/s]Capturing num tokens (num_tokens=576 avail_mem=74.38 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.52it/s]Capturing num tokens (num_tokens=576 avail_mem=74.38 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=512 avail_mem=74.36 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=480 avail_mem=74.38 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=416 avail_mem=73.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.16it/s]

    Capturing num tokens (num_tokens=384 avail_mem=73.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=384 avail_mem=73.95 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=352 avail_mem=73.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=208 avail_mem=73.92 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=192 avail_mem=73.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.01it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.01it/s]

    Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.01it/s]Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=144 avail_mem=73.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=128 avail_mem=73.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.86it/s] Capturing num tokens (num_tokens=80 avail_mem=73.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=80 avail_mem=73.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.07it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.07it/s]Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.07it/s]Capturing num tokens (num_tokens=32 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.07it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.07it/s]

    Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.07it/s]Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=16 avail_mem=73.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=8 avail_mem=73.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s] Capturing num tokens (num_tokens=4 avail_mem=73.86 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 41.87it/s]


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
    Generated text:  Kim and I'm a Business Student at Sydney University. I have a passion for writing about big ideas and how they are transforming the way we live and work. I love writing about new and exciting innovations that will change the world for the better in the near future. I write about things I'm passionate about like AI, robotics, sustainable energy, and so much more. I'm also a long time fan of Harry Potter. How can I better discuss the future of AI and robotics in my writing? In your opinion, what are the most exciting and transformative innovations that are currently shaping the future of technology? Additionally, what are the potential risks
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person who is the leader of the United States. He or she is elected by the people of the United States for a term of five years. He or she is the head of government and the highest decision maker of the country. The president is the commander-in-chief of the armed forces. The president has the power to issue executive orders. The president's most important duty is to protect the nation's unity and stability. The president may make decisions on foreign policy, defense, and international relations. The president may appoint ambassadors, ambassadors and other high-ranking officials of foreign countries. The president may also appoint generals and other high-ranking officials.
    ===============================
    Prompt: The capital of France is
    Generated text: : Paris
    
    The capital of France is: Paris. 
    
    To explain in simpler terms: Paris is like the main city in a big country called France. It's the most important place in the whole country, kind of like how your house is the most important place in your neighborhood. Paris is also the capital, which means it's the big, important city that helps run the country. It's the home of the president of France, who's like the boss of the whole country. Paris is a very famous and important place!
    ===============================
    Prompt: The future of AI is
    Generated text:  here, but the future of AI in the UK is still largely in its infancy. The UK is seen as a leader in the AI field, with some of the world’s leading AI researchers and institutions being based in the UK. However, the AI sector in the UK is still relatively new, and the industry is still developing rapidly. There is still a lot of uncertainty surrounding the future of AI in the UK, and this uncertainty is compounded by the fact that many of the AI researchers and institutions in the UK are based in the UK, rather than in the United States, as is the case for many countries in the U.S. In


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" or "La Ville Blanche" (White City). It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major cultural and economic center, with a rich history and a diverse population. The city is known for its art, music, and cuisine, and is a popular tourist destination. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there will be an increased focus on ethical AI. This will include issues such as bias, transparency, accountability, and the impact of AI on society.
    
    2. More integration with human decision-making: AI is likely to become more integrated with human decision-making, allowing for more complex and nuanced decision-making. This will require a greater understanding of human emotions and motivations.
    
    3. Greater use of AI in healthcare
    


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
    Generated text:  [Name], and I am [Age]. I am a [Type of Person] who is [Brief Description of Character]. I have [Number of Unique Skills] unique skills that make me stand out from the crowd. My biggest strength is my [Strength]. I am an [Occupation] who is [Job Title or Summary of What I Do]. I am passionate about [My Favorite Activity or Hobby]. What kind of personality do you have? What are some of your most memorable moments or experiences?
    My personality is [Short Answer]. I am [Majestically Serene]. I have an incredibly open heart and a deep desire
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    I apologize, but I cannot provide an answer to your question as it is not factual. France's capital city is Paris, but it is not a city in itself. Is there a specific topic or question you would like me to assist you with related to Paris? Please let me know. I'll do my best to assist you. 
    
    Paris is the capital city of France and one of the most important cities in the world. It is known for its rich history, beautiful architecture, and famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several trends:
    
    1. Increased AI Integration: AI will continue to integrate more deeply into our daily lives. We will see increased integration with other technologies such as the Internet of Things, 5G networks, and blockchain. This integration will lead to more complex and sophisticated AI systems that can perform tasks that we currently consider to be beyond the reach of AI.
    
    2. AI will become more human-like: As AI becomes more sophisticated, it will become increasingly human-like. This could lead to more intuitive and efficient AI systems that can perform tasks that were previously considered to be intractable or impossible.
    
    3. AI will


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

     a

     friendly

    ,

     helpful

    ,

     and

     patient

     person

    .

     I

     enjoy

     spending

     my

     free

     time

     with

     friends

     and

     family

     and

     I

     love

     trying

     new

     foods

    .

     I

     have

     a

     love

     for

     history

    ,

     and

     I

     enjoy

     learning

     about

     different

     cultures

     and

     their

     customs

    .

     I

    'm

     always

     looking

     for

     new

     experiences

     and

     adventures

    ,

     and

     I

    'm

     excited

     to

     meet

     new

     people

     and

     make

     new

     friends

    .

     If

     you

     ever

     need

     someone

     to

     talk

     to

     or

     someone

     to

     listen

    ,

     I

    'm

     here

     for

     you

    .

     [

    Name

    ]

     [

    Phone

     Number

    ]

     [

    Email

     Address

    ]

     [

    Website

     URL

    ]

     [

    Social

     Media

     Links

    ]

     [

    Career

     Goals

    ]

     Hi

     there

    !

     I

    'm

     [

    Name

    ],

     a

     friendly

    ,

     helpful

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Mediterranean

     coast

    ,

     and

     is

     a

     significant

     cultural

     and

     historical

     center

    .

     It

     is

     known

     for

     its

     vibrant

     arts

     and

     entertainment

     scene

    ,

     beautiful

     beaches

    ,

     and

     iconic

     landmarks

     such

     as

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

     home

     to

     the

     European

     Parliament

     and

     the

     French

     national

     museum

    ,

     as

     well

     as

     its

     major

     international

     institutions

     and

     attractions

     such

     as

     the

     Lou

    vre

     Museum

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     Paris

     is

     one

     of

     the

     world

    's

     most

     visited

     cities

     and

     a

     major

     tourist

     destination

    .

     Its

     diverse

     architecture

    ,

     artistic

     culture

    ,

     and

     historical

     significance

     make

     it

     a

     must

    -

    visit

     for

     anyone

     interested

     in

     French

     culture

     and

     history

    .

     [

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     number

     of

     different

     trends

     that

     will

     continue

     to

     shape

     the

     way

     we

     interact

     with

     technology

     and

     make

     decisions

    .

     Some

     of

     the

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     diagnose

     and

     treat

     diseases

    ,

     and

     the

     technology

     is

     expected

     to

     continue

     to

     advance

     in

     this

     area

    .
    


    2

    .

     Enhanced

     machine

     learning

    :

     With

     the

     increasing

     availability

     of

     data

     and

     the

     development

     of

     new

     algorithms

    ,

     machine

     learning

     will

     become

     more

     sophisticated

    ,

     allowing

     for

     even

     more

     accurate

     predictions

     and

     more

     personalized

     outcomes

    .
    


    3

    .

     AI

     for

     education

    :

     AI

     will

     continue

     to

     be

     used

     in

     education

     to

     personalize

     learning

     experiences

    ,

     increase

     accessibility

    



```python
llm.shutdown()
```
