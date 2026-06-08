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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.02it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.02it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.02it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.02it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.02it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.02it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.02it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.02it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.02it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.02it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.53it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.53it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.53it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.53it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.53it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.53it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.53it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.53it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.53it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.35it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.35it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.35it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.35it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.35it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.35it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.35it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.35it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.35it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.35it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.22it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 38.11it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 38.11it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.36 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.36 GB):   3%|▎         | 2/58 [00:00<00:03, 14.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=70.35 GB):   3%|▎         | 2/58 [00:00<00:03, 14.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.35 GB):   3%|▎         | 2/58 [00:00<00:03, 14.87it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=70.35 GB):   3%|▎         | 2/58 [00:00<00:03, 14.87it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.35 GB):   9%|▊         | 5/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.35 GB):   9%|▊         | 5/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.34 GB):   9%|▊         | 5/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.33 GB):   9%|▊         | 5/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.33 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.33 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.37it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=70.33 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.32 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.31 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.31 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.31 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.04it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=70.31 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.30 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.30 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.30 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.30 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.19it/s]Capturing num tokens (num_tokens=960 avail_mem=70.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.19it/s] Capturing num tokens (num_tokens=896 avail_mem=70.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.19it/s]

    Capturing num tokens (num_tokens=896 avail_mem=70.29 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=832 avail_mem=70.29 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=768 avail_mem=70.28 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=704 avail_mem=70.28 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=640 avail_mem=70.28 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=576 avail_mem=70.28 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.36it/s]Capturing num tokens (num_tokens=576 avail_mem=70.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.78it/s]Capturing num tokens (num_tokens=512 avail_mem=70.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.78it/s]Capturing num tokens (num_tokens=480 avail_mem=70.28 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=448 avail_mem=70.27 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=416 avail_mem=70.27 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.78it/s]

    Capturing num tokens (num_tokens=416 avail_mem=70.27 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.80it/s]Capturing num tokens (num_tokens=384 avail_mem=70.27 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.80it/s]Capturing num tokens (num_tokens=352 avail_mem=70.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.80it/s]Capturing num tokens (num_tokens=320 avail_mem=70.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.80it/s]Capturing num tokens (num_tokens=288 avail_mem=70.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.80it/s]Capturing num tokens (num_tokens=288 avail_mem=70.26 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=256 avail_mem=70.25 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=240 avail_mem=70.25 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=224 avail_mem=70.25 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=208 avail_mem=70.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.54it/s]

    Capturing num tokens (num_tokens=192 avail_mem=70.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=192 avail_mem=70.24 GB):  71%|███████   | 41/58 [00:01<00:00, 37.59it/s]Capturing num tokens (num_tokens=176 avail_mem=70.24 GB):  71%|███████   | 41/58 [00:01<00:00, 37.59it/s]Capturing num tokens (num_tokens=160 avail_mem=70.24 GB):  71%|███████   | 41/58 [00:01<00:00, 37.59it/s]Capturing num tokens (num_tokens=144 avail_mem=70.23 GB):  71%|███████   | 41/58 [00:01<00:00, 37.59it/s]Capturing num tokens (num_tokens=128 avail_mem=70.23 GB):  71%|███████   | 41/58 [00:01<00:00, 37.59it/s]Capturing num tokens (num_tokens=128 avail_mem=70.23 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=112 avail_mem=70.23 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=96 avail_mem=70.23 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.20it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=70.22 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=64 avail_mem=70.22 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=64 avail_mem=70.22 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.72it/s]Capturing num tokens (num_tokens=48 avail_mem=70.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.72it/s]Capturing num tokens (num_tokens=32 avail_mem=70.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.72it/s]Capturing num tokens (num_tokens=28 avail_mem=70.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.72it/s]Capturing num tokens (num_tokens=24 avail_mem=70.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.72it/s]Capturing num tokens (num_tokens=24 avail_mem=70.20 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=20 avail_mem=70.20 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.34it/s]

    Capturing num tokens (num_tokens=16 avail_mem=70.20 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=12 avail_mem=70.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=8 avail_mem=70.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.34it/s] Capturing num tokens (num_tokens=4 avail_mem=70.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=4 avail_mem=70.19 GB): 100%|██████████| 58/58 [00:01<00:00, 37.10it/s]Capturing num tokens (num_tokens=4 avail_mem=70.19 GB): 100%|██████████| 58/58 [00:01<00:00, 32.24it/s]


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
    Generated text:  Robert. I am a 16-year-old boy. I like to play basketball. I want to become a basketball player one day. I want to play basketball well. I want to play basketball for my team. I don't like playing games with my friends. When I play basketball, I will work hard to improve my basketball skills. When I have a basketball game, I will try to win. I want to play basketball with my parents. They love me. Now, I am 16 years old. I want to play basketball with my parents. I will make my parents happy. They will also like me more.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. According to a recent presidential election, what does this indicate about the political landscape in the United States? A presidential election results in a winner of the presidency. In the United States, the person who wins a presidential election is the person who has received the most votes, and the person who has received the most votes is the candidate who is leading in the polls. A winner of the presidency is someone who has won a term of office, typically for a two-year term.
    In the United States, presidential elections are held every four years, and the winner of the election is generally expected to serve for a term of office. This
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. In a geography class, the teacher asks the students to model the distance between two cities on a map. Each city is represented as a point, and the distance between them is measured as the straight-line distance on the map. The teacher asks the students to find the shortest path between two cities, A and B, on a map that is a rectangle. The vertices of the rectangle are located at points (0, 0), (10, 0), (10, 5), and (0, 5). The shortest path between cities A and B is to move directly across the rectangle. 
    
    Calculate the
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting, but the technology is still very young and very diverse. There are many different areas of AI research, from algorithms to machine learning to computer vision, and all of them have the potential to revolutionize the way we live and work. However, one area that is still relatively underexplored is the potential of AI to make health care more accessible and affordable.
    
    AI has the potential to transform the healthcare industry by providing more personalized and accurate diagnoses and treatments, while also reducing the time and cost associated with traditional medical practices. AI algorithms can analyze vast amounts of data on patients, identify patterns and trends, and provide insights that can help doctors


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


    Generated text:  Paris, also known as the City of Light, and is the largest city in the European Union. It is located on the Seine River and is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its rich history, art, and culture, and is a popular tourist destination. The city is also home to many famous landmarks and attractions, including the Louvre Museum, the Champs-Élysées, and the Arc de Triomphe. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage. It is a major economic center and a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This will require a greater understanding of human emotions, motivations, and cognitive processes.
    
    
    


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
    Generated text:  [Your Name], and I am an avid reader and traveler. I have traveled extensively across the globe, and I have a deep appreciation for the variety of cultures and experiences that I have encountered. I am passionate about exploring new places and experiencing the world in a meaningful way. I am always eager to learn and expand my horizons. I look forward to continuing to grow and develop as a person. What is your favorite book or movie? As an AI language model, I do not have personal preferences or emotions. However, I am programmed to provide information and answer questions related to the topics I am trained on, such as literature, history
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is located in the Île-de-France region and is the largest city and most populous metropolitan area in Europe. The city is known for its rich history, cultural diversity, and significant contributions to French and European art, literature, and science. It is also one of the most cosmopolitan cities in the world, with many international institutions and landmarks in the city center. Paris has a rich heritage, including contributions from various civilizations and cultures, and is considered the epitome of European culture and art. Its status as the capital of France is recognized internationally, and the city has a strong sense of national identity and pride. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, as it is continually evolving and changing. However, some of the trends that are likely to shape the future of AI include:
    
    1. Increased Integration with Other Technologies: AI is becoming more integrated with other technologies, such as the internet of things (IoT) and blockchain, to create a more seamless and efficient experience for users.
    
    2. Development of Advanced AI Models: As AI models become more complex and sophisticated, we can expect to see more advanced and accurate models that can perform complex tasks with higher accuracy and speed.
    
    3. Human-Centered AI: AI is becoming increasingly human-centered, with the goal of creating AI that


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

     [

    Age

    ],

     [

    Occup

    ation

    ].

     I

    'm

     [

    Short

     Biography

    ]

     and

     I

    'm

     here

     to

     share

     my

     thoughts

     and

     experiences

    .

     I

    'm

     always

     ready

     to

     learn

    ,

     and

     I

    'm

     always

     looking

     for

     new

     challenges

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?


    Hello

    ,

     my

     name

     is

     [

    Name

    ]

     and

     I

    'm

     a

     [

    Age

    ],

     [

    Occup

    ation

    ].

     I

    'm

     here

     to

     share

     my

     thoughts

     and

     experiences

    .

     I

    'm

     always

     ready

     to

     learn

    ,

     and

     I

    'm

     always

     looking

     for

     new

     challenges

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    Fill

     in

     the

     blank

     with

     relevant

     information

    ].

     Well

    ,

     that

    's

     great

     to

     hear

    .

     I

    'm

     glad

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     and

     largest

     city

     of

     France

    .

     It

     is

     also

     the

     seat

     of

     government

    ,

     the

     seat

     of

     state

     institutions

    ,

     and

     the

     cultural

    ,

     political

    ,

     and

     commercial

     center

     of

     the

     country

    .

     The

     city

     is

     a

     major

     financial

     and

     economic

     hub

    ,

     with

     a

     long

     history

     of

     developing

     into

     an

     international

     met

    ropolis

    .

     Paris

     is

     known

     for

     its

     iconic

     landmarks

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Mus

    ée

     de

     la

     Machine

    .
    


    In

     recent

     years

    ,

     Paris

     has

     become

     a

     major

     global

     city

     and

     is

     home

     to

     numerous

     important

     international

     institutions

    ,

     including

     UNESCO

    ,

     the

     French

     Academy

     of

     Sciences

    ,

     and

     the

     World

     Trade

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     continued

     advancements

     in

     areas

     such

     as

     self

    -driving

     cars

    ,

     artificial

     intelligence

     in

     healthcare

    ,

     and

     the

     integration

     of

     AI

     into

     everyday

     life

    .

     However

    ,

     we

     will

     also

     see

     the

     potential

     for

     AI

     to

     be

     used

     for

     unethical

     and

     potentially

     harmful

     purposes

    ,

     including

     the

     proliferation

     of

     surveillance

    ,

     the

     use

     of

     AI

     in

     decision

    -making

     and

     discrimination

    ,

     and

     the

     promotion

     of

     biased

     algorithms

    .

     Additionally

    ,

     there

     will

     be

     ongoing

     efforts

     to

     develop

     better

     AI

     algorithms

     and

     technologies

    ,

     as

     well

     as

     to

     create

     regulations

     and

     standards

     to

     ensure

     the

     responsible

     use

     of

     AI

    .
    


    As

     AI

     becomes

     more

     advanced

     and

     integrated

     into

     our

     lives

    ,

     we

     will

     need

     to

     carefully

     consider

     the

     ethical

     implications

     of

     its

     use

     and

     how

     it

     should

    



```python
llm.shutdown()
```
