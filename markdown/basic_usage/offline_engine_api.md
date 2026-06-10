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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.55it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.40it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.40it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.40it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.40it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.40it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.40it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.40it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.40it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.40it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.40it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 20.19it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 20.19it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 20.19it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 20.19it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 20.19it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 20.19it/s]

    Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 20.19it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 20.19it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 20.19it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 27.15it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 37.48it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 37.48it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 37.48it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 37.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.82it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.22 GB):   3%|▎         | 2/58 [00:00<00:03, 18.58it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.22 GB):   3%|▎         | 2/58 [00:00<00:03, 18.58it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.21 GB):   3%|▎         | 2/58 [00:00<00:03, 18.58it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.21 GB):   3%|▎         | 2/58 [00:00<00:03, 18.58it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.21 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.21 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.19 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.19 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.19 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.91it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.19 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.91it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.18 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.91it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.18 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.91it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.18 GB):  21%|██        | 12/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.18 GB):  21%|██        | 12/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.18 GB):  21%|██        | 12/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.17 GB):  21%|██        | 12/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.17 GB):  21%|██        | 12/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.17 GB):  21%|██        | 12/58 [00:00<00:01, 29.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.17 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.14 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=960 avail_mem=72.15 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.14it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=72.15 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.99it/s]Capturing num tokens (num_tokens=896 avail_mem=72.15 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.99it/s]Capturing num tokens (num_tokens=832 avail_mem=72.15 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.99it/s]Capturing num tokens (num_tokens=768 avail_mem=72.14 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.99it/s]Capturing num tokens (num_tokens=704 avail_mem=72.14 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.99it/s]Capturing num tokens (num_tokens=640 avail_mem=72.14 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.99it/s]Capturing num tokens (num_tokens=640 avail_mem=72.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.61it/s]Capturing num tokens (num_tokens=576 avail_mem=72.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.61it/s]Capturing num tokens (num_tokens=512 avail_mem=72.12 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.61it/s]Capturing num tokens (num_tokens=480 avail_mem=72.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.61it/s]Capturing num tokens (num_tokens=448 avail_mem=72.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.61it/s]Capturing num tokens (num_tokens=416 avail_mem=72.13 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.61it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.13 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=384 avail_mem=72.13 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=352 avail_mem=72.13 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=320 avail_mem=72.12 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=288 avail_mem=72.12 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=256 avail_mem=72.12 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.12it/s]Capturing num tokens (num_tokens=256 avail_mem=72.12 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.02it/s]Capturing num tokens (num_tokens=240 avail_mem=72.11 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.02it/s]Capturing num tokens (num_tokens=224 avail_mem=72.11 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=208 avail_mem=72.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=192 avail_mem=72.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=176 avail_mem=72.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.02it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.10 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=160 avail_mem=72.10 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=144 avail_mem=72.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=128 avail_mem=72.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=112 avail_mem=72.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=96 avail_mem=72.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.84it/s] Capturing num tokens (num_tokens=96 avail_mem=72.09 GB):  81%|████████  | 47/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=80 avail_mem=72.08 GB):  81%|████████  | 47/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=64 avail_mem=72.08 GB):  81%|████████  | 47/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=48 avail_mem=72.08 GB):  81%|████████  | 47/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=32 avail_mem=72.07 GB):  81%|████████  | 47/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=28 avail_mem=72.07 GB):  81%|████████  | 47/58 [00:01<00:00, 45.13it/s]

    Capturing num tokens (num_tokens=28 avail_mem=72.07 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=24 avail_mem=72.07 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=20 avail_mem=72.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=16 avail_mem=72.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=12 avail_mem=72.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=8 avail_mem=72.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.34it/s] Capturing num tokens (num_tokens=8 avail_mem=72.05 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=4 avail_mem=72.05 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=4 avail_mem=72.05 GB): 100%|██████████| 58/58 [00:01<00:00, 40.21it/s]


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
    Generated text:  Jack Smith and I'm an avid skier. I've been skiing for over 10 years now, and I've been training every day. I love skiing and I enjoy the sport very much. My name is Jack Smith and I'm an avid skier. I've been skiing for over 10 years now, and I've been training every day. I love skiing and I enjoy the sport very much. 
    
    Write a short paragraph about yourself. My name is Jack Smith and I'm an avid skier. I've been skiing for over 10 years now, and I've been training every day. I love
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing to create a new national park in the western part of the United States. The park would be a wilderness area that would protect the natural beauty of the area. The president wants to know the cost of creating the park. He wants to know the cost of the park, how much it would cost to create the park, how much it would cost to maintain the park, and how much it would cost to fund the park. Can you help him with this?
    Certainly! Creating a national park in the western part of the United States can be a complex endeavor that involves numerous costs. Here are the key expenses and costs associated with creating,
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris, which has a population of 2.1 million people. Given that the population of Paris is approximately 2.1 million, how many people live in Paris? How many people live in Paris?
    1. **Initial Population**: The population of Paris is given as 2,100,000 people.
    2. **Population Calculation**:
        - The problem states that the population of Paris is approximately 2,100,000.
        - Therefore, the population of Paris is exactly 2,100,000 people.
    3. **Answer**: The
    ===============================
    Prompt: The future of AI is
    Generated text:  in motion. Tech companies are finding their own path through the technical maze, and everyone is talking about what it will take to make a game of this. Will it be a fun game with things that will make you laugh or a game that will be a spectacle of numbers? We’ll take you on a journey through the new AI and the impact it will have on our world. Join us for a day of exploration into the future of AI and its impact on the future of games. What kind of impact will the new AI have on the gaming industry, and what are some of the new features and technologies that are expected to be introduced in


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for being in the industry], and I'm always looking for ways to [action or goal]. I'm a [reason for being a good fit for the company], and I'm excited to [reason for being a good fit for the company]. I'm a [reason for being a good fit for the company], and I'm looking forward to [reason for being a good fit for the company]. I'm a [reason for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in France and the second largest in the European Union. The city is located on the Seine River and is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its rich history, art, and culture, and is a major tourist destination. The city is also home to many important institutions, including the French Academy of Sciences and the French National Library. Paris is a vibrant and diverse city with a rich cultural and artistic heritage. It is a popular destination for tourists and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn and adapt to new situations. This could lead to more complex and sophisticated AI systems that can perform tasks that require human-like intelligence.
    
    2. Enhanced privacy and security: As AI systems become more advanced, there will be an increased focus on privacy and security. This could lead to more stringent regulations and standards for AI systems, as well as greater emphasis on data privacy and security.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve diagnosis,
    


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
    Generated text:  [Name] and I'm a [Job Title] at [Company Name]. I'm passionate about [Personal Interest], [Describe something specific about yourself that makes you stand out].
    As an AI language model, I don't have personal interests or emotions, but I can generate a short, neutral self-introduction for a fictional character based on your requirements. Here's an example:
    Hello, my name is [Name] and I'm a [Job Title] at [Company Name]. I'm passionate about [Personal Interest], [Describe something specific about yourself that makes you stand out].
    This self-introduction would include the character's name,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known as the city of love. It is the largest and most populous city in France and is home to a number of world-renowned landmarks such as the Eiffel Tower, Louvre Museum, and Notre Dame Cathedral. French cuisine is also famous for its rich and complex flavors, and the city is a popular destination for tourism and cultural events. Despite the challenges of climate change and poverty, Paris is a vibrant and dynamic city that continues to thrive. 
    
    The answer to the question is: Paris is the capital city of France, known as the city of love, and is a large, populous city with a rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and difficult to predict, but there are a number of trends that are likely to shape the field in the coming years.
    
    One of the most significant trends is the increasing use of AI in areas such as healthcare, finance, and marketing. In healthcare, AI is already being used to automate routine tasks and improve patient outcomes. In finance, AI is being used to detect fraud and identify potential risks in the financial market. In marketing, AI is being used to personalize advertising and improve customer engagement.
    
    Another trend is the development of more sophisticated machine learning algorithms that can perform complex tasks more accurately and efficiently than traditional methods. This could lead to


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

    ],

     and

     I

     am

     a

     writer

    .

     I

     currently

     work

     as

     a

     freelance

     writer

     for

     a

     small

     online

     writing

     agency

     called

     [

    Writing

     Agency

     Name

    ].

     I

     specialize

     in

     writing

     novels

    ,

     short

     stories

    ,

     and

     creative

     non

    -fiction

    ,

     and

     I

     have

     a

     passion

     for

     exploring

     the

     human

     experience

     through

     storytelling

    .

     I

     believe

     in

     the

     power

     of

     words

     to

     make

     people

     feel

    ,

     and

     I

     strive

     to

     write

     in

     a

     way

     that

     reson

    ates

     with

     readers

     and

     inspires

     them

     to

     connect

     with

     the

     world

     around

     them

    .

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     as

     a

     writer

    ,

     and

     I

     am

     eager

     to

     share

     my

     work

     with

     you

     all

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

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     city

     that

     is

     famous

     for

     its

     rich

     history

     and

     culture

    .

     It

     is

     home

     to

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

     many

     other

     landmarks

     that

     showcase

     France

    's

     art

    ,

     architecture

    ,

     and

     history

    .

     Paris

     is

     a

     bustling

     city

     with

     a

     diverse

     population

     that

     is

     known

     for

     its

     fashion

    ,

     food

    ,

     and

     music

    .

     The

     city

     is

     also

     known

     for

     its

     annual

     Carn

    aval

    ,

     a

     traditional

     festival

     that

     takes

     place

     in

     the

     summer

    .

     Overall

    ,

     Paris

     is

     a

     city

     that

     is

     a

     true

     reflection

     of

     France

    's

     cultural

     and

     political

     heritage

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     profoundly

     transformative

    ,

     resh

    aping

     industries

    ,

     economies

    ,

     and

     societies

     in

     ways

     unseen

     before

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     AI

     Ethics

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     lives

    ,

     ethical

     considerations

     will

     become

     increasingly

     important

    .

     This

     will

     include

     ensuring

     that

     AI

     systems

     are

     transparent

    ,

     accountable

    ,

     and

     balanced

    .
    


    2

    .

     Multi

    -

    Modal

     AI

    :

     AI

     will

     become

     more

     capable

     of

     understanding

     and

     processing

     a

     wider

     range

     of

     data

     sources

    ,

     from

     text

    ,

     speech

    ,

     and

     images

     to

     social

     media

     and

     other

     forms

     of

     media

    .
    


    3

    .

     AI

     for

     Health

    :

     AI

     will

     play

     a

     key

     role

     in

     improving

     and

     developing

     new

     treatments

     for

     diseases

    ,

     such

     as

     cancer

    



```python
llm.shutdown()
```
