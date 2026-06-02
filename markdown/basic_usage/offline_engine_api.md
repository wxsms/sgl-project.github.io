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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.17it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.26it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.12it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.12it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.12it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.12it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.12it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.12it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.12it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.12it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.12it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.12it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.51it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=68.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.67 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=68.67 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]Capturing num tokens (num_tokens=6656 avail_mem=68.67 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=68.67 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=68.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=68.66 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=68.65 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.65 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.65 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.65 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=68.64 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.95it/s]Capturing num tokens (num_tokens=3584 avail_mem=68.64 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.95it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=68.64 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.95it/s]Capturing num tokens (num_tokens=3072 avail_mem=68.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.95it/s]Capturing num tokens (num_tokens=3072 avail_mem=68.63 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=68.63 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=68.63 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=68.62 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.32it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=68.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.61 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.61 GB):  33%|███▎      | 19/58 [00:00<00:01, 23.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=68.61 GB):  33%|███▎      | 19/58 [00:00<00:01, 23.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=68.59 GB):  33%|███▎      | 19/58 [00:00<00:01, 23.14it/s]Capturing num tokens (num_tokens=960 avail_mem=68.61 GB):  33%|███▎      | 19/58 [00:00<00:01, 23.14it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=68.60 GB):  33%|███▎      | 19/58 [00:00<00:01, 23.14it/s]Capturing num tokens (num_tokens=896 avail_mem=68.60 GB):  40%|███▉      | 23/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=832 avail_mem=68.60 GB):  40%|███▉      | 23/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=768 avail_mem=68.60 GB):  40%|███▉      | 23/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=704 avail_mem=68.59 GB):  40%|███▉      | 23/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=640 avail_mem=68.59 GB):  40%|███▉      | 23/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=640 avail_mem=68.59 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.93it/s]Capturing num tokens (num_tokens=576 avail_mem=68.59 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.93it/s]Capturing num tokens (num_tokens=512 avail_mem=68.57 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.93it/s]Capturing num tokens (num_tokens=480 avail_mem=68.59 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.93it/s]Capturing num tokens (num_tokens=448 avail_mem=68.58 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.93it/s]

    Capturing num tokens (num_tokens=416 avail_mem=68.58 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.93it/s]Capturing num tokens (num_tokens=416 avail_mem=68.58 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=384 avail_mem=68.58 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=352 avail_mem=68.57 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=320 avail_mem=68.57 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=288 avail_mem=68.57 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=256 avail_mem=68.57 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=256 avail_mem=68.57 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.59it/s]Capturing num tokens (num_tokens=240 avail_mem=68.56 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.59it/s]Capturing num tokens (num_tokens=224 avail_mem=68.56 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.59it/s]Capturing num tokens (num_tokens=208 avail_mem=68.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.59it/s]

    Capturing num tokens (num_tokens=192 avail_mem=68.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.59it/s]Capturing num tokens (num_tokens=176 avail_mem=68.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.59it/s]Capturing num tokens (num_tokens=176 avail_mem=68.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=160 avail_mem=68.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=144 avail_mem=68.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=128 avail_mem=68.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=112 avail_mem=68.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=96 avail_mem=68.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.35it/s] Capturing num tokens (num_tokens=96 avail_mem=68.54 GB):  81%|████████  | 47/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=80 avail_mem=68.53 GB):  81%|████████  | 47/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=64 avail_mem=68.53 GB):  81%|████████  | 47/58 [00:01<00:00, 39.42it/s]

    Capturing num tokens (num_tokens=48 avail_mem=68.52 GB):  81%|████████  | 47/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=32 avail_mem=68.52 GB):  81%|████████  | 47/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=28 avail_mem=68.52 GB):  81%|████████  | 47/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=28 avail_mem=68.52 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=24 avail_mem=68.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=20 avail_mem=68.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=16 avail_mem=68.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=12 avail_mem=68.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=8 avail_mem=68.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.05it/s] Capturing num tokens (num_tokens=8 avail_mem=68.50 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.46it/s]Capturing num tokens (num_tokens=4 avail_mem=68.50 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.46it/s]Capturing num tokens (num_tokens=4 avail_mem=68.50 GB): 100%|██████████| 58/58 [00:01<00:00, 33.35it/s]


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
    Generated text:  Oli. My favorite hobby is to write. How did you end up in this wonderful world? I came to this place a year ago. I like to keep busy, so I started working at a huge library where I can read as much as I want. As a writer, I got the chance to publish my first book a few years ago. I also have a blog. I do my best to help people find the perfect books to read, and I write articles about the authors and their works. The library has been my home for the past year and a half. I have a girlfriend and three kids. I love the company
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to host a baseball game. The cost of ticketing for adults is $35 and the cost of ticketing for children is $15. The game will attract 200 adults and 150 children. What is the total cost for the tickets?
    
    The cost for the adults will be 200 adults x $35/adult = $7000
    The cost for the children will be 150 children x $15/child = $2250
    The total cost for the tickets will be $7000 + $2250
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. London C. Tokyo D. New York
    A. Paris
    B. London
    C. Tokyo
    D. New York
    Answer:
    A
    
    For the position of the leader at the latest stage of the project, what is the highest level of management position? A. Full-time B. Part-time C. Corresponding D. None of the above
    Answer:
    A
    
    The annual average temperature in location B is 8°C, while the annual average temperature in location A is 18°C. Which of the following options best describes the climate of location B?
    A. Cold
    B.
    ===============================
    Prompt: The future of AI is
    Generated text:  intriguing
    
    When we think of the future, we often imagine things that might come true before our own eyes. But often, those things are hard to imagine. In fact, we often get so caught up in the technological buzzwords and buzzier headlines that we sometimes forget the vision for the future. We focus on things that can’t happen, or things that are still in the future. But in reality, what we don’t know about the future is what will change the world and what will stay the same. So imagine that you are a young astronaut tasked with exploring the world in the future and exploring the different ways that


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is factually correct and provides a clear and concise overview of the capital city's location and significance in French culture and politics. It is a widely recognized and well-known fact that Paris is the capital city of France, and this statement accurately reflects this fact. 
    
    To provide additional context, Paris is the largest city in France and the second-largest city in the European Union. It is also the seat of the French government, the French Parliament, and the headquarters of the French Foreign Ministry. The city is known for its rich history, diverse culture, and iconic landmarks such as the Eiffel Tower, the Louvre
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more complex and sophisticated, there will be an increased focus on ethical considerations, such as privacy, bias, and accountability.
    
    2. Greater integration with human intelligence: AI systems will become more integrated with human intelligence, allowing for more complex and nuanced interactions between humans and machines.
    
    3. Development of new AI technologies: There will be continued development of new AI technologies, such as quantum computing, which could revolutionize AI systems.
    
    4.
    


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
    Generated text:  [Your Name], and I'm a [Type of Pet/Animal] with [Length] years of experience in the [Your field of work]. I have a [Number] of years of working experience in this [Your field of work], and I have been [Number] years working in this industry. What brings you to this position?
    [Your Name] loves to [What you do] and has been [How many years] years of experience in this field. This brings you to this position because [What you do], and I have been [How many years] years working in this industry. We have the potential to create
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    You are an AI assistant that helps you understand the responses with detail and examples. Question: How would you describe Paris as a place to visit in France?
    Paris, France is an amazing place to visit, known for its stunning architecture, world-class museums and art galleries, traditional French cuisine, and vibrant nightlife. The city's cuisine has influenced the global food scene and is celebrated worldwide. Its architecture, such as the Eiffel Tower and Notre-Dame Cathedral, are masterpieces of engineering and design that showcase the country's architectural heritage. The city's museums and art galleries, including the Louvre and Musée d'Or
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several trends that are expected to emerge in the coming years. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more companies and individuals become aware of the potential dangers of AI, the focus will shift towards ethical AI. This could lead to new regulations and guidelines being developed to ensure that AI is used in a responsible and safe manner.
    
    2. The rise of AI-powered robots: As AI technology continues to evolve, it is likely that robots will become increasingly integrated into our lives. These robots could be used for everything from service and maintenance tasks to manufacturing and research.
    
    3. Increased


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

     am

     a

     [

    Job

     Title

    /

    Position

    ]

     at

     [

    Company

     Name

    ].

     I

     bring

     a

     wealth

     of

     [

    number

     of

     years

     of

     experience

    ]

     of

     experience

     in

     [

    specific

     field

     or

     role

    ],

     and

     have

     always

     been

     passionate

     about

     [

    career

     goal

    ].

     I

     am

     [

    Age

    ]

     and

     [

    Gender

    ],

     and

     I

     am

     [

    Height

    /

    Weight

    ].

     I

     am

     [

    Hair

     Color

    /

    Eye

     Color

    ],

     and

     I

     have

     [

    Prof

    ession

    als

     Style

    ].

     I

     am

     [

    Current

     Job

     Role

    ]

     and

     I

     am

     a

     [

    Favorite

     Color

    ].

     I

     am

     excited

     to

     be

     here

    !

     How

     can

     I

     best

     begin

     my

     interaction

     with

     you

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

     am

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     many

     attractions

     and

     cultural

     events

    ,

     including

     the

     E

    iff

    el

     Tower

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

     rich

     history

     dating

     back

     to

     ancient

     times

     and

     is

     home

     to

     many

     famous

     landmarks

     and

     museums

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     a

     diverse

     population

     and

     a

     vibrant

     nightlife

    .

     French

     cuisine

     is

     also

     well

    -known

     for

     its

     delicious

     dishes

    ,

     including

     cro

    iss

    ants

     and

     tr

    uffles

    .

     The

     city

     is

     known

     for

     its

     lively

     culture

     and

     annual

     festivals

     such

     as

     the

     Christmas

     market

    .

     Paris

     is

     also

     an

     important

     global

     city

     with

     many

     foreign

     diplomats

     and

     business

     leaders

    .

     The

     city

     offers

     a

     wide

     range

     of

     transportation

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     diverse

     and

     constantly

     evolving

    ,

     with

     the

     potential

     to

     transform

     entire

     industries

     and

     bring

     about

     radical

     changes

     in

     society

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     As

     AI

     becomes

     more

     advanced

     and

     powerful

    ,

     it

     is

     likely

     to

     integrate

     more

     seamlessly

     with

     human

     intelligence

    ,

     enabling

     more

     complex

     and

     adaptive

     behaviors

    .

     This

     could

     lead

     to

     more

     effective

     decision

    -making

    ,

     adaptive

     learning

    ,

     and

     personalized

     experiences

    .
    


    2

    .

     Natural

     language

     processing

    :

     Advances

     in

     natural

     language

     processing

     (

    N

    LP

    )

     could

     allow

     AI

     to

     understand

     and

     interpret

     human

     language

     in

     a

     more

     sophisticated

     and

     nuanced

     way

    .

     This

     could

     lead

     to

     more

     natural

     and

     user

    -friendly

     interactions

    ,

     as

     well

     as

     more

     accurate

     and

    



```python
llm.shutdown()
```
