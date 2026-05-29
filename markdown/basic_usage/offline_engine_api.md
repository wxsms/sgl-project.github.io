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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.20it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.15it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  9.15it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  9.15it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 22.74it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 22.74it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 22.74it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 22.74it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 22.74it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 22.74it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 22.74it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 22.74it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 22.74it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 22.74it/s] 

    Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:04<00:00, 22.74it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.66it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.66it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.66it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.66it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.38it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.38it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.53it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.94it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.94it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.94it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.94it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.94it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.94it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.95it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.94it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.94it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 39.78it/s]


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
    Generated text:  Kim, I'm a little older than you. I'm 21 years old. My favorite subject is Math. I like to eat ice cream and watch TV. I play baseball very well and I'm in the middle of my first year at school. What's your favorite subject? My favorite subject is Math, and I love ice cream. What sport are you in? I play baseball very well. My favorite sport is baseball. How about you? I like to play soccer. My favorite sport is soccer. What does your school do? It's a small school with only a few hundred students. We have a lot of small
    ===============================
    Prompt: The president of the United States is
    Generated text:  a president who is 6 feet tall and weighs 200 pounds. The vice president of the United States is 5 feet tall and weighs 150 pounds. How many pounds does the president weigh?
    
    To determine how many pounds the president weighs, we need to add his height to his weight. The president's height is 6 feet and his weight is 200 pounds. Since there are 12 inches in a foot, we can convert the president's height from feet to inches and then add it to his weight in pounds.
    
    First, we convert the president's height from feet to inches:
    \[ 
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city in France and the second largest city in Europe. It is also a UNESCO World Heritage site and is the capital of the Ile-de-France region, the largest region in France.
    Paris is a historical, artistic, and cultural center, the seat of the President of the Republic and the largest city in France. It is also home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, the Musée d'Orsay, the Musée Rodin, the Musée Rodin, the Musée Carnavalet, and the Musée d'Art Moderne. Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  to make the technology accessible to everyone and to help to solve some of the biggest issues of our time. While many believe that the development of AI is right and that it is needed, the technology is still in the process of development and improvement.
    Developing AI technology requires a significant amount of resources, including computer power, software, and data. AI technology is also limited by the amount of data it can process and the speed at which it can access and utilize that data.
    The most common way that AI technology is being used is by the healthcare industry, which is able to use AI to assist in diagnosis, treatment, and prevention of diseases


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and festivals. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also known for its rich history, including the influence of French Revolution and Napoleon Bonaparte. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is also home to many international organizations and institutions, including the European Union and the United Nations. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This could lead to the creation of new jobs, but it could also lead to the displacement of human workers in certain roles.
    
    2. Enhanced privacy and security: As AI technology becomes more advanced, we are likely to see an increase in the use of AI in areas that involve sensitive data, such as healthcare and finance. This could lead
    


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
    Generated text:  _____. I'm a/an 18-year-old _____. And I'm also the star of _____. I love _____. What's your favorite hobby or activity? 
    
    Additional requirements:
    - Your introduction should be concise yet informative, highlighting your identity, profession, and most importantly, your passion or interests. 
    - Use active verbs and describe your character in a way that paints a picture for the reader.
    - Mention at least one piece of personal information about you, like your birth date or a notable achievement.
    - End with a statement of your relationship to the character you are introducing, which could be a quote, a quote, or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Explanation for an 8-year old: Paris is like a big city where lots of people live, and many things are made there. It's also the city where lots of important people work and talk to each other. 
    
    In French, Paris is called "Lyon" and in English, it's called "Paris." Paris is a very big city with lots of tall buildings and pretty trees. It's the biggest city in Europe, and people like to travel there a lot! 
    
    But guess what? Paris isn't the whole country. There are lots of other cities nearby too, like Nice and Var, so Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  increasingly focused on emerging trends such as:
    
    1. AI and machine learning: Advances in machine learning are leading to more complex and sophisticated AI systems, which are becoming increasingly capable of handling increasingly complex tasks.
    
    2. Explainability and transparency: As AI systems become more complex and autonomous, there is a need for better explanations of their decisions and actions, in order to ensure accountability and trust.
    
    3. Ethics and responsible AI: As AI systems become more autonomous and complex, there is a need for a clearer understanding of the ethical implications of their use. This includes considerations of bias, discrimination, and the potential for harm.
    
    4. Relevance and


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

    occupation

    ]

     who

     has

     been

     following

     my

     passion

     for

     [

    profession

    ].

     I

    'm

     a

     [

    profession

    ]

     that

     specializes

     in

     [

    profession

    ]

     and

     have

     been

     working

     for

     [

    years

    ]

     at

     [

    company

    ].

     I

    've

     always

     been

     passionate

     about

     [

    profession

    ]

     and

     have

     been

     dedicated

     to

     [

    profession

    ]

     for

     years

    .

     I

    'm

     always

     on

     the

     lookout

     for

     new

     opportunities

     to

     expand

     my

     knowledge

     and

     gain

     new

     experiences

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     help

     others

     achieve

     their

     goals

    .

     I

     believe

     in

     the

     power

     of

     learning

     and

     always

     strive

     to

     do

     my

     best

    ,

     no

     matter

     what

    .

     Thank

     you

     for

     the

     opportunity

     to

     introduce

     myself

     to

     you

    !

     Let

     me

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    What

     is

     the

     capital

     of

     France

    ?
    


    What

     is

     the

     capital

     of

     France

    ?

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     exciting

     and

     diverse

    ,

     driven

     by

     both

     the

     progress

     in

     AI

     technology

     and

     the

     continuous

     development

     of

     new

     applications

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

     Integration

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     integrate

     with

     other

     technologies

    ,

     such

     as

     sensors

    ,

     robots

    ,

     and

     other

     machine

     learning

     algorithms

    ,

     to

     create

     more

     complex

     and

     diverse

     systems

    .

     This

     will

     enable

     AI

     to

     learn

     from

     and

     adapt

     to

     the

     environment

    ,

     making

     it

     more

     efficient

     and

     effective

    .
    


    2

    .

     Enhanced

     human

    -machine

     interaction

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     will

     increasingly

     be

     able

     to

     interact

     with

     humans

     in

     more

     natural

     and

     intuitive

     ways

    .

     This

     will

     require

     a

     greater

     emphasis

     on

     human

    -machine

     integration

    



```python
llm.shutdown()
```
