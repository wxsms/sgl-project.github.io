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


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=5632):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=3328):  10%|█         | 6/58 [00:04<00:31,  1.67it/s]

    Compiling num tokens (num_tokens=3072):  10%|█         | 6/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  22%|██▏       | 13/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=960):  22%|██▏       | 13/58 [00:05<00:10,  4.42it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.07it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.07it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.07it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.07it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.07it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.07it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.07it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.07it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.07it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.16it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.16it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.16it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.16it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.16it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.16it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.16it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.16it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.16it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.11it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.11it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.11it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.11it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.11it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.11it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.11it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.11it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.11it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.11it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 39.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.27 GB):   3%|▎         | 2/58 [00:00<00:03, 17.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.27 GB):   3%|▎         | 2/58 [00:00<00:03, 17.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.26 GB):   3%|▎         | 2/58 [00:00<00:03, 17.45it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.26 GB):   3%|▎         | 2/58 [00:00<00:03, 17.45it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.26 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.26 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.25 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.24 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.24 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.84it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.23 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.23 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.22 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.22 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.22 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.22 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.21 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.21 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.17it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.21 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.19 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.17it/s]Capturing num tokens (num_tokens=960 avail_mem=72.20 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.17it/s] Capturing num tokens (num_tokens=960 avail_mem=72.20 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.14it/s]Capturing num tokens (num_tokens=896 avail_mem=72.20 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.14it/s]Capturing num tokens (num_tokens=832 avail_mem=72.20 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.14it/s]Capturing num tokens (num_tokens=768 avail_mem=71.67 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.14it/s]Capturing num tokens (num_tokens=704 avail_mem=71.67 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.14it/s]Capturing num tokens (num_tokens=640 avail_mem=71.67 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.14it/s]Capturing num tokens (num_tokens=640 avail_mem=71.67 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=576 avail_mem=69.83 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.11it/s]

    Capturing num tokens (num_tokens=512 avail_mem=69.82 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=480 avail_mem=69.83 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=448 avail_mem=69.83 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=416 avail_mem=69.83 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=416 avail_mem=69.83 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=384 avail_mem=69.83 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=352 avail_mem=69.82 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=320 avail_mem=69.82 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=288 avail_mem=69.82 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=256 avail_mem=69.81 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.10it/s]

    Capturing num tokens (num_tokens=256 avail_mem=69.81 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.24it/s]Capturing num tokens (num_tokens=240 avail_mem=69.81 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.24it/s]Capturing num tokens (num_tokens=224 avail_mem=69.81 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.24it/s]Capturing num tokens (num_tokens=208 avail_mem=69.80 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.24it/s]Capturing num tokens (num_tokens=192 avail_mem=69.80 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.24it/s]Capturing num tokens (num_tokens=176 avail_mem=69.80 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.24it/s]Capturing num tokens (num_tokens=176 avail_mem=69.80 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=160 avail_mem=69.80 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=144 avail_mem=69.79 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=128 avail_mem=69.79 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s]

    Capturing num tokens (num_tokens=112 avail_mem=69.79 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=96 avail_mem=69.78 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s] Capturing num tokens (num_tokens=96 avail_mem=69.78 GB):  81%|████████  | 47/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=80 avail_mem=69.78 GB):  81%|████████  | 47/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=64 avail_mem=69.78 GB):  81%|████████  | 47/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=48 avail_mem=69.77 GB):  81%|████████  | 47/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=32 avail_mem=69.77 GB):  81%|████████  | 47/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=28 avail_mem=69.76 GB):  81%|████████  | 47/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=28 avail_mem=69.76 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=24 avail_mem=69.76 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=20 avail_mem=69.76 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s]

    Capturing num tokens (num_tokens=16 avail_mem=69.76 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=12 avail_mem=69.75 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=8 avail_mem=69.75 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.65it/s] Capturing num tokens (num_tokens=8 avail_mem=69.75 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=4 avail_mem=69.75 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=4 avail_mem=69.75 GB): 100%|██████████| 58/58 [00:01<00:00, 37.55it/s]


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
    Generated text:  Dafny Chokri and I'm a computer science student at the University of Delaware. I'm really excited to start my course this semester on the topic of "Homework Problems: The Graph Theory part". I'm working on the project "The Maximum Flow Problem". This is the first part of the assignment where I am supposed to draw a tree diagram, then compute the maximum flow and minimum cut. I will need the help of the teacher to solve the problem. Thank you. The assignment has many parts and I'm currently stuck on part (a) of this problem, I don't understand how to solve the first part of
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to invite Hillary Clinton or Joe Biden to a major international summit on climate change. He wants to know the likelihood that he will be invited. The probability that Clinton will be invited is 0.4 and the probability that Biden will not be invited is 0.3. Calculate the probability that the president will be invited.
    To determine the probability that the president of the United States will be invited to the major international summit on climate change, we need to consider the probabilities of each of the two possible scenarios: the president being invited or not being invited.
    
    Given:
    - The probability that Hillary Clinton will be invited is \(
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, located on the banks of the Seine River. The main landmarks of the city include the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and the Palace of Versailles. 
    
    There are several cities in the world with the same name and the same meaning, such as New York, Los Angeles, and Chicago. Is it possible to have a capital city that is located in a country other than France, and named after the same meaning as Paris, while also being a major tourist destination? Yes, it is possible to have a capital city that is located in a country other than France, and named
    ===============================
    Prompt: The future of AI is
    Generated text:  changing, and there's no shortage of exciting news and innovation in this emerging field. From the high-flying tech companies to the sobering challenges that exist in the field of AI, we've got a lot of interesting stuff to discuss.
    Technology is changing faster than ever before, and the future of AI is something that many people are looking forward to. But the good news is that even though the field of AI is growing, it's also developing very quickly.
    As AI continues to develop, it's becoming more and more complex, and there's a lot of uncertainty surrounding its future. However, with the right tools and technologies, the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about your personality or skills]. I'm always looking for new opportunities to grow and learn, and I'm always eager to share my knowledge and experience with others. What's your background and what do you bring to the table? I have a [insert a short, positive, enthusiastic statement about your background or skills]. I'm always looking for new challenges and opportunities to grow and learn, and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the European Union. The city is located on the Seine River and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is known for its rich history, art, and culture, and is a popular tourist destination. It is also home to many important institutions such as the French Academy of Sciences and the French National Library. The city is known for its fashion industry, with many famous designers and boutiques. Paris is a vibrant and dynamic city that
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, including in areas such as diagnosis, treatment planning, and patient monitoring.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management and fraud detection. As AI technology continues
    


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
    Generated text:  [Your Name], and I am a [character type] who is passionate about [the topic of your character]. I love exploring new ideas and learning new things, and I am always eager to share my knowledge with others. I am an [character type] who is [character trait] and I am always looking for ways to make the world a better place. I love [character trait] and I am always ready to do whatever it takes to help others. I'm excited to meet you! Let's connect. [Your Name] [Your contact information] [Your image if available] [Your social media handles] [Your character
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, an ancient city located in the northwestern region of France, on the Île de la Cité.
    
    That statement summarizes the capital city of France, which is Paris. The city is home to the historic Louvre Museum, Notre Dame Cathedral, and the Arc de Triomphe, among others. Paris is a major cultural and economic hub in France and plays a significant role in the country's identity and politics. Its name translates to "City of Light" in Latin, referring to its long history as a city of light. The city is also known for its numerous museums, fashion shows, and street performers. Paris has a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  certainly uncertain, but there are several trends that are likely to shape the technology in the coming years:
    
    1. More integration of AI in various industries: As AI becomes more advanced, it is likely to be integrated into various industries, from healthcare to finance to manufacturing. This will result in more efficient and cost-effective processes, as well as improved decision-making based on data analysis.
    
    2. Development of new AI technologies: AI is evolving rapidly, and new technologies will be developed to further enhance its capabilities. This will include more advanced natural language processing, robotics, and machine learning.
    
    3. AI ethics and legal considerations: As AI becomes more prevalent


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

    ],

     and

     I

     am

     a

     [

    Job

     Title

    ]

     with

     over

     [

    Number

     of

     Years

     in

     Industry

    ]

     years

     of

     experience

     in

     [

    Industry

    ].

     I

     have

     [

    Number

     of

     Success

     Stories

    ]

     successes

    ,

     and

     I

    've

     learned

     from

     each

     one

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

    Name

    ]

     is

     a

     [

    job

     title

    ]

     with

     over

     [

    number

     of

     years

     in

     industry

    ]

     years

     of

     experience

     in

     [

    industry

    ].

     I

     have

     [

    number

     of

     success

     stories

    ]

     success

     stories

    ,

     and

     I

    've

     learned

     from

     each

     one

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     I

     am

     a

     [

    job

     title

    ],

     [

    job

     title

    ],

     and

     [

    job

     title

    ].

     I

     have

     [

    number

     of

     years

     in

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     C

    ité

    ."


    Paris

     is

     the

     largest

     city

     in

     France

    ,

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

    ,

     and

     is

     home

     to

     the

     cit

    ad

    els

     of

     Paris

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

     It

     is

     also

     a

     symbol

     of

     French

     culture

    ,

     history

    ,

     and

     fashion

    .

     Paris

     is

     a

     bustling

     city

     with

     a

     rich

     cultural

     heritage

     and

     a

     vibrant

     economy

    ,

     and

     its

     museums

     and

     attractions

     attract

     millions

     of

     visitors

     each

     year

    .

     The

     city

     has

     a

     long

     and

     stor

    ied

     history

    ,

     with

     a

     rich

     tape

    stry

     of

     French

     culture

     and

     influences

    ,

     including

     the

     influence

     of

     Renaissance

     art

    ,

     the

     Bar

    oque

     style

    ,

     and

     the

     Romantic

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     promising

     and

     varied

    ,

     and

     there

     are

     many

     possible

     trends

     that

     could

     shape

     the

     field

    .

     Here

     are

     some

     of

     the

     most

     likely

     and

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     understand

     and

     respond

     to

     individual

     human

     needs

    ,

     leading

     to

     more

     personalized

     and

     effective

     solutions

     for

     people

    .
    


    2

    .

     Better

     Translation

     and

     Communication

    :

     AI

     will

     help

     with

     translation

     and

     communication

     across

     multiple

     languages

    ,

     making

     it

     easier

     for

     people

     to

     communicate

     and

     collaborate

     with

     others

    .
    


    3

    .

     Personal

    ized

     Learning

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     understand

     and

     analyze

     the

     human

     learning

     process

    ,

     leading

     to

     more

     personalized

     learning

     experiences

     for

     individuals

    .
    


    4

    .

     Improved

     Healthcare

    



```python
llm.shutdown()
```
