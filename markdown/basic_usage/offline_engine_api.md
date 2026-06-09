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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.25it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.25it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.25it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.25it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.25it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.25it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.25it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.25it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.25it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 14.49it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 14.49it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 14.49it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 14.49it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:02, 14.49it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:02, 14.49it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:02, 14.49it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:02, 14.49it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:02, 14.49it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:02, 14.49it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 21.71it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 21.71it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 21.71it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 21.71it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 21.71it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 21.71it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.71it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.71it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.93it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.93it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.93it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.93it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.93it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.93it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.93it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.93it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.93it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.93it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.21 GB):   3%|▎         | 2/58 [00:00<00:05, 11.05it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.21 GB):   3%|▎         | 2/58 [00:00<00:05, 11.05it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.21 GB):   3%|▎         | 2/58 [00:00<00:05, 11.05it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.21 GB):   7%|▋         | 4/58 [00:00<00:04, 12.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.21 GB):   7%|▋         | 4/58 [00:00<00:04, 12.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.20 GB):   7%|▋         | 4/58 [00:00<00:04, 12.37it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=57.20 GB):  10%|█         | 6/58 [00:00<00:04, 11.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.07 GB):  10%|█         | 6/58 [00:00<00:04, 11.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.06 GB):  10%|█         | 6/58 [00:00<00:04, 11.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.06 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.06 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.69it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.06 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.05 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.05 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.69it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=61.05 GB):  21%|██        | 12/58 [00:00<00:02, 20.26it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.05 GB):  21%|██        | 12/58 [00:00<00:02, 20.26it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.05 GB):  21%|██        | 12/58 [00:00<00:02, 20.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.04 GB):  21%|██        | 12/58 [00:00<00:02, 20.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.04 GB):  21%|██        | 12/58 [00:00<00:02, 20.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.04 GB):  21%|██        | 12/58 [00:00<00:02, 20.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=960 avail_mem=61.02 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=61.02 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=896 avail_mem=61.02 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=832 avail_mem=61.02 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=768 avail_mem=61.01 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=704 avail_mem=61.01 GB):  38%|███▊      | 22/58 [00:01<00:01, 33.46it/s]Capturing num tokens (num_tokens=640 avail_mem=61.01 GB):  38%|███▊      | 22/58 [00:01<00:01, 33.46it/s]Capturing num tokens (num_tokens=640 avail_mem=61.01 GB):  47%|████▋     | 27/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=576 avail_mem=61.01 GB):  47%|████▋     | 27/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=512 avail_mem=60.99 GB):  47%|████▋     | 27/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=480 avail_mem=61.01 GB):  47%|████▋     | 27/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=448 avail_mem=61.00 GB):  47%|████▋     | 27/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=416 avail_mem=61.00 GB):  47%|████▋     | 27/58 [00:01<00:00, 37.70it/s]

    Capturing num tokens (num_tokens=416 avail_mem=61.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.80it/s]Capturing num tokens (num_tokens=384 avail_mem=61.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.80it/s]Capturing num tokens (num_tokens=352 avail_mem=61.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.80it/s]Capturing num tokens (num_tokens=320 avail_mem=60.99 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.80it/s]Capturing num tokens (num_tokens=288 avail_mem=60.99 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.80it/s]Capturing num tokens (num_tokens=256 avail_mem=60.99 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.80it/s]Capturing num tokens (num_tokens=256 avail_mem=60.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=240 avail_mem=60.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=224 avail_mem=60.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=208 avail_mem=60.97 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=192 avail_mem=60.97 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=176 avail_mem=60.97 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.74it/s]

    Capturing num tokens (num_tokens=176 avail_mem=60.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.24it/s]Capturing num tokens (num_tokens=160 avail_mem=60.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.24it/s]Capturing num tokens (num_tokens=144 avail_mem=60.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.24it/s]Capturing num tokens (num_tokens=128 avail_mem=60.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.24it/s]Capturing num tokens (num_tokens=112 avail_mem=60.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.24it/s]Capturing num tokens (num_tokens=96 avail_mem=60.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.24it/s] Capturing num tokens (num_tokens=96 avail_mem=60.96 GB):  81%|████████  | 47/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=80 avail_mem=60.95 GB):  81%|████████  | 47/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=64 avail_mem=60.95 GB):  81%|████████  | 47/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=48 avail_mem=60.94 GB):  81%|████████  | 47/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=32 avail_mem=60.94 GB):  81%|████████  | 47/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=28 avail_mem=60.94 GB):  81%|████████  | 47/58 [00:01<00:00, 45.01it/s]

    Capturing num tokens (num_tokens=28 avail_mem=60.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=24 avail_mem=60.93 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=20 avail_mem=60.93 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=16 avail_mem=60.93 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=12 avail_mem=60.92 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=8 avail_mem=60.92 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.50it/s] Capturing num tokens (num_tokens=8 avail_mem=60.92 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.24it/s]Capturing num tokens (num_tokens=4 avail_mem=60.92 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.24it/s]Capturing num tokens (num_tokens=4 avail_mem=60.92 GB): 100%|██████████| 58/58 [00:01<00:00, 33.98it/s]


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
    Generated text:  Peter. I am an engineer. I love music. I like to sing and play the guitar. But I don't have many friends because I don't have much time. I like playing games on the computer. I am from America. I have some friends in China. There are many Chinese students in my school. I like them very much. I am in the US school. I have some Chinese friends. My Chinese teacher is very nice. He often helps me with my Chinese. What can we learn from the passage? A) Peter likes playing computer games. B) Peter likes Chinese food very much. C) Peter likes America
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting three different countries. He stops in Beijing, a city far away where the traffic is considered more efficient. The president wants to visit each country in a single day. He decides to visit the first country first, then the second country, and finally the third country. If it takes him 3 hours to travel from Beijing to his first country, 2 hours to travel from his first country to the second country, and 4 hours to travel from the second country to his third country, how many total hours will it take for him to visit all three countries?
    To determine the total time it will take for the president to visit all
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. London
    C. New York
    D. Tokyo
    
    To determine the capital of France, let's break down the options and analyze them step by step.
    
    1. **Paris**:
       - Paris is the capital of France.
       - Paris is renowned for its historical significance, art, and cuisine.
       - The other options (London, New York, Tokyo) are not capitals of France.
    
    2. **London**:
       - London is the capital of the United Kingdom.
       - London is famous for its rich history, culture, and international connections.
       - The other options (
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and growing.
    But it is also very complex and controversial.
    A lot of people are trying to figure out how to use AI to help solve problems and advance society.
    As always, there are some good and bad things to expect.
    This talk will cover some basic concepts about AI, including why it is a good idea and a bad idea, and how we can help make it work in our best interest.
    The talk will also provide some general advice on how to approach using AI in your own life.
    Some basic concepts about AI, including why it is a good idea and a bad idea, and how we can help make it work in


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the country and the most populous city in Europe. It is located on the Seine River and is the seat of the French government, the French Parliament, and the French Supreme Court. Paris is known for its rich history, art, and culture, and is a major tourist destination. It is also home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a vibrant and diverse city with a rich cultural and artistic heritage, and is a popular tourist destination for its beautiful architecture, food,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more transparent and accountable AI systems that are designed to
    


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
    Generated text:  [Character Name] and I'm a [Job Title] with [Company Name]. I am passionate about [Your Passion]. I have always been driven by [Your Passion], and I have been dedicated to achieving my goals and making a difference in the world. I am always looking for ways to enhance my skills and knowledge, and I am eager to learn from those who can offer valuable insights and advice. I have a strong work ethic and a sense of responsibility, and I am always focused on delivering results to the best of my abilities. I believe in the power of innovation and creativity, and I am always eager to share my ideas and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. It is the most populous city in France and the third most populous in the European Union. It is also the cultural and economic center of the country, and known for its rich history, beautiful architecture, and delicious cuisine. Paris is located on the Seine River, in the center of the city, and is home to a wide variety of attractions and cultural events. The city is a hub for art, music, literature, and film, and hosts numerous events throughout the year, including the World Fair and the Paris Fashion Week. Despite its size and popularity, Paris is also known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a number of significant trends. One of the most promising trends is the continued development of machine learning and deep learning. These technologies have the potential to significantly increase the speed and accuracy with which we can solve complex problems, and they are also expected to become increasingly affordable and accessible. The development of AI-powered autonomous vehicles is also a potential future trend, as they have the potential to greatly reduce traffic accidents and accidents in general. Additionally, AI will likely continue to play a role in areas such as healthcare, finance, and manufacturing, where they can help automate and optimize processes. Finally, AI will likely continue to be used


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

    occupation

    ]

     with

     [

    degree

    ]

     in

     [

    field

     of

     study

    ].

     I

    'm

     a

     [

    type

     of

     person

    ],

     [

    strength

    s

    ],

     and

     [

    weak

    ness

    es

    ].

     I

    'm

     [

    career

     goal

    ],

     [

    impact

     on

     community

    ],

     and

     [

    short

    coming

    ].

     Whether

     I

    'm

     [

    learning

     from

     experience

    ],

     [

    keeping

     up

     with

     industry

     trends

    ],

     or

     [

    learning

     from

     mistakes

    ],

     I

    'm

     always

     looking

     for

     ways

     to

     [

    adv

    antage

    ].

     [

    Name

    ]

     is

     [

    age

    ],

     [

    national

    ity

    ],

     [

    rel

    igion

    ],

     [

    language

    ].

     I

    'm

     a

     [

    mot

    iv

    ator

    ],

     [

    support

    er

    ],

     and

     [

    amb

    assador

    ].

     I

    'm

     [

    position

    ],

     and

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     seat

     of

     the

     French

     government

     and

     the

     heart

     of

     the

     French

     society

    .

     It

     is

     also

     the

     most

     important

     city

     in

     the

     world

     in

     the

     world

     of

     media

    ,

     fashion

    ,

     and

     film

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     lively

     nightlife

    .

     France

    ’s

     capital

     is

     located

     in

     the

     Î

    le

     de

     la

     C

    ité

    ,

     a

     beautiful

     island

     on

     the

     Se

    ine

     River

    .

     It

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Lights

    "

     due

     to

     its

     historical

     significance

     and

     the

     city

    ’s

     contribution

     to

     modern

     art

     and

     culture

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

    ,

     and

     it

     is

     the

     birth

    place

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

     and

     has

     the

     potential

     to

     revolution

    ize

     numerous

     industries

    ,

     transforming

     the

     way

     we

     live

    ,

     work

    ,

     and

     interact

     with

     technology

    .

     Here

     are

     some

     potential

     trends

     that

     could

     emerge

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     every

     aspect

     of

     daily

     life

    :

     AI

     will

     become

     increasingly

     integrated

     into

     our

     daily

     lives

    ,

     from

     our

     homes

     to

     our

     workplaces

    ,

     cars

    ,

     and

     entertainment

    .

     This

     will

     likely

     result

     in

     a

     more

     seamless

     and

     intuitive

     user

     experience

    ,

     with

     AI

     making

     many

     tasks

     easier

     and

     more

     efficient

    .
    


    2

    .

     AI

    -driven

     automation

     and

     reduced

     human

     involvement

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     it

     is

     likely

     to

     automate

     many

     tasks

     that

     require

     human

     input

     or

     coordination

    



```python
llm.shutdown()
```
