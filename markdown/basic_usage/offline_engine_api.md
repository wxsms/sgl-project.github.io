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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.26it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.67it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.67it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.67it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.67it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.67it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.67it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.67it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.67it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.67it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 13.89it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 13.89it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 13.89it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 13.89it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 13.89it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 13.89it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.89it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.89it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.89it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 20.07it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 20.07it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 20.07it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 20.07it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 20.07it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 20.07it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 20.07it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 20.07it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 20.07it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 20.07it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 28.23it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 28.23it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 37.96it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 37.96it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 37.96it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 37.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 15.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 15.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 15.56it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=75.13 GB):   7%|▋         | 4/58 [00:00<00:03, 17.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   7%|▋         | 4/58 [00:00<00:03, 17.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   7%|▋         | 4/58 [00:00<00:03, 17.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   7%|▋         | 4/58 [00:00<00:03, 17.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.73it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.66 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.23it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.23it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.23it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s] Capturing num tokens (num_tokens=896 avail_mem=74.64 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.57it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.57it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.57it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.57it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.57it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=384 avail_mem=74.62 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.29it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.05it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.05it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.05it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.05it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.05it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.73it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.73it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.73it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.73it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.73it/s]

    Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=112 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 31.57it/s] Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 34.49it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=20 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=16 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.40it/s] Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.14it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.14it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 32.54it/s]


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
    Generated text:  Alice Green. I'm a 16-year-old student. I come from a small village in England. I live in London. I go to school from Monday to Friday. I often play sports with my classmates. My favorite sport is soccer. I like to eat lots of vegetables and fruits. My parents don't smoke or drink. They always keep a healthy diet. I have many hobbies. I like reading and playing the guitar. What do you think of this school? I like this school very much because I have so many things to do and I have lots of friends to play games with. I also have so many interesting classes
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. What is the probability that the president of the United States is a woman? To determine the probability that the president of the United States is a woman, we need to know the total number of male and female presidents and the total number of presidents in the United States.
    
    Assuming we have the following information (this is a hypothetical scenario for the purpose of this problem):
    
    - There are 44 male presidents.
    - There are 43 female presidents.
    
    The total number of presidents is the sum of the male and female presidents:
    \[ 44 + 43 = 87 \]
    
    The probability that the president
    ===============================
    Prompt: The capital of France is
    Generated text:  called Paris. The population of Paris is 2,000,000. The capital of Italy is called Rome. The population of Rome is 3,000,000. How many more people are in Paris than in Rome? To determine how many more people are in Paris than in Rome, we need to subtract the population of Rome from the population of Paris. Here are the steps:
    
    1. Identify the population of Paris.
       \[
       \text{Population of Paris} = 2,000,000
       \]
    
    2. Identify the population of Rome.
    
    ===============================
    Prompt: The future of AI is
    Generated text:  inherently unpredictable and will likely evolve in ways we cannot fully predict. So, what exactly does that mean? And how can we prepare for it?
    
    Is AI more likely to be a disruptive technology, or an existential threat?
    
    What will the AI industry look like in 2040?
    
    In this roundtable, we'll explore the future of AI and its potential impact on society, from the horizon up to a global crisis. But before we get to the big picture, let's dive into a few questions for the roundtable's participants.
    
    In a world where AI is increasingly embedded in our daily lives, do you think the field has


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a unique trait or skill that sets me apart from other people]. And what's your background? I'm a [insert a unique background or experience that sets me apart from other people]. And what's your favorite hobby or activity? I'm a [insert a unique hobby or activity that sets me apart from other people]. And what's your favorite book or movie? I'm a [insert a unique favorite book or movie that sets me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third-largest city in the world by population. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for finance, fashion, and entertainment. Paris is a city that is constantly evolving and changing, with new developments and cultural events taking place all the time. The city is a symbol of France and a major tourist destination for millions of visitors each year. Paris is a city that is both beautiful and exciting,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to become more capable of learning from large datasets and making more accurate predictions and decisions. This could lead to more efficient and effective use of resources, as well as better decision-making in various industries.
    
    3. Greater reliance on AI for decision-making: AI is
    


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
    Generated text:  [Name] and I am a [Age] year old, [Gender] girl. I am a [Occupation] who is very [In-charge] in my [Occupation] at [Company]. I am [Occupation], and I have been in [Occupation] for [X] years. I have always been passionate about [Occupation] and I have a deep [In-charge] in the industry. I am an [Occupation] with [X] years of experience, and I have a [X] of [X] awards. I am [Occupation], and I am a [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris, the city of love and culture, is one of the most popular cities in the world, known for its stunning architecture, rich history, and vibrant street life. Its iconic landmarks include the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a popular tourist destination, offering a blend of modern amenities and historical heritage. It's a bustling city with a strong sense of community, attracting millions of visitors annually. The city's culture, food, and traditions are deeply ingrained in its identity, making it an important part of French society. 
    
    In conclusion, Paris stands as a unique melting
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and constantly evolving. Here are some possible trends that could shape AI in the years to come:
    
    1. Advancements in machine learning: AI will continue to make significant strides in machine learning. Advances in deep learning, natural language processing, and computer vision will lead to even more sophisticated algorithms that can recognize patterns and make predictions.
    
    2. Personalization: AI will continue to improve as it learns more about users and their needs. Personalized AI will allow users to receive more tailored information and recommendations based on their interactions and preferences.
    
    3. Ethical considerations: As AI becomes more integrated into our daily lives, ethical concerns will rise.


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

    ].

     I

    'm

     [

    Age

    ]

     years

     old

     and

     [

    Gender

    ]

     (

    f

     or

     m

    ).

     I

    'm

     [

    Occup

    ation

    ]

     and

     I

    've

     been

     working

     here

     [

    Job

    ]

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

     grew

     up

     in

     [

    Your

     Place

     of

     Origin

    ]

     and

     started

     here

     [

    Your

     Country

     of

     Origin

    ].

     I

    'm

     from

     [

    Your

     Home

     Town

    ]

     and

     I

     speak

     [

    Your

     Language

    ].

     I

     love

     my

     hometown

    ,

     [

    H

    ometown

     Name

    ],

     and

     my

     country

    ,

     [

    Country

     Name

    ].

     In

     my

     free

     time

    ,

     I

     enjoy

     [

    I

     enjoy

     doing

    ].

     I

     have

     a

     passion

     for

     [

    I

     have

     a

     passion

     for

    ].

     I

    'm

     a

     [

    I

     am

     a

    ]

     person

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Se

    ine

     River

     in

     the

     center

     of

     the

     country

     and

     is

     the

     largest

     city

     in

     the

     European

     Union

    .

     It

     is

     also

     the

     oldest

     capital

     city

     in

     Europe

    ,

     having

     been

     the

     seat

     of

     the

     French

     monarchy

     since

     the

     

    1

    2

    th

     century

    .

     The

     city

     is

     home

     to

     many

     iconic

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     known

     for

     its

     rich

     cultural

     heritage

    ,

     including

     its

     oper

    atic

     history

    ,

     modern

     art

     scenes

    ,

     and

     romantic

     art

     nouveau

     style

    .

     The

     city

     is

     also

     known

     for

     its

     cuisine

    ,

     including

     its

     famous

     cro

    iss

    ants

     and

     its

     many

     international

     restaurants

     and

     bars

    .

     Paris

     has

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     transformative

    ,

     with

     potential

     to

     transform

     almost

     every

     industry

     and

     aspect

     of

     human

     life

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

     depth

     of

     natural

     language

     processing

    :

     AI

     will

     continue

     to

     improve

     in

     terms

     of

     its

     ability

     to

     understand

     and

     generate

     human

    -like

     speech

     and

     language

    ,

     which

     will

     allow

     for

     more

     sophisticated

     and

     nuanced

     interactions

     with

     people

     and

     machines

    .
    


    2

    .

     Personal

    ized

     and

     contextual

     AI

    :

     AI

     will

     become

     more

     personalized

     and

     context

    -aware

    ,

     allowing

     machines

     to

     learn

     from

     individual

     users

     and

     respond

     to

     their

     specific

     needs

     and

     preferences

    .
    


    3

    .

     Enhanced

     machine

     learning

     algorithms

    :

     AI

     will

     become

     even

     more

     sophisticated

    ,

     with

     the

     ability

     to

     learn

     from

     large

     datasets

    ,

     recognize

     patterns

    ,

    



```python
llm.shutdown()
```
