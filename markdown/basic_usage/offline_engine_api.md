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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.98it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.86it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.86it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.86it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.86it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.86it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.86it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.86it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:04<00:07,  6.20it/s] 

    Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 15.92it/s]

    Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 23.42it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 23.42it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 23.42it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 23.42it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 23.42it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 23.42it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 23.42it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:00, 23.42it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:00, 23.42it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:00, 23.42it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 32.39it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 32.39it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 32.39it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 32.39it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 32.39it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 32.39it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 32.39it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 32.39it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 38.49it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 38.49it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 38.49it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 38.49it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 38.49it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 38.49it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 38.49it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.94 GB):   2%|▏         | 1/58 [00:00<00:06,  8.27it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.22 GB):   2%|▏         | 1/58 [00:00<00:06,  8.27it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=71.21 GB):   2%|▏         | 1/58 [00:00<00:06,  8.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.21 GB):   5%|▌         | 3/58 [00:00<00:05, 10.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.20 GB):   5%|▌         | 3/58 [00:00<00:05, 10.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.99 GB):   5%|▌         | 3/58 [00:00<00:05, 10.73it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=70.99 GB):   9%|▊         | 5/58 [00:00<00:04, 13.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.99 GB):   9%|▊         | 5/58 [00:00<00:04, 13.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.98 GB):   9%|▊         | 5/58 [00:00<00:04, 13.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.98 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.15 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.16 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.51it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=71.16 GB):  16%|█▌        | 9/58 [00:00<00:03, 16.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.16 GB):  16%|█▌        | 9/58 [00:00<00:03, 16.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.15 GB):  16%|█▌        | 9/58 [00:00<00:03, 16.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.14 GB):  16%|█▌        | 9/58 [00:00<00:03, 16.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.14 GB):  21%|██        | 12/58 [00:00<00:02, 18.53it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.13 GB):  21%|██        | 12/58 [00:00<00:02, 18.53it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.13 GB):  21%|██        | 12/58 [00:00<00:02, 18.53it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=71.06 GB):  21%|██        | 12/58 [00:00<00:02, 18.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.06 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.09 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.05 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.08 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.09 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.09 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.08 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.07 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.52it/s]

    Capturing num tokens (num_tokens=960 avail_mem=71.08 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.52it/s] Capturing num tokens (num_tokens=896 avail_mem=71.08 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.52it/s]Capturing num tokens (num_tokens=896 avail_mem=71.08 GB):  40%|███▉      | 23/58 [00:01<00:01, 27.50it/s]Capturing num tokens (num_tokens=832 avail_mem=71.07 GB):  40%|███▉      | 23/58 [00:01<00:01, 27.50it/s]Capturing num tokens (num_tokens=768 avail_mem=71.04 GB):  40%|███▉      | 23/58 [00:01<00:01, 27.50it/s]Capturing num tokens (num_tokens=704 avail_mem=71.04 GB):  40%|███▉      | 23/58 [00:01<00:01, 27.50it/s]Capturing num tokens (num_tokens=640 avail_mem=71.03 GB):  40%|███▉      | 23/58 [00:01<00:01, 27.50it/s]Capturing num tokens (num_tokens=640 avail_mem=71.03 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.38it/s]Capturing num tokens (num_tokens=576 avail_mem=71.04 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.38it/s]Capturing num tokens (num_tokens=512 avail_mem=71.02 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.38it/s]

    Capturing num tokens (num_tokens=480 avail_mem=71.04 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.38it/s]Capturing num tokens (num_tokens=448 avail_mem=71.03 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.38it/s]Capturing num tokens (num_tokens=448 avail_mem=71.03 GB):  53%|█████▎    | 31/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=416 avail_mem=71.04 GB):  53%|█████▎    | 31/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=384 avail_mem=71.03 GB):  53%|█████▎    | 31/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=352 avail_mem=71.00 GB):  53%|█████▎    | 31/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=320 avail_mem=70.99 GB):  53%|█████▎    | 31/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=288 avail_mem=71.01 GB):  53%|█████▎    | 31/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=288 avail_mem=71.01 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.47it/s]Capturing num tokens (num_tokens=256 avail_mem=71.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.47it/s]Capturing num tokens (num_tokens=240 avail_mem=71.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.47it/s]

    Capturing num tokens (num_tokens=224 avail_mem=70.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.47it/s]Capturing num tokens (num_tokens=208 avail_mem=70.98 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.47it/s]Capturing num tokens (num_tokens=192 avail_mem=70.96 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.47it/s]Capturing num tokens (num_tokens=192 avail_mem=70.96 GB):  71%|███████   | 41/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=176 avail_mem=70.95 GB):  71%|███████   | 41/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=160 avail_mem=70.97 GB):  71%|███████   | 41/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=144 avail_mem=70.96 GB):  71%|███████   | 41/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=128 avail_mem=70.95 GB):  71%|███████   | 41/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=128 avail_mem=70.95 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.80it/s]Capturing num tokens (num_tokens=112 avail_mem=70.95 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.80it/s]

    Capturing num tokens (num_tokens=96 avail_mem=70.94 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.80it/s] Capturing num tokens (num_tokens=80 avail_mem=70.94 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.80it/s]Capturing num tokens (num_tokens=64 avail_mem=70.93 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.80it/s]Capturing num tokens (num_tokens=64 avail_mem=70.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.30it/s]Capturing num tokens (num_tokens=48 avail_mem=70.90 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.30it/s]Capturing num tokens (num_tokens=32 avail_mem=70.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.30it/s]Capturing num tokens (num_tokens=28 avail_mem=70.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.30it/s]Capturing num tokens (num_tokens=24 avail_mem=70.90 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.30it/s]Capturing num tokens (num_tokens=24 avail_mem=70.90 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=20 avail_mem=70.90 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.34it/s]

    Capturing num tokens (num_tokens=16 avail_mem=70.89 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=12 avail_mem=70.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=8 avail_mem=70.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.34it/s] Capturing num tokens (num_tokens=8 avail_mem=70.88 GB):  98%|█████████▊| 57/58 [00:02<00:00, 36.77it/s]Capturing num tokens (num_tokens=4 avail_mem=70.87 GB):  98%|█████████▊| 57/58 [00:02<00:00, 36.77it/s]Capturing num tokens (num_tokens=4 avail_mem=70.87 GB): 100%|██████████| 58/58 [00:02<00:00, 28.56it/s]


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
    Generated text:  Richard. I have a history of being homesick and depressed when I'm in high school. I went to my school friend's house to help me deal with my sadness. It was a great time and I had a great time at the party. I ended up telling my friend about my depressed mood. My friend said he would make me a necklace. I was so happy. I was so happy and I tried to tell my teacher about it. I was worried my friend would get mad at me. When I told them the truth, I received a gold heart necklace for my school friend. I was very sad. I was very depressed
    ===============================
    Prompt: The president of the United States is
    Generated text:  in a house, and we want to know how many people are in the house. The president is sleeping. The other people are on their phones. One of the people on the phone is answering the phone and asking how many people are in the house. The president is asleep. The other people are not bothering to count. Now, how many people are in the house? 
    (A) 2 people 
    (B) 3 people 
    (C) 4 people 
    (D) 5 people 
    (E) 6 people 
    (F) 7 people
    
    To solve this problem, we need to understand the context and the information given.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Does it follow that if Paris is the capital of France, then all countries in Europe have capitals? 
    Options:
    [+] yes
    [+] it is not possible to tell
    [+] no
    It is not possible to tell
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain. On one hand, we have the promise of pushing the boundaries of what we know about the world and our capabilities, and on the other, we have the potential to build systems that are both powerful and dangerous. As an AI researcher, my job is to work with these complex systems to ensure that they are used ethically and safely. In order to do this, I must be able to understand the nature of AI, the ethical considerations it raises, and the potential risks it may pose. By doing so, I can help to ensure that AI is used for the betterment of society, while also taking steps to mitigate its potential


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and what you're looking for in a job. Let's chat! [Name] [Job Title] [Company Name] [Company Address] [City, State, ZIP Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a city with a rich history and culture. It is the largest city in France and the second-largest city in the European Union. Paris is known for its beautiful architecture, vibrant nightlife, and world-class museums and attractions. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a popular tourist destination and a cultural hub, attracting millions of visitors each year. The city is known for its cuisine, fashion, and art, and is a major center of business and finance in the world. Paris is a city of contrasts,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI systems, there will be a greater focus on ethical AI. This will include developing AI that is designed to be transparent, accountable, and fair, and that is used to make decisions that are in the best interests of the people who use it.
    
    2. Greater integration with other technologies: AI is becoming increasingly integrated with other technologies,
    


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
    Generated text:  Emily. I am 32 years old, and I come from a humble background, but I have a passion for learning and love helping others. I'm always looking for new experiences and learning opportunities, and I'm always eager to share my knowledge with those around me. I love to travel and explore new places, and I'm always eager to discover new things. I'm also a good cook and enjoy trying new recipes. I believe that everyone has the potential to become great at whatever they want to learn, and I'm here to help anyone who is willing to give it a try. I'm not afraid to step out of my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest and most populous city in France and its political, cultural, and economic center. Paris is known for its rich history, beautiful architecture, and diverse cultural scene. It is home to numerous iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a world-renowned destination for tourists, with its charming neighborhoods, trendy cafes, and vibrant festivals. Its position on the river Seine and its location on the Île de la Cité provide access to Europe's most famous landmarks and attractions. The city is known for its art, literature, and music
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a combination of many different trends, including:
    
    1. Personalization: AI will continue to become more personalized, with more accurate predictions and recommendations for users based on their unique preferences and behaviors.
    
    2. Natural Language Processing (NLP): Advances in NLP will make it possible for AI to understand and interpret human language in a way that is more nuanced and context-dependent. This will allow for more natural and human-like interactions between users and AI systems.
    
    3. Autonomous Agents: AI will continue to evolve towards more autonomous agents that can perform a wide range of tasks on their own, without the need for human intervention.
    
    


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

     an

     experienced

     [

    occupation

     or

     hobby

    ]

     with

     over

     [

    number

     of

     years

    ]

     years

     of

     experience

    .

     I

    'm

     a

     [

    specific

     skill

     or

     strength

    ]

     with

     a

     unique

     ability

     that

     set

     me

     apart

     from

     my

     peers

    .

     I

    'm

     always

     looking

     for

     opportunities

     to

     learn

     and

     improve

    ,

     and

     I

     believe

     that

     my

     knowledge

     and

     skills

     will

     make

     a

     positive

     impact

     on

     the

     world

    .

     Thank

     you

     for

     having

     me

     today

    !

     What

    's

     your

     profession

    ,

     and

     how

     have

     you

     gained

     your

     expertise

    ?

     As

     an

     AI

     language

     model

    ,

     my

     skill

    set

     is

     primarily

     in

     natural

     language

     processing

    ,

     but

     I

     have

     also

     been

     trained

     on

     a

     vast

     amount

     of

     text

     data

     to

     understand

     and

     generate

     human

     language

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     rich

     cultural

     heritage

    ,

     iconic

     landmarks

    ,

     and

     vibrant

     street

     life

    .

     The

     city

     boasts

     stunning

     architecture

    ,

     museums

    ,

     restaurants

    ,

     and

     a

     thriving

     art

     scene

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

     Paris

     has

     a

     history

     dating

     back

     over

     

    2

    0

    0

    0

     years

    ,

     from

     the

     ancient

     Roman

     Empire

     to

     the

     Renaissance

     and

     Romantic

     era

    .

     Today

    ,

     it

     is

     a

     hub

     of

     commerce

    ,

     politics

    ,

     and

     culture

    ,

     with

     its

     iconic

     landmarks

     such

     as

     Notre

     Dame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     has

     a

     diverse

     population

     and

     is

     home

     to

     over

     

    7

     million

     residents

    ,

     including

     the

     famous

     Paris

    ian

     café

     culture

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     rapidly

     evolving

    ,

     but

     there

     are

     several

     potential

     trends

     that

     are

     likely

     to

     shape

     the

     technology

     and

     applications

     in

     the

     coming

     years

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

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

     assist

     doctors

     in

     diagnosis

     and

     treatment

     planning

    .

     As

     AI

     technology

     improves

     and

     becomes

     more

     accessible

    ,

     we

     may

     see

     a

     significant

     increase

     in

     its

     use

     in

     healthcare

    ,

     particularly

     in

     areas

     such

     as

     diagnosis

    ,

     drug

     discovery

    ,

     and

     personalized

     medicine

    .
    


    2

    .

     AI

     in

     finance

    :

     AI

     is

     already

     being

     used

     to

     automate

     and

     improve

     financial

     services

    ,

     such

     as

     credit

     scoring

     and

     fraud

     detection

    .

     As

     more

     companies

     adopt

     AI

    -powered

     technology

    ,

    



```python
llm.shutdown()
```
