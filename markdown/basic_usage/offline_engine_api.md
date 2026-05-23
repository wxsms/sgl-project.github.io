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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.97it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.44it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.94it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.16it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.16it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.16it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.16it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.16it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.16it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.16it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.16it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.16it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.16it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 23.16it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.12it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.12it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.12it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.12it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.12it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.12it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.12it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.12it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:04, 13.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:04, 13.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:04, 13.80it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:04, 13.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:03, 16.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:03, 16.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:03, 16.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:03, 16.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:03, 16.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.61it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.21it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.21it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.21it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.21it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.21it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.21it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.33it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.33it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.33it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.33it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.33it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.33it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 42.40it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 42.40it/s]

    Capturing num tokens (num_tokens=448 avail_mem=75.75 GB):  50%|█████     | 29/58 [00:00<00:00, 42.40it/s]Capturing num tokens (num_tokens=416 avail_mem=75.75 GB):  50%|█████     | 29/58 [00:00<00:00, 42.40it/s]Capturing num tokens (num_tokens=384 avail_mem=75.67 GB):  50%|█████     | 29/58 [00:00<00:00, 42.40it/s]Capturing num tokens (num_tokens=352 avail_mem=75.04 GB):  50%|█████     | 29/58 [00:00<00:00, 42.40it/s]Capturing num tokens (num_tokens=352 avail_mem=75.04 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.41it/s]Capturing num tokens (num_tokens=320 avail_mem=75.03 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.41it/s]Capturing num tokens (num_tokens=288 avail_mem=75.03 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.41it/s]Capturing num tokens (num_tokens=256 avail_mem=75.03 GB):  59%|█████▊    | 34/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=240 avail_mem=75.02 GB):  59%|█████▊    | 34/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=224 avail_mem=75.02 GB):  59%|█████▊    | 34/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=208 avail_mem=75.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=208 avail_mem=75.01 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=192 avail_mem=75.01 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.58it/s]

    Capturing num tokens (num_tokens=176 avail_mem=75.01 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=160 avail_mem=75.01 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=144 avail_mem=75.00 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=128 avail_mem=75.00 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=128 avail_mem=75.00 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.41it/s]Capturing num tokens (num_tokens=112 avail_mem=75.00 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.41it/s]Capturing num tokens (num_tokens=96 avail_mem=75.00 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.41it/s] Capturing num tokens (num_tokens=80 avail_mem=74.99 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.41it/s]Capturing num tokens (num_tokens=64 avail_mem=74.99 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.41it/s]Capturing num tokens (num_tokens=48 avail_mem=74.99 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.41it/s]Capturing num tokens (num_tokens=48 avail_mem=74.99 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=32 avail_mem=74.98 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.66it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.98 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=24 avail_mem=74.98 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=20 avail_mem=74.97 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=16 avail_mem=74.97 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=16 avail_mem=74.97 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=12 avail_mem=74.97 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=8 avail_mem=74.96 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.09it/s] Capturing num tokens (num_tokens=4 avail_mem=74.96 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=4 avail_mem=74.96 GB): 100%|██████████| 58/58 [00:01<00:00, 39.29it/s]


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
    Generated text:  Tasha and I'm 22 years old, I'm pretty healthy and I love to read. I have a lot of friends and I often make friends with people who are also interested in books and books. I love to learn new things and I'm always curious. I often spend time exploring new places and trying new foods. What are some of your favorite books or movies that you enjoy reading or watching? I really enjoy going to new places and trying new foods.
    Certainly! I'd be happy to share a few books and movies with you that you might enjoy reading or watching. Here are a couple of my favorites:
    
    ### Books
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to improve the budget deficit. He has decided to balance the budget. He plans to decrease the deficit by increasing government revenue by 30% and decreasing government expenses by 20%. If the deficit is currently $500 billion, what is the new budget deficit? Let's denote the current deficit by \( D \). According to the problem, the current deficit is $500 billion. The president plans to balance the budget by increasing government revenue by 30% and decreasing government expenses by 20%. Let's denote the new revenue by \( R \) and the new expenses by \( E \).
    
    
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris  
    B) London  
    C) Rome  
    D) Berlin
    
    To determine the capital of France, let's start by recalling the official definition of a capital city, which is the largest city in a country's administrative divisions. The United Kingdom officially defines a capital city as the largest city in its administrative divisions.
    
    The United Kingdom consists of two administrative divisions: the North America (United Kingdom) and the British Isles (North Ireland).
    
    1. The North America (United Kingdom) includes the following cities: London, Belfast, Cardiff, Edinburgh, Liverpool, Glasgow, Birmingham, Manchester, and York.
    2. The British Isles
    ===============================
    Prompt: The future of AI is
    Generated text:  in the making. We are already seeing the impact of AI on our lives from face recognition, chatbots, to smart home devices. While AI is creating exciting new opportunities, it also poses a serious challenge to society and humanity. AI is already disrupting business, government, and society as we know it. When this disruption happens at a human level, it poses a serious threat to humanity. The challenge of AI is to find a balance between the benefits and harms of AI. If we fail to do this, the consequences could be catastrophic.
    Our focus in this blog post is on the importance of AI ethics. We will look at how AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your job or profession]. I enjoy [insert a brief description of your hobbies or interests]. What's your favorite [insert a short description of your hobby or interest]. I'm always looking for ways to [insert a short description of a new skill or hobby you're learning]. Thank you for having me. [Name] [Company name] [Job title] [Company website] [LinkedIn profile] [Twitter handle
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination, known for its rich history, art, and cuisine. The city is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a vibrant and diverse city with a rich cultural heritage, making it a popular destination for both locals and tourists alike. The city is also known for its cuisine, with a wide range
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in even more areas,
    


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
    Generated text:  [Name]. I'm a [fill in the blank] with [number] [fill in the blank] experience. If you would like to know more about me, I'd be happy to share my [specific skills, interests, or knowledge] with you. What do you want to know? [Name]: "Hi, I'm [Name] and I'm a [fill in the blank] with [number] [fill in the blank] experience. If you would like to know more about me, I'd be happy to share my [specific skills, interests, or knowledge] with you. What do you want to know
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    I'm ready to help! Do you have a specific question about Paris that you'd like me to answer? Let me know and I'll do my best to provide you with accurate information. How can I assist you today? [Type your question here] (Note: For real-time translation services, use the "Translate to French" option) Paris is the capital city of France, located on the western bank of the Seine River, near the city of Paris. It is the largest city in France and the largest metropolitan area in Europe. The city is known for its rich history, stunning architecture, and vibrant culture. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and will depend on many factors, including technological advancements, societal changes, and the emergence of new technologies that could have profound impacts on society. Here are some potential future trends in AI that are currently being explored and discussed:
    
    1. Increased Automation: As technology continues to advance, we are likely to see an increase in the automation of jobs, especially in industries that rely on repetitive tasks. AI-powered automation systems will likely replace humans in a variety of tasks, from manufacturing to service industries.
    
    2. Improved Personalized AI: AI-powered personalized learning will become more prevalent, as machines can learn and adapt to individual learning styles and needs. This


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

     software

     engineer

     with

     over

     

    1

    0

     years

     of

     experience

     in

     developing

     cutting

    -edge

     software

     solutions

    .

     I

     have

     a

     keen

     eye

     for

     detail

    ,

     a

     passion

     for

     learning

     new

     technologies

    ,

     and

     a

     track

     record

     of

     creating

     innovative

     solutions

     that

     help

     solve

     complex

     problems

    .

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     learn

     more

    .

     Thanks

     for

     having

     me

    .

     Here

     is

     a

     sample

     self

    -int

    roduction

    :


    "

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

     software

     engineer

     with

     over

     

    1

    0

     years

     of

     experience

     in

     developing

     cutting

    -edge

     software

     solutions

    .

     As

     a

     seasoned

     professional

     with

     a

     strong

     background

     in

     computer

     science

     and

     engineering

    ,

     I

     have

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     historical

     landmarks

    ,

     vibrant

     arts

     scene

    ,

     and

     cosm

    opolitan

     atmosphere

    .
    


    Paris

    ,

     the

     cultural

     and

     historical

     capital

     of

     France

    ,

     is

     renowned

     for

     its

     rich

     history

    ,

     vibrant

     arts

     scene

    ,

     and

     cosm

    opolitan

     atmosphere

    .

     Its

     iconic

     landmarks

     include

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    ,

     while

     the

     city

     hosts

     a

     wide

     range

     of

     cultural

     events

    ,

     including

     world

    -ren

    owned

     festivals

     and

     street

     performances

    .

     Paris

     also

     boasts

     a

     vibrant

     arts

     scene

     with

     numerous

     galleries

    ,

     cafes

    ,

     and

     theaters

    ,

     making

     it

     a

     popular

     destination

     for

     art

     lovers

     and

     tourists

     alike

    .

     Despite

     its

     cosm

    opolitan

     vibe

    ,

     Paris

     remains

     a

     city

     of

     historical

     significance

     and

     continues

     to

     attract

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     several

     trends

    ,

     including

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     and

     societal

     implications

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

     there

     will

     be

     increased

     pressure

     to

     address

     the

     ethical

     and

     societal

     implications

     of

     AI

    .

     This

     includes

     concerns

     about

     privacy

    ,

     bias

    ,

     transparency

    ,

     and

     the

     potential

     for

     AI

     to

     out

    smart

     human

     beings

    .
    


    2

    .

     AI

     integration

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     become

     more

     integrated

     with

     other

     technologies

    ,

     including

     blockchain

    ,

     artificial

     intelligence

    ,

     and

     

    5

    G

     networks

    .

     This

     integration

     could

     lead

     to

     new

     applications

     and

     technologies

    ,

     such

     as

     the

     use

     of

     AI

     to

     power

     autonomous

     vehicles

    .
    


    3

    .

     AI

    -as

    -a

    -

    Service

     (

    aaS

    )

    



```python
llm.shutdown()
```
