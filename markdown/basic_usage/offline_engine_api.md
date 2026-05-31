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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 14.87it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 14.87it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.69it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.69it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.69it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.69it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.69it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.69it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.69it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.69it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.69it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.69it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.56it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.56it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  21%|██        | 12/58 [00:00<00:01, 28.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.51it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.51it/s] Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.22it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.24it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.24it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.24it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.24it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.24it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.24it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.39it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.39it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.39it/s]

    Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.59it/s] Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 43.14it/s]

    Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  81%|████████  | 47/58 [00:01<00:00, 43.14it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.89it/s] Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.89it/s]


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
    Generated text:  Alex Johnson. I am the President of the Major League Baseball Centralization Network, a non-profit organization that is dedicated to promoting the idea that baseball should be centralized in an organized manner. Our objective is to promote the benefits of a centralized baseball system that supports efficient and fair playing fields for all players, teams, and fans, while also addressing the concerns of those who are concerned about the loss of homegrown talent and the need for increased competition. Let me know if you would like to know more about me or the Major League Baseball Centralization Network. [ ] Hello, my name is Alex Johnson. I am the President of the Major League
    ===============================
    Prompt: The president of the United States is
    Generated text:  a candidate of the Democratic Party. Which of the following is the political party that the president is affiliated with? A. Republicans B. Democrats C. Independents D. None of the above
    Answer:
    
    B
    
    The principle that ensures the safe operation and maintenance of power systems and the reliable supply of electric energy is called ____.
    A. Safety
    B. Standardization
    C. Legality
    D. Economic
    Answer:
    
    A
    
    1. Which of the following is NOT a correct method for reducing the risk of coughing and coughing spells? 
    A. Avoid smoking
    B. Avoid drinking strong tea
    C. Reduce salt
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the northwestern part of the country. The capital is Paris. It is the largest city in Europe. France is the third largest country by land area in the world. The country is divided into 17 regions. The capital is surrounded by eight other regions. Its population is about 6. 4 million. It is the third largest metropolitan area in the world.
    
    What are the capital of France? The capital of France is Paris. It is located in the northwestern part of the country, surrounded by eight other regions. Its population is about 6. 4 million and it is the third largest metropolitan area in
    ===============================
    Prompt: The future of AI is
    Generated text:  more complex than ever. In the past, AI systems were limited by the availability of data and the predictability of the algorithms used. However, with the increasing availability of data and the rise of large-scale machine learning models, the scope of AI systems has expanded dramatically. With the ability to learn from large amounts of data and the flexibility to adapt to changing conditions, AI systems are now able to solve complex problems in a way that was previously impossible.
    One of the most powerful features of AI systems is their ability to understand and interpret complex data. Machine learning algorithms are able to identify patterns and relationships in data that humans may not be able to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its fashion industry, art scene, and its role in French politics and diplomacy. It is the largest city in France and is home to many of the country's most famous landmarks and attractions. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more personalized and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent
    


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
    Generated text:  [Name] and I'm a/an [occupation] by profession. I'm passionate about [career goal or hobby]. I enjoy [activity or challenge] and I love [occupation or hobby]. I believe that [reason for success or strength of character]. I'm always looking for opportunities to [personal growth goal or step]. I believe in [philosophy or principle]. I believe that I can [strength or ability] [reason for success or strength of character]. I'm [age], [gender], [nationality], and [place of birth]. I'm a/an [occupation] by profession. I'm passionate about [career
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light and the City of Light. It is a metropolitan area and includes a city, a metropolitan region, and a metropolitan area. France’s capital city, located on the Île de France, is situated at 49°44′N latitude and 2°57′E longitude. It is one of the largest and most populous cities in Europe, and is home to many world-renowned landmarks and cultural institutions. The city is also known for its rich history, including the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral. Paris has a diverse
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and will likely continue to evolve in a number of ways. Some of the possible future trends in AI include:
    
    1. Increased use of AI in healthcare: AI is already being used in the healthcare industry to automate the diagnostic process, reduce errors, and improve patient outcomes. As AI becomes more advanced, we may see even more personalized and accurate diagnoses, as well as more precise treatment plans.
    
    2. Expansion of AI in finance: With the growing importance of AI in finance, we may see more applications of AI in areas like fraud detection, trading, and investment management. As AI becomes more integrated into the financial industry, we may see


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

     Jane

    .

     I

    'm

     a

     

    2

    5

    -year

    -old

     software

     developer

     who

     enjoys

     working

     with

     technology

     and

     coding

    .

     I

     have

     a

     keen

     interest

     in

     AI

     and

     machine

     learning

     and

     often

     volunteer

     my

     time

     as

     a

     mentor

     to

     beginners

    .

     I

    'm

     always

     up

     for

     a

     challenge

    ,

     and

     I

     enjoy

     solving

     complex

     problems

     and

     coming

     up

     with

     innovative

     solutions

    .

     Whether

     I

    'm

     coding

     on

     my

     computer

     or

     solving

     a

     puzzle

    ,

     I

    'm

     always

     ready

     to

     make

     a

     difference

    .

     How

     about

     you

    ?

     What

    's

     your

     name

     and

     what

     do

     you

     do

    ?

     I

    'd

     love

     to

     hear

     about

     your

     experiences

     and

     what

     you

    're

     up

     to

    .

     
    


    Remember

    ,

     I

     want

     to

     make

     a

     good

     first

     impression

    ,

     so

     please

     add

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     located

     in

     the

     eastern

     part

     of

     the

     country

    .

     It

     is

     one

     of

     the

     most

     populous

     cities

     in

     the

     world

    ,

     with

     an

     estimated

     population

     of

     over

     

    2

    0

     million

     people

    ,

     making

     it

     one

     of

     the

     largest

     cities

     in

     Europe

    .

     Paris

     is

     renowned

     for

     its

     romantic

     architecture

    ,

     diverse

     cultural

     scene

    ,

     and

     world

    -ren

    owned

     art

     and

     music

     scenes

    .

     It

     is

     also

     a

     hub

     for

     fashion

    ,

     cinema

    ,

     and

     other

     creative

     industries

    .

     Paris

     is

     known

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

     beautiful

     scenery

    ,

     making

     it

     a

     popular

     tourist

     destination

     worldwide

    .

     The

     city

     has

     a

     rich

     cultural

     heritage

     and

     is

     home

     to

     many

     important

     landmarks

     and

     museums

    ,

     including

     the

     Lou

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     unpredictable

     and

     subject

     to

     many

     factors

    ,

     including

     advancements

     in

     technology

    ,

     shifts

     in

     societal

     values

    ,

     and

     the

     development

     of

     new

     technologies

    .

     However

    ,

     here

     are

     some

     possible

     trends

     that

     may

     be

     seen

     in

     the

     future

     of

     artificial

     intelligence

    :
    


    1

    .

     Increased

     AI

     autonomy

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

     likely

     be

     a

     growing

     trend

     towards

     more

     autonomous

     AI

     systems

    .

     This

     could

     mean

     that

     machines

     can

     make

     decisions

     and

     actions

     without

     human

     intervention

    .
    


    2

    .

     AI

     with

     emotional

     intelligence

    :

     AI

     is

     already

     becoming

     more

     capable

     of

     understanding

     and

     processing

     human

     emotions

    ,

     and

     there

    's

     potential

     to

     further

     develop

     this

     capability

    .

     AI

     systems

     could

     learn

     to

     understand

     human

     emotions

     and

     respond

     appropriately

    ,

     such

    



```python
llm.shutdown()
```
