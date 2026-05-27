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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:43,  3.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:43,  3.93s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:43,  3.93s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:43,  3.93s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:43,  3.93s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  19%|█▉        | 11/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 10.42it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03, 10.42it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03, 10.42it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.95it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.95it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.95it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.95it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.95it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.95it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.95it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.95it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.95it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.95it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 16.95it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 25.53it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 25.53it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 25.53it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 25.53it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 25.53it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 25.53it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 25.53it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 25.53it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 25.53it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 25.53it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 25.53it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.64 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.64 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=54.64 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.64 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=54.64 GB):   9%|▊         | 5/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.63 GB):   9%|▊         | 5/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=54.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.47it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.47it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.47it/s]Capturing num tokens (num_tokens=960 avail_mem=54.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.47it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=54.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.47it/s]Capturing num tokens (num_tokens=832 avail_mem=54.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.47it/s]Capturing num tokens (num_tokens=832 avail_mem=54.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.55it/s]Capturing num tokens (num_tokens=768 avail_mem=54.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.55it/s]Capturing num tokens (num_tokens=704 avail_mem=54.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.55it/s]Capturing num tokens (num_tokens=640 avail_mem=54.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.55it/s]Capturing num tokens (num_tokens=576 avail_mem=54.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.55it/s]Capturing num tokens (num_tokens=512 avail_mem=54.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.55it/s]Capturing num tokens (num_tokens=512 avail_mem=54.54 GB):  50%|█████     | 29/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=480 avail_mem=54.56 GB):  50%|█████     | 29/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=448 avail_mem=54.55 GB):  50%|█████     | 29/58 [00:00<00:00, 39.22it/s]

    Capturing num tokens (num_tokens=416 avail_mem=54.55 GB):  50%|█████     | 29/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=384 avail_mem=54.55 GB):  50%|█████     | 29/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=352 avail_mem=54.54 GB):  50%|█████     | 29/58 [00:00<00:00, 39.22it/s]Capturing num tokens (num_tokens=352 avail_mem=54.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=320 avail_mem=54.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=288 avail_mem=54.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=256 avail_mem=54.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=240 avail_mem=54.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.60it/s]Capturing num tokens (num_tokens=224 avail_mem=54.53 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=224 avail_mem=54.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=208 avail_mem=54.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=192 avail_mem=54.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.46it/s]

    Capturing num tokens (num_tokens=176 avail_mem=54.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=160 avail_mem=54.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=144 avail_mem=54.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=144 avail_mem=54.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=128 avail_mem=54.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=112 avail_mem=54.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=96 avail_mem=54.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s] Capturing num tokens (num_tokens=80 avail_mem=54.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=64 avail_mem=54.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=64 avail_mem=54.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=48 avail_mem=54.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=32 avail_mem=54.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]

    Capturing num tokens (num_tokens=28 avail_mem=54.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=24 avail_mem=54.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=20 avail_mem=54.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=20 avail_mem=54.48 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.58it/s]Capturing num tokens (num_tokens=16 avail_mem=54.48 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.58it/s]Capturing num tokens (num_tokens=12 avail_mem=54.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.58it/s]Capturing num tokens (num_tokens=8 avail_mem=54.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.58it/s] Capturing num tokens (num_tokens=4 avail_mem=54.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.58it/s]Capturing num tokens (num_tokens=4 avail_mem=54.47 GB): 100%|██████████| 58/58 [00:01<00:00, 40.01it/s]


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
    Generated text:  Eddie and I am a computer science student at the University of California, Berkeley. I am currently in my third year of my program, and I am interested in the field of Artificial Intelligence and Machine Learning. I have been working on a project that involves building a neural network that can predict the outcome of a chess game. I am looking for some advice on how to optimize my project, specifically on how to handle the time complexity of the algorithm. Do you have any insights or recommendations for me to consider? Additionally, I am curious about the role of machine learning and neural networks in modern gaming, and I am interested in learning more about the
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military uniforms to issue to his army.
    
    There are two types of uniforms: Type A, which consists of 100 individual shirts and 200 individual pants; and Type B, which consists of 100 individual shirts and 100 individual pants. Each shirt is a square with side length 1 foot, and each pant has a square shape with side length 1 foot as well. All the shirts and pants are of the same material and the same size, but the type of shirt matters.
    
    The president wants to ensure that at least one uniform is needed. However, he wants
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. 
    
    Hugo wants to build a bridge over a river to connect two opposite banks of the river. The river is wide enough to accommodate the bridge over the river. The river is a straight line, and the two banks have the same width. The banks are not flat, but are hills. Hugo can only build the bridge on the riverbank that is above the banks.
    
    What is the minimum length of the bridge that Hugo needs to build?
    
    To determine the minimum length of the bridge Hugo needs to build, we need to understand the constraints and the nature of the river and the hills. The key is to recognize that the river
    ===============================
    Prompt: The future of AI is
    Generated text:  digital. The digital age is not only about the digital world, but also about the digital tools that are used for research, development, and deployment in artificial intelligence.
    
    The need for digital tools is constant, and the need for digital tools is not only for research, development, and deployment, but also for understanding and managing AI models. Therefore, it is necessary to introduce a new tool for understanding and managing AI models.
    
    In this context, the paper presents an overview of the tools that can be used to understand and manage AI models, the role of the digital tool in this context, and the challenges that must be overcome to develop a new


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


    Generated text:  Paris. It is the largest city in Europe and the third largest city in the world. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also a major financial center and a major tourist destination. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a city that has a rich cultural heritage and is a major center of learning and innovation. It is a city that is constantly evolving and changing, with new developments and cultural events taking place throughout the year. Paris is a city that is a true reflection of France's
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be increased focus on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions between humans and machines. This could lead to more personalized and adaptive AI systems.
    
    3. Increased use of AI in healthcare: AI is already being
    


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
    Generated text:  [insert first and last name], and I'm a [insert occupation or profession here]. I'm a [insert major field or area of interest here]. I have always been interested in [insert interest or hobby here], which has driven me to pursue [insert related activities here], and I've always been fascinated by [insert related topic here]. I'm looking forward to meeting you and finding out more about you! [insert how you got to know me here]. I hope we can have a conversation that will be both informative and enjoyable. [insert how you would like to be introduced here]. I'm excited to discuss this topic and learn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Sure! Paris is the capital city of France and is known for its beautiful museums, iconic landmarks, and rich history. Here are some key facts about Paris:
    
    1. Location: Paris is the capital of France and is located in the northwestern part of the country.
    
    2. Population: As of 2021, the population of Paris is approximately 2.3 million people.
    
    3. Capital: Paris is the capital city of France and its language is French.
    
    4. Official languages: French is the main language, but other official languages include French, French, and English.
    
    5. Culture: Paris is known
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by several key trends:
    
    1. Automation: AI will continue to automate tasks, from manufacturing and logistics to customer service and administrative roles. This trend will likely become more widespread as AI technology advances and becomes more integrated into everyday life.
    
    2. Ethical and ethical concerns: AI will increasingly be used for applications that could have negative consequences for society. Ethical considerations will be a key factor in the development of AI, as companies will need to carefully consider the potential impacts of their AI solutions.
    
    3. Data privacy and security: With the increasing amount of data being generated by AI, there will be a growing concern over data


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

    name

    ],

     and

     I

    'm

     a

     [

    career

     field

    /

    role

    ]

    !

     I

    've

     been

     in

     the

     industry

     for

     [

    number

    ]

     years

    ,

     and

     I

    've

     gained

     a

     lot

     of

     knowledge

     and

     experience

     in

     my

     field

    ,

     while

     also

     expanding

     my

     skill

     set

     and

     helping

     other

     professionals

    .

     I

    'm

     always

     looking

     for

     new

     ways

     to

     improve

     and

     learn

     new

     things

    ,

     and

     I

    'm

     always

     open

     to

     new

     challenges

     and

     opportunities

    .

     How

     can

     I

     assist

     you

     today

    ?

     [

    Your

     name

    ]

    !

     [

    Your

     title

    ]

     |

     [

    Your

     profession

    ]

     |

     [

    Your

     specialty

    ]

     |

     [

    Your

     strengths

    /

    adv

    antages

    ]

     [

    Your

     weaknesses

    /

    areas

     for

     improvement

    ]


    [

    Name

    ]:

     Hello

    ,

     my

     name

     is

     [

    name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     

    1

    8

    th

     most

     populous

     city

     in

     the

     world

    .

     Paris

     is

     known

     for

     its

     iconic

     landmarks

    ,

     vibrant

     nightlife

    ,

     and

     art

     museums

    .

     The

     city

     has

     a

     rich

     history

     and

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     It

     is

     a

     major

     transportation

     hub

     and

     a

     major

     financial

     center

     in

     Europe

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     often

     referred

     to

     as

     the

     "

    city

     of

     a

     thousand

     gardens

    ."

     It

     is

     also

     home

     to

     several

     major

     universities

    ,

     including

     the

     É

    cole

     poly

    techn

    ique

    .

     The

     city

     has

     a

     diverse

     population

     and

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     it

     is

     difficult

     to

     predict

     exactly

     what

     trends

     will

     emerge

    .

     However

    ,

     based

     on

     current

     trends

     and

     developments

     in

     the

     field

    ,

     there

     are

     some

     possible

     future

     trends

     that

     could

     potentially

     affect

     AI

     in

     the

     coming

     years

    .
    


    1

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     have

     already

     begun

     to

     emerge

     in

     the

     market

    ,

     and

     they

     are

     expected

     to

     continue

     to

     grow

     in

     popularity

     as

     their

     technology

     becomes

     more

     advanced

    .

     These

     vehicles

     will

     be

     able

     to

     navigate

     roads

    ,

     roads

    ,

     and

     public

     transport

     systems

    ,

     and

     they

     will

     be

     able

     to

     communicate

     with

     each

     other

     and

     the

     road

     infrastructure

    .

     Autonomous

     vehicles

     will

     also

     be

     able

     to

     make

     decisions

     on

     their

     own

     and

     reduce

     the

     number

     of

     human

     drivers

     on

     the

     road

    



```python
llm.shutdown()
```
