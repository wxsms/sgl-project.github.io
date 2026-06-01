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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.72it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.42it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.16it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.16it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 22.16it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 22.16it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 22.16it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 22.16it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 22.16it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 22.16it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 22.16it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 22.16it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.42it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.87it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.88it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.88it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.88it/s]

    Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.88it/s] Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.88it/s]Capturing num tokens (num_tokens=832 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.88it/s]Capturing num tokens (num_tokens=832 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=640 avail_mem=71.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=576 avail_mem=71.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.60it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  50%|█████     | 29/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=480 avail_mem=71.18 GB):  50%|█████     | 29/58 [00:00<00:00, 41.62it/s]

    Capturing num tokens (num_tokens=448 avail_mem=71.18 GB):  50%|█████     | 29/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=416 avail_mem=71.18 GB):  50%|█████     | 29/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=384 avail_mem=71.17 GB):  50%|█████     | 29/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=352 avail_mem=71.17 GB):  50%|█████     | 29/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=352 avail_mem=71.17 GB):  59%|█████▊    | 34/58 [00:00<00:00, 38.60it/s]Capturing num tokens (num_tokens=320 avail_mem=71.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 38.60it/s]Capturing num tokens (num_tokens=288 avail_mem=71.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 38.60it/s]Capturing num tokens (num_tokens=256 avail_mem=71.16 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.60it/s]Capturing num tokens (num_tokens=240 avail_mem=71.16 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.60it/s]Capturing num tokens (num_tokens=224 avail_mem=71.15 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.60it/s]Capturing num tokens (num_tokens=224 avail_mem=71.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=208 avail_mem=71.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.53it/s]

    Capturing num tokens (num_tokens=192 avail_mem=71.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=176 avail_mem=71.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=160 avail_mem=71.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=144 avail_mem=71.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=144 avail_mem=71.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=128 avail_mem=71.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=112 avail_mem=71.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=96 avail_mem=71.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.75it/s] Capturing num tokens (num_tokens=80 avail_mem=71.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=64 avail_mem=71.12 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=64 avail_mem=71.12 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=48 avail_mem=71.12 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.85it/s]

    Capturing num tokens (num_tokens=32 avail_mem=71.12 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=28 avail_mem=71.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=24 avail_mem=71.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=20 avail_mem=71.10 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=20 avail_mem=71.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=16 avail_mem=71.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=12 avail_mem=71.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=8 avail_mem=71.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.90it/s] Capturing num tokens (num_tokens=4 avail_mem=71.09 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=4 avail_mem=71.09 GB): 100%|██████████| 58/58 [00:01<00:00, 39.73it/s]


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
    Generated text:  Jessica and I live in the city of Oxford, England. I love art and enjoy looking at the world from a new perspective. I have a love of jazz music and I have a passion for helping people learn about classical music and its traditions.
    I am a great musician and I can play a wide variety of instruments. I like to play jazz and classical music and I am also very happy to play with other musicians.
    I have been a jazz musician for about 10 years now and I have been doing this since I was a child. I learned the piano when I was a baby and I am very good at it. I also
    ===============================
    Prompt: The president of the United States is
    Generated text:  in a wheelchair. He has to use a computer to operate the phone. However, he uses the phone to call his wife, who is also in a wheelchair. One day, the president calls his wife and the phone rings and goes silent in a matter of seconds. Then, the president presses a button, and the phone rings and goes silent again. The phone rings and goes silent for 10 seconds. The president presses a button again, and the phone rings and goes silent for another 10 seconds. Finally, the president presses a button, and the phone rings and goes silent for another 10 seconds.
    
    How many
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Here is the information about the capital city of France:
    
    - Area: 603 km²
    - Population: 2.2 million
    - Economy: based on tourism, industry, services
    - Population density: 2300 people/km²
    - Time zone: UTC+1
    - Capital of the European Union
    - Population (2021): 2.2 million
    - Political: President: Emmanuel Macron
    
    Based on the information above, answer the following questions:
    
    a) Is Paris the capital of France?
    
    b) What is the population of Paris in 2021
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, and as such, AI is an essential tool that will help us to solve many of the world’s most difficult problems. However, like any tool, AI is not without its flaws. One of the most significant is the bias that can be present in its training data, algorithms, and even its programming. In this article, we will take a look at the types of biases present in AI and how they can impact the AI model.
    Bias in AI refers to the existence of differences in the way that algorithms and models are trained on data. These differences can be due to factors such as the way that the data is preprocessed,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I enjoy [job title] because [reason why you enjoy it]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. 
    
    (Note: The statement should be a single, clear sentence that captures the essence of Paris's importance and cultural significance.) 
    
    Please format your response as a JSON object with the following keys and values:
    {
      "city": "Paris",
      "famous_attractions": ["Eiffel Tower", "Notre-Dame Cathedral", "Cultural scene"],
      "historical significance": "Important cultural and historical center"
    } 
    
    Note: The statement should be grammatically correct and include all necessary information. The answer should be
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, AI is likely to play an increasingly important role in shaping the future of work, with the rise of automation and artificial intelligence becoming increasingly prevalent in industries such as manufacturing, finance, and healthcare. Finally, AI is likely to continue to be a topic of public interest and debate, with ongoing discussions about issues such as privacy, bias, and the ethics of AI. Overall,
    


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
    Generated text:  [Name]. I'm a [Age] year old, [Occupation] and I live in [City/Country]. I have a passion for [Favorite Activity/Interest/Subject/Interest]. I believe that [One Thing I Love to Do]. I enjoy [Why I Love It]. If you asked me what I do for a living, I would say that I [What I Do For A Living]. I hope that you're able to meet me and let's get to know each other. That's all for now. Thank you! [Name]
    Hello, my name is [Name]. I'm a [Age] year
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cuisine. The French capital, with a population of over 6. 5 million, is a bustling metropolis with a rich history and vibrant culture. Paris is also known for its cultural institutions, such as the Louvre Museum and the Notre-Dame Cathedral, and its influence on French society and literature. It is a popular tourist destination and a major economic center, with important industries and a rich cultural heritage. Its status as the capital is also recognized by its position in Europe and its role as a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and can be divided into three main trends that are constantly evolving:
    
    1. Deep learning: This is the most advanced form of AI and it is used to solve complex problems that are beyond the capabilities of traditional machine learning models. Deep learning algorithms can recognize patterns in images, speech, and text, and they can even learn to recognize different emotions or human behaviors. Deep learning is currently being applied in various areas such as image recognition, natural language processing, and autonomous vehicles.
    
    2. Natural language processing: Natural language processing is a subset of AI that is used to enable machines to understand and process human language. This includes tasks such as machine


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

    /an

     [

    Age

    ]

     year

     old

     [

    Occup

    ation

    /

    Background

    ].

     I

    ’m

     passionate

     about

     [

    My

     Big

     Thing

    ],

     which

     is

     my

     favorite

     hobby

     that

     I

     like

     to

     spend

     my

     time

     on

    .

     I

     enjoy

     being

     creative

     and

     expressing

     myself

     through

     various

     mediums

    ,

     like

     music

    ,

     painting

    ,

     writing

    ,

     and

     even

     performing

    .

     I

    ’m

     excited

     to

     share

     that

     I

     am

     a

     lifelong

     learner

     and

     enjoy

     taking

     on

     new

     challenges

    .

     I

     am

     always

     ready

     to

     learn

     and

     grow

    ,

     and

     I

    ’m

     always

     looking

     for

     opportunities

     to

     challenge

     myself

     and

     grow

    .

     I

    ’m

     looking

     forward

     to

     meeting

     you

     and

     learning

     more

     about

     you

    .
    


    Please

     provide

     feedback

     on

     my

     self

    -int

    roduction

    .

     Based

     on

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historical

     and

     cultural

     center

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     cuisine

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     Site

    ,

     and

     its

     iconic

     landmarks

     include

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

     the

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     cuisine

    ,

     and

     music

    .

     It

     is

     a

     major

     tourist

     destination

    ,

     with

     millions

     of

     visitors

     annually

    .

     Paris

     is

     home

     to

     many

     popular

     attractions

     such

     as

     the

     Lou

    vre

    ,

     Mont

    mart

    re

    ,

     and

     Mont

    par

    n

    asse

    .

     It

     is

     also

     home

     to

     the

     iconic

     "

    Two

     Towers

    "

     monument

    ,

     which

     is

     located

     in

     the

     heart

     of

     the

     city

    .

     In

     summary

    ,

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     potential

     trends

     that

     will

     shape

     how

     it

     is

     used

     and

     developed

    .

     Some

     of

     the

     most

     promising

     trends

     include

    :
    


    1

    .

     Increased

     automation

     and

     efficiency

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

     will

     be

     able

     to

     perform

     a

     wider

     range

     of

     tasks

     more

     efficiently

     and

     with

     greater

     accuracy

     than

     humans

    .

     This

     will

     lead

     to

     increased

     productivity

     and

     fewer

     human

     workers

    ,

     which

     can

     be

     both

     positive

     and

     negative

     depending

     on

     how

     it

     is

     used

    .
    


    2

    .

     Enhanced

     human

     abilities

    :

     AI

     will

     continue

     to

     improve

     and

     expand

     in

     capabilities

    ,

     which

     will

     enable

     it

     to

     perform

     more

     complex

     tasks

     and

     find

     new

     ways

     to

     interact

     with

     humans

    .

     This

     could

     lead

     to

     new

     forms

    



```python
llm.shutdown()
```
