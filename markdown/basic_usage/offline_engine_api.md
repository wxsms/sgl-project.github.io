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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.71it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03, 10.37it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03, 10.37it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03, 10.37it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03, 10.37it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03, 10.37it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03, 10.37it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:03, 10.37it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:03, 10.37it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:01, 16.16it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:01, 16.16it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:01, 16.16it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:01, 16.16it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:01, 16.16it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:01, 16.16it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:04<00:01, 16.16it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:04<00:01, 16.16it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:04<00:01, 16.16it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:00, 23.82it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:05<00:00, 23.82it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 33.53it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:05<00:00, 33.53it/s]

    Compiling num tokens (num_tokens=20):  76%|███████▌  | 44/58 [00:05<00:00, 33.53it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 44.61it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 44.61it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 44.61it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 44.61it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 44.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.21it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.21 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.20 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.13 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=960 avail_mem=74.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.14 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=832 avail_mem=74.14 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=832 avail_mem=74.14 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.56it/s]Capturing num tokens (num_tokens=768 avail_mem=74.14 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.56it/s]Capturing num tokens (num_tokens=704 avail_mem=74.13 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.56it/s]Capturing num tokens (num_tokens=640 avail_mem=74.13 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.56it/s]Capturing num tokens (num_tokens=576 avail_mem=74.13 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.56it/s]Capturing num tokens (num_tokens=512 avail_mem=74.11 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.56it/s]Capturing num tokens (num_tokens=512 avail_mem=74.11 GB):  50%|█████     | 29/58 [00:00<00:00, 42.72it/s]Capturing num tokens (num_tokens=480 avail_mem=74.13 GB):  50%|█████     | 29/58 [00:00<00:00, 42.72it/s]Capturing num tokens (num_tokens=448 avail_mem=74.13 GB):  50%|█████     | 29/58 [00:00<00:00, 42.72it/s]Capturing num tokens (num_tokens=416 avail_mem=74.13 GB):  50%|█████     | 29/58 [00:00<00:00, 42.72it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.12 GB):  50%|█████     | 29/58 [00:00<00:00, 42.72it/s]Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  50%|█████     | 29/58 [00:00<00:00, 42.72it/s]Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=320 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=288 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=256 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=208 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=192 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.35it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=144 avail_mem=74.09 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=144 avail_mem=74.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.14it/s] Capturing num tokens (num_tokens=80 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.96it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.80it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 40.66it/s]


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
    Generated text:  Kaisa. I'm a high school student from the United States. I'm a big fan of technology, especially computers and smart phones. I have a lot of friends who use computers and smart phones. I have a lot of knowledge about technology. And I have a lot of new experiences to tell you. My hobbies are movies, music, and sports. My favorite movie is "The Big Lebowski" and I like to watch movies on my phone. My favorite music is R&B and I love to listen to my music on my phone. My favorite sport is tennis. It's my passion. I have a lot of
    ===============================
    Prompt: The president of the United States is
    Generated text:  a representative of which country?
    A) United Kingdom
    B) United States
    C) Canada
    D) Europe
    
    The president of the United States is the head of state of the United States. He is the president of the United States, which is a representative of the United States.
    Answer: B
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city in Europe and is the capital of the department of the Île de France. France is the fifth largest country in the world by land area. It is mostly a mountainous country. The population of Paris is about 2.5 million people. It is the most populous city in France. It has a population of about 2.3 million people. 
    
    What is the population of Paris? The population of Paris is about 2.5 million people.
    You are a helpful assistant with my software project. Can you write a code snippet that can tell me the number of days between two dates?
    ===============================
    Prompt: The future of AI is
    Generated text:  both exciting and challenging. What are the major areas of development and what are the potential impacts on society and individuals?
    
    AI is a rapidly advancing technology that is transforming the way we live, work, and interact with each other. It has the potential to revolutionize many aspects of our lives, from healthcare to transportation to entertainment. However, it is also poised to have significant impacts on society and individuals, both positive and negative.
    
    One of the most exciting areas of development for AI is the ability of machines to learn and adapt. This technology has the potential to revolutionize industries like healthcare, finance, and education, where it can help automate routine


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [industry]. I'm a [job title] at [company name], and I'm always looking for ways to [describe a goal or activity]. I'm a [job title] at [company name], and I'm always looking for ways to [describe a goal or activity]. I'm a [job title] at [company name], and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination, attracting millions of visitors each year. The city is known for its rich history, diverse culture, and vibrant nightlife. It is also home to many famous French artists, writers, and musicians. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also known for its cuisine, with its famous
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI becomes more prevalent in various industries, there will be a greater emphasis on ethical considerations and the responsible use of AI.
    
    3. Development of more advanced AI: AI is likely to become more advanced and capable of performing tasks that were previously thought to be beyond the capabilities of humans.
    
    4. Increased use of AI in healthcare: AI is likely to be used in healthcare to improve patient
    


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
    Generated text:  [Name]. I'm a friendly and outgoing person who enjoys exploring the outdoors and meeting new people. Whether it's hiking through the woods or attending a charity event, I'm always up for adventure. I like to keep a positive attitude and be happy to help others, no matter how small. My favorite hobby is gardening, and I enjoy spending time in the garden with my friends and family. I love to read and learn new things, and I'm always up for a good book. I'm excited to meet someone new and have a meaningful conversation. Let's get to know each other better! [Name] [Address] (Phone
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the most populous city and a major economic center, with a rich historical and cultural heritage. The city is known for its art and architecture, including the Eiffel Tower and the Louvre Museum. Paris is also famous for its fashion industry, music, and culinary traditions. The city is home to many renowned universities and prestigious museums. Paris is the cultural capital of France and one of the world's most influential cities. It has a diverse and vibrant community of residents, including a large French-speaking population and many immigrants from other countries. The city is also known for its prestigious gastronomy, with many renowned restaurants, food
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting, with potential applications in numerous fields, including healthcare, transportation, and energy. Here are some possible trends in AI in the coming years:
    
    1. Increased focus on ethical considerations: AI is increasingly being used in applications that have significant implications for society, such as autonomous vehicles and facial recognition. As these technologies become more widespread, there will be increasing focus on ethical considerations, including issues related to bias, privacy, and transparency.
    
    2. Greater emphasis on machine learning and deep learning: As AI becomes more advanced, there will be an increasing emphasis on machine learning and deep learning, which are techniques that allow machines to learn from data and make


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

    insert

     name

    ]

     and

     I

    'm

     a

     [

    insert

     profession

     or

     occupation

    ]

     with

     [

    insert

     professional

     title

     or

     work

     experience

    ],

     and

     I

    'm

     currently

     pursuing

     [

    insert

     degree

     or

     certification

    ].

     I

     have

     a

     passion

     for

     [

    insert

     area

     of

     interest

     or

     hobby

    ],

     and

     I

    'm

     always

     eager

     to

     learn

     more

     about

     [

    insert

     field

     of

     interest

    ].

     In

     my

     free

     time

    ,

     I

     enjoy

     [

    insert

     interests

     or

     activities

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    insert

     hobbies

     or

     interests

    ].

     I

    'm

     always

     open

     to

     new

     experiences

     and

     excited

     to

     learn

     more

     about

     the

     world

     around

     me

    .

     What

    's

     your

     background

    ,

     and

     what

     brings

     you

     here

     to

     work

     with

     us

    ?
    


    Remember

    ,

     you

    're

     not

     just

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     of

     love

    ,

     romance

    ,

     and

     art

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     iconic

     landmarks

     such

     as

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

     food

     scene

    ,

     and

     international

     cuisine

    .

     Visitors

     to

     Paris

     can

     experience

     the

     city

    's

     diverse

     culture

    ,

     from

     traditional

     cuisine

     to

     modern

     art

    .

     The

     French

     love

     to

     celebrate

     in

     the

     evening

     with

     par

    ades

    ,

     concerts

    ,

     and

     festivals

    ,

     making

     Paris

     one

     of

     the

     most

     famous

     cities

     in

     the

     world

    .

     Paris

     is

     also

     known

     for

     its

     love

     of

     music

    ,

     with

     a

     thriving

     music

     industry

     and

     numerous

     music

     festivals

    .

     Its

     reputation

     for

     excellent

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     rapidly

     evolving

     field

    ,

     with

     many

     potential

     trends

     shaping

     its

     development

     and

     impact

    .

     Here

     are

     some

     possible

     trends

     that

     could

     be

     expected

     in

     the

     field

    :
    


    1

    .

     Increased

     demand

     for

     AI

    -powered

     automation

    :

     With

     the

     growth

     of

     automation

     in

     industries

     such

     as

     manufacturing

    ,

     healthcare

    ,

     and

     transportation

    ,

     it

     is

     likely

     that

     AI

     will

     continue

     to

     be

     a

     key

     driver

     of

     automation

    ,

     driving

     cost

     savings

     and

     increasing

     efficiency

    .
    


    2

    .

     Development

     of

     more

     sophisticated

     AI

     models

    :

     AI

     models

     are

     becoming

     more

     complex

     and

     capable

     of

     learning

     from

     large

     datasets

    ,

     which

     could

     lead

     to

     better

     performance

     on

     a

     variety

     of

     tasks

    .

     Future

     AI

     systems

     will

     likely

     continue

     to

     be

     developed

     with

     this

     in

     mind

    .
    


    3

    .

     Integration

     of

    



```python
llm.shutdown()
```
