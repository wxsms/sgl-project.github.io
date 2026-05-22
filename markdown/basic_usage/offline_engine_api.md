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

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.70it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.48it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 22.68it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 22.68it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 22.68it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 22.68it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 22.68it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 22.68it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 22.68it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 22.68it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 22.68it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 22.68it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:04<00:00, 22.68it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.67it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.67it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.67it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.67it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.78it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.78it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.78it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.78it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.78it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.78it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 42.82it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  50%|█████     | 29/58 [00:00<00:00, 42.82it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 42.82it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 42.82it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 42.82it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 42.82it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.83it/s]

    Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.02it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.53it/s]

    Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.21it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 39.65it/s]


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
    Generated text:  Juan and I am from Mexico.
    I have learned to play violin as a child and I have been playing for many years. I play for my family and my music fans.
    My music career began when I started playing in a band called I Am The Soul. I began to hear the music of many artists and became interested in the music of other cultures.
    I started to learn music and I started to write music and in 2014 I wrote a song called "Ava para los niños" which became popular in the music scene.
    I have been involved in different projects throughout my life.
    Now, I am working on a new
    ===============================
    Prompt: The president of the United States is
    Generated text:  a great man and he is very popular. The president is the leader of the United States. His job is very important. He usually works very hard. The president’s job is to help the United States. He helps the United States by thinking about what the President says and by making decisions. The president also helps the people of the United States. People of the United States always want their president to work very hard. The president is not happy when he does not work very hard. The president is very tired when he works very hard. Sometimes the president works very hard when he is not sick. The president helps the United States by being
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It's the largest city in the world in terms of population. Paris is located in the southwest of France. The city has a long history. It was founded by the Romans in 500 BC, and was later ruled by the French kings. It is famous for its palaces, monuments, and theatres. 
    
    One of the most famous attractions in Paris is the Arc de Triomphe, which is a symbol of French history. 
    
    The Arc de Triomphe is a square with a rectangular shape. The square has a length of 500 meters on one side and a width of 30
    ===============================
    Prompt: The future of AI is
    Generated text:  not merely about having a specific technology, but about integrating it into existing systems and providing new possibilities for improved productivity, efficiency, and accuracy. The field of AI is constantly evolving, with new tools and techniques being developed all the time. As technology continues to advance, it will become increasingly important to have a comprehensive understanding of AI and its implications on various industries and sectors. Additionally, the integration of AI into existing systems will not only enhance productivity and efficiency, but also enable the development of new products and services that can benefit people and businesses in new and innovative ways.
    
    To achieve this, it will be essential for researchers, developers, and industry


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character or personality here]. What do you like to do? I enjoy [insert a short description of your hobbies or interests here]. What do you like to do in your free time? I like to [insert a short description of your hobbies or interests here]. What do you like to do with your family? I like to [insert a short description of your hobbies or interests here]. What do you like to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose". It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, art, and culture, and is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major economic and political center of France, with a large number of international institutions and organizations. Paris is a popular tourist destination and a major cultural hub, attracting millions of visitors each year. The city is also home to many other important cultural and artistic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI systems, there is likely to be a greater emphasis on ethical considerations and the development of more transparent and accountable AI systems.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, and transportation. As these technologies continue to evolve, it is likely that AI will be integrated with other technologies to create even more
    


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
    Generated text:  [Name]. I am a [Job Title], [Name] at [Company Name]. I am dedicated to [Your Goal/Job], and I am constantly learning and improving. Whether it's [Skill 1], [Skill 2], or [Skill 3], I strive to achieve excellence. I am an [Personality Trait], [Name] and I am always there to support my team members, providing guidance, advice, and encouragement. I am a [Personality Trait], [Name] and I am always there to support my team members, providing guidance, advice, and encouragement. I am a [Personality Trait
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known as the "City of Love" due to its romantic and charming atmosphere. 
    
    The statement is: "Paris, the "City of Love," is a major French capital located in the heart of the French Riviera, surrounded by the Alps and the Mediterranean Sea." 
    
    This statement encapsulates the main features of Paris, including its romantic nature and its geographical location. It also highlights its proximity to the French Riviera, which is a major tourist attraction. Finally, it specifies its proximity to the Mediterranean Sea, which is an important factor in its appeal as a tourist destination. 
    
    This statement provides a brief but accurate
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be a bright one, with many exciting developments in this field. Here are some possible trends that are expected to shape the future of AI:
    
    1. Improved AI ethics: As AI becomes more prevalent in our daily lives, it is important that we establish ethical guidelines and regulations to ensure that AI is used responsibly. This could involve developing more advanced ethical frameworks, promoting transparency and accountability in AI projects, and addressing potential biases in AI algorithms.
    
    2. More diverse AI research: AI has the potential to revolutionize various industries, from healthcare to finance to transportation. To realize these benefits, we will need to develop AI that is more diverse


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

    Your

     Name

    ].

     I

     am

     a

     

    2

    9

    -year

    -old

     software

     engineer

     who

     has

     been

     working

     in

     the

     tech

     industry

     for

     over

     

    1

    0

     years

     now

    .

     I

     am

     passionate

     about

     programming

     and

     have

     a

     knack

     for

     solving

     complex

     problems

     through

     algorithm

    ic

     thinking

    .

     I

     am

     currently

     pursuing

     a

     degree

     in

     Computer

     Science

     and

     I

     am

     currently

     working

     on

     a

     project

     that

     involves

     developing

     a

     new

     language

     for

     artificial

     intelligence

    .

     I

     love

     to

     travel

     and

     spend

     time

     with

     my

     family

     and

     friends

    .

     How

     can

     I

     get

     in

     touch

     with

     you

    ?


    Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ].

     I

     am

     a

     

    2

    9

    -year

    -old

     software

     engineer

     who

     has

     been

     working

     in

     the

     tech

     industry

     for

     over

     

    1

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

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
    


    Paris

    ,

     the

     capital

     of

     France

    ,

     is

     renowned

     for

     its

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

     rich

     history

    ,

     diverse

     culture

    ,

     and

     vibrant

     nightlife

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     considered

     one

     of

     the

     most

     beautiful

     cities

     in

     the

     world

    .

     With

     its

     beautiful

     architecture

    ,

     beautiful

     architecture

    ,

     and

     rich

     history

    ,

     Paris

     continues

     to

     be

     an

     important

     center

     of

     culture

     and

     civilization

     in

     Europe

     and

     beyond

    .

     
    


    That

    's

     very

     interesting

    !

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     rapidly

     evolving

    ,

     with

     many

     possibilities

     and

     potential

     developments

     to

     consider

    .

     Some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     in

     healthcare

    :

     AI

    -powered

     systems

     will

     be

     increasingly

     used

     in

     healthcare

     to

     diagnose

     and

     treat

     diseases

    ,

     automate

     patient

     care

    ,

     and

     improve

     treatment

     outcomes

    .

     This

     will

     require

     a

     more

     in

    -depth

     understanding

     of

     human

     anatomy

     and

     physiology

    ,

     as

     well

     as

     the

     ability

     to

     handle

     complex

     data

     and

     complex

     patient

     interactions

    .
    


    2

    .

     More

     advanced

     self

    -driving

     cars

    :

     Self

    -driving

     cars

     are

     expected

     to

     become

     more

     advanced

     and

     autonomous

    ,

     with

     the

     ability

     to

     recognize

     and

     respond

     to

     a

     wider

     range

     of

     road

     conditions

     and

     accidents

    .

     This

     will

     require

     significant

     advancements

    



```python
llm.shutdown()
```
