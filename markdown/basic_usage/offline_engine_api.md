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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.36it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:43,  3.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:43,  3.91s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:43,  3.91s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:43,  3.91s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:43,  3.91s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.91it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.91it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.39it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.39it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.39it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.39it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.39it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.39it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.39it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.39it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.39it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.39it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.39it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.92it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.00 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.00 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.99 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.99 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.99 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=52.99 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.99 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.98 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.97 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.97 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.96 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.96 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.96 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.46it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=52.96 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.96 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.95 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.95 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.95 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.26it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.94 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.26it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.94 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.26it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.94 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.94 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.92 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s]Capturing num tokens (num_tokens=960 avail_mem=52.93 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s] Capturing num tokens (num_tokens=896 avail_mem=52.93 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s]

    Capturing num tokens (num_tokens=832 avail_mem=52.93 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s]Capturing num tokens (num_tokens=832 avail_mem=52.93 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.54it/s]Capturing num tokens (num_tokens=768 avail_mem=52.92 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.54it/s]Capturing num tokens (num_tokens=704 avail_mem=52.92 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.54it/s]Capturing num tokens (num_tokens=640 avail_mem=52.91 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.54it/s]Capturing num tokens (num_tokens=576 avail_mem=52.91 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.54it/s]Capturing num tokens (num_tokens=512 avail_mem=52.90 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.54it/s]Capturing num tokens (num_tokens=512 avail_mem=52.90 GB):  50%|█████     | 29/58 [00:00<00:00, 43.85it/s]Capturing num tokens (num_tokens=480 avail_mem=52.91 GB):  50%|█████     | 29/58 [00:00<00:00, 43.85it/s]Capturing num tokens (num_tokens=448 avail_mem=52.91 GB):  50%|█████     | 29/58 [00:00<00:00, 43.85it/s]Capturing num tokens (num_tokens=416 avail_mem=52.91 GB):  50%|█████     | 29/58 [00:00<00:00, 43.85it/s]Capturing num tokens (num_tokens=384 avail_mem=52.91 GB):  50%|█████     | 29/58 [00:00<00:00, 43.85it/s]

    Capturing num tokens (num_tokens=352 avail_mem=52.90 GB):  50%|█████     | 29/58 [00:00<00:00, 43.85it/s]Capturing num tokens (num_tokens=352 avail_mem=52.90 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=320 avail_mem=52.90 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=288 avail_mem=52.90 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=256 avail_mem=52.89 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=240 avail_mem=52.89 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=224 avail_mem=52.89 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=224 avail_mem=52.89 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.48it/s]Capturing num tokens (num_tokens=208 avail_mem=52.88 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.48it/s]Capturing num tokens (num_tokens=192 avail_mem=52.88 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.48it/s]Capturing num tokens (num_tokens=176 avail_mem=52.88 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.48it/s]Capturing num tokens (num_tokens=160 avail_mem=52.88 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.48it/s]

    Capturing num tokens (num_tokens=144 avail_mem=52.87 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.48it/s]Capturing num tokens (num_tokens=144 avail_mem=52.87 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=128 avail_mem=52.87 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=112 avail_mem=52.87 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=96 avail_mem=52.86 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s] Capturing num tokens (num_tokens=80 avail_mem=52.86 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=64 avail_mem=52.86 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=64 avail_mem=52.86 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=48 avail_mem=52.85 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=32 avail_mem=52.85 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=28 avail_mem=52.84 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=24 avail_mem=52.84 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.09it/s]

    Capturing num tokens (num_tokens=20 avail_mem=52.84 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=20 avail_mem=52.84 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=16 avail_mem=52.84 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=12 avail_mem=52.83 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=8 avail_mem=52.83 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.64it/s] Capturing num tokens (num_tokens=4 avail_mem=52.83 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=4 avail_mem=52.83 GB): 100%|██████████| 58/58 [00:01<00:00, 42.23it/s]


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
    Generated text:  Ravi. I am a software developer from India and I am now based in the United States. I have 10 years of experience in the field of software development, and I have worked on many projects from small to medium size. I have a passion for innovation, and I enjoy using my skills and knowledge to improve the world around me.
    I started my career in the field of software development, and I have gained a lot of experience in the field. I have developed software applications that are designed to meet the needs of a variety of industries, and I have a deep understanding of the technology and the tools used to develop them.
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  66 years old now. In how many years will he be 3 times as old as before?
    
    To determine in how many years the president of the United States will be 3 times as old as he was currently, we can set up an equation based on the information given.
    
    Let's denote the number of years from now as \( x \).
    
    Currently, the president's age is 66 years. In \( x \) years, his age will be:
    \[ 66 + x \]
    
    According to the problem, at that point, he will be 3 times as old as he was currently. Therefore,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is situated on the Left Bank, 50 kilometers west of the center of Paris. It is known for its iconic landmarks such as the Eiffel Tower and the Louvre Museum. The city is a significant hub of commerce, culture, and politics. Paris is a popular tourist destination, with millions of visitors each year. It is also known for its cuisine, art, and history. It is the capital of France and the largest city in the country. 
    
    What are the most significant features of Paris? 
    
    1. The Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral
    2
    ===============================
    Prompt: The future of AI is
    Generated text:  bright but it is not bright enough
    
    I just came across a statement in a recent post on Medium regarding the future of AI and machine learning (among other topics). It’s an interesting post in its own right, but I think it’s important to highlight the fact that while AI is potentially bright, it is not bright enough. This is one of the main problems with machine learning.
    
    What makes machine learning bright is that it enables us to build and train large models that can learn from data. This is the core of the machine learning concept: the ability to learn from data without being explicitly programmed. Even when the data is very small,


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


    Generated text:  [Name] and I am a [occupation] who has been [number of years] in the industry. I am passionate about [reason for interest in the industry]. I am always looking for new opportunities and learning new things. I am a [number of years] in the industry and have [number of years] of experience in [occupation]. I am a [number of years] in the industry and have [number of years] of experience in [occupation]. I am a [number of years] in the industry and have [number of years] of experience in [occupation]. I am a [number of years] in the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville-Marie" and "La Ville-Est". It is the largest city in France and the second-largest city in the European Union. The city is located on the Seine River and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is known for its rich history, art, and culture, and is a popular tourist destination for visitors from around the world. The city is also home to many important institutions, including the French Academy of Sciences and the French National Library. Paris is a vibrant and dynamic city with a rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the coming years.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology continues to improve, we can expect to see even
    


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
    Generated text:  Sarah. I'm a successful entrepreneur with a passion for adventure and sustainability. I'm passionate about creating eco-friendly products and marketing them to the world. I'm a natural problem-solver and always up for trying new ideas to make a positive impact. I'm a supportive team player who thrives in fast-paced environments and enjoys collaborating with others. I'm eager to learn and grow, and I'm constantly looking for new challenges and opportunities to grow. I'm excited to start a new chapter in my life and help others on their journey towards sustainability. How would you describe Sarah's personality traits? Sarah's personality traits include a natural sense of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and one of the most populous in the world. Paris has a rich history dating back to ancient times and is known for its beautiful architecture, vibrant culture, and iconic landmarks such as the Eiffel Tower. It is also a major center for politics, economics, and culture. The city is home to many notable museums, museums, and art galleries, as well as many historic neighborhoods, including the Latin Quarter and the Marais. Paris is a major tourist destination and a popular place for leisure and entertainment. It is also an important hub for finance, politics, and international diplomacy. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several trends that could significantly impact various sectors of society. Here are some of the most likely future trends in AI:
    
    1. Increased automation and efficiency: AI has the potential to automate many mundane tasks, freeing up workers to focus on more complex and creative work. This could lead to increased efficiency and productivity across various industries, including transportation, manufacturing, and healthcare.
    
    2. Increased human-AI collaboration: The increasing use of AI in healthcare and other areas of medicine is likely to lead to more AI-powered tools and services, such as telemedicine and personalized medicine. This could lead to a more collaborative relationship between humans and


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

    'm

     [

    Age

    ].

     I

    'm

     a

     [

    field

    ]

     professional

     with

     [

    number

     of

     years

    ]

     years

     of

     experience

    ,

     and

     I

    'm

     [

    objective

     summary

     of

     your

     professional

     experience

    ].

     I

    'm

     passionate

     about

     [

    something

     you

     do

     or

     teach

    ],

     and

     I

     believe

     in

     [

    reason

     for

     belief

    ].

     I

     believe

     that

     [

    reason

     why

     I

     believe

    ].

     I

    'm

     excited

     to

     share

     my

     knowledge

     and

     experience

     with

     you

    .

     Let

    's

     get

     started

    !

     (

    pause

    )

     And

    ,

     my

     name

     is

     [

    Name

    ]

     again

    .

     I

    'm

     [

    Name

    ]

     and

     I

    'm

     a

     [

    Name

    ]

     professional

     with

     [

    number

     of

     years

    ]

     years

     of

     experience

    ,

     and

     I

    'm

     [

    objective

     summary

     of

     your

     professional

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     and

     most

     populous

     city

     in

     the

     country

    .


    Paris

     is

     the

     capital

     of

     France

     and

     serves

     as

     the

     political

    ,

     economic

    ,

     cultural

    ,

     and

     administrative

     center

     of

     the

     country

    .

     It

     is

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

     The

     city

     is

     also

     known

     for

     its

     vibrant

     music

     scene

    ,

     fashion

     industry

    ,

     and

     food

     culture

    ,

     and

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     restaurants

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

    City

     of

     Love

    "

     due

     to

     its

     romantic

     and

     artistic

     atmosphere

    .

     As

     the

     largest

     city

     in

     the

     world

    ,

     Paris

     is

     an

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     constantly

     evolving

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     industry

     in

     the

     coming

     years

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

     With

     the

     increasing

     availability

     of

     big

     data

     and

     advanced

     algorithms

    ,

     AI

     is

     increasingly

     being

     used

     in

     healthcare

     to

     diagnose

     and

     treat

     diseases

    .

     Machine

     learning

     and

     natural

     language

     processing

     could

     be

     used

     to

     create

     more

     accurate

     and

     personalized

     treatments

    ,

     as

     well

     as

     to

     automate

     routine

     medical

     tasks

     like

     appointment

     scheduling

     and

     medication

     management

    .
    


    2

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     is

     increasingly

     integrated

     into

     daily

     life

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     ethical

     considerations

    .

     This

     could

     include

     issues

     like

     bias

    ,

     transparency

    ,

     and

     accountability

    ,

     and

     could

     lead

    



```python
llm.shutdown()
```
