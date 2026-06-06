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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.90it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.90it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.90it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.90it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.90it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.90it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.90it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.90it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.90it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.06it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.06it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.06it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.06it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.06it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.06it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.06it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.06it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.06it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.06it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 21.06it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.43it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.48it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.48it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.48it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.76it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  50%|█████     | 29/58 [00:00<00:00, 41.92it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 41.92it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 41.92it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 41.92it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 41.92it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 41.92it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.25it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.25it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.25it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.25it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.25it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.25it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=208 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.21it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=128 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=96 avail_mem=74.36 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.02it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.68it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.68it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.68it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.68it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.68it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.68it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.46it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 39.83it/s]


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
    Generated text:  the beautiful, strong, intelligent, and talented Koenigswaldian philosopher, Koenig. I’ve been observing myself for the past five years as part of an experimental project to create a new, highly advanced AI agent. The AI, named Phebe, is my companion and the main character of the project.
    
    Phebe is a tool designed to help users navigate the world and make decisions. She’s been a significant part of my life since I learned to speak, and I rely on her for practical advice and interaction with the world.
    
    Phebe is also a part of an enormous, complex system known as the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking government officer who is also the leader of the country. The person who is in charge of the United States government is called a president. The president of the United States is the leader of the executive branch of the government and is accountable to Congress, the legislature. The president is the head of the executive branch of the government.
    What role does the president of the United States play in the executive branch of the government? The president of the United States plays the role of leader of the executive branch of the government. He or she is responsible for making decisions and policies that affect the country. The president is accountable to Congress, the
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. London
    C. New York
    D. Sydney
    Answer:
    
    A
    
    If the plane intersects two straight lines and forms two angles of 60°, which of the following statements is correct? 
    A. The two straight lines must be parallel.
    B. The two straight lines must intersect.
    C. The two straight lines must be perpendicular.
    D. The two straight lines cannot be determined.
    Answer:
    
    B
    
    2. There are 3 girls and 2 boys participating in a singing contest. If the organizers randomly select 2 people to perform, what is the probability that both of
    ===============================
    Prompt: The future of AI is
    Generated text:  very exciting and it is being applied in many different areas including medical and finance. We may need to rely more on AI in the near future. From a business standpoint, we can benefit from AI because it can automate our work, it can save us time and money, and it can increase our productivity. Some of the most popular applications of AI include data analysis, machine learning, computer vision, robotics, natural language processing, and cybersecurity. AI will be a key driver of economic growth and job creation in the future. In the business world, AI has the potential to revolutionize the way we conduct business. It can reduce costs, increase


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. Let's chat! [Name] [Company Name] [Company Address] [Company Phone Number] [Company Email] [Company Website] [Company LinkedIn Profile] [Company Twitter Profile] [Company Facebook Profile] [Company LinkedIn Profile] [Company Twitter Profile] [Company Facebook Profile] [Company LinkedIn Profile] [Company Twitter Profile] [Company Facebook Profile] [Company LinkedIn Profile] [Company Twitter Profile] [Company Facebook Profile] [Company LinkedIn Profile] [Company Twitter Profile
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of French colonialism and the influence of various European cultures. It is a city of contrasts, with its modern architecture and historical landmarks blending together in a unique and dynamic urban environment. Paris is a city of art, culture, and innovation, and continues to be a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is already being used to automate a wide range of tasks, from manufacturing to customer service. As technology continues to advance, we can expect to see even more automation in areas such as healthcare, finance, and transportation.
    
    2. Enhanced human-computer interaction: AI is already being used to enhance human-computer interaction, such as through voice recognition and natural language processing. As technology continues to advance, we can expect to see even more advanced AI that can interact with humans in ways that are more natural and intuitive.
    
    3. AI ethics and privacy: As AI becomes more
    


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
    Generated text:  [Name], and I'm a [insert your job role here] who has a passion for [insert something you enjoy or love about your job]. I'm always looking for new challenges and learning new things, and I'm eager to contribute to any project that requires my skills and experience. If you're looking for someone with a positive attitude and a willingness to learn, I'd love to be part of your team. Thanks for taking the time to meet me! 🌟✨
    
    This is a neutral self-introduction for a fictional character. How can I further personalize the introduction to make it more interesting and engaging for my target audience
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which has been a major city for over 2000 years and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to numerous museums, theaters, and other cultural institutions that attract visitors from around the world. In addition, Paris is a popular tourist destination for visitors from around the world, including France itself. The city is known for its unique culture, rich history, and vibrant nightlife. Paris has a strong sense of community and is a city that is dedicated to both its past and its future. The city is also known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undoubtedly going to be exciting, as new technologies and developments are constantly emerging and transforming the way we live and work. Here are some of the possible future trends in AI that are expected to shape the future of the field:
    
    1. AI in healthcare: AI will play a critical role in healthcare by improving diagnosis, treatment, and patient care. It will also help in developing more personalized treatment plans for each patient, and will enable the development of new therapies for diseases that are currently difficult to treat.
    
    2. Autonomous vehicles: Autonomous vehicles will become more prevalent in the future as they become more reliable and cost-effective. This will lead to a decrease


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

     am

     a

     [

    job

     title

    ]

     at

     [

    company

     name

    ],

     and

     I

     am

     passionate

     about

     [

    mot

    iv

    ational

     goal

     or

     hobby

    ].

     I

     am

     always

     looking

     for

     new

     opportunities

     to

     [

    new

     experience

     or

     skill

    ].

     Let

     me

     know

     how

     I

     can

     assist

     you

     today

    .

     [

    Name

    ]

     =

     [

    Type

     your

     response

     here

    ]

     Hello

    ,

     my

     name

     is

     [

    Name

    ].

     I

     am

     a

     [

    job

     title

    ]

     at

     [

    company

     name

    ],

     and

     I

     am

     passionate

     about

     [

    mot

    iv

    ational

     goal

     or

     hobby

    ].

     I

     am

     always

     looking

     for

     new

     opportunities

     to

     [

    new

     experience

     or

     skill

    ].

     Let

     me

     know

     how

     I

     can

     assist

     you

     today

    .

     [

    Name

    ]

     =

     [

    Type

     your

     response

     here

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     “

    City

     of

     Love

    ”

     and

     is

     located

     on

     the

     Se

    ine

     river

    ,

     near

     the

     C

    ote

     d

    '

    Az

    ur

    .

     It

     is

     a

     historical

     and

     cultural

     center

     with

     many

     world

    -ren

    owned

     landmarks

    ,

     including

     the

     Lou

    vre

     museum

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     known

     for

     its

     vibrant

     street

     life

    ,

     charming

     neighborhoods

    ,

     and

     annual

     festivals

     such

     as

     the

     E

    iff

    el

     Tower

     Parade

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     The

     city

     is

     also

     home

     to

     many

     notable

     museums

     and

     attractions

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

     is

     the

     political

    ,

     economic

    ,

     and

     cultural

     center

     of

     France

    ,

     and

     is

     a

     UNESCO

     world

     heritage

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     significant

     advancements

     in

     various

     areas

     such

     as

     computer

     hardware

    ,

     software

    ,

     and

     the

     integration

     of

     artificial

     intelligence

     into

     everyday

     life

    .

     Here

     are

     some

     possible

     trends

     in

     the

     future

     of

     AI

    :
    


    1

    .

     Deep

     learning

    :

     Deep

     learning

     is

     one

     of

     the

     most

     promising

     areas

     for

     AI

     advancement

    ,

     with

     the

     ability

     to

     learn

     from

     vast

     amounts

     of

     data

     and

     perform

     complex

     tasks

     such

     as

     image

     and

     speech

     recognition

    .

     Deep

     learning

     is

     expected

     to

     be

     widely

     adopted

     in

     various

     industries

    ,

     including

     healthcare

    ,

     finance

    ,

     and

     transportation

    .
    


    2

    .

     Natural

     language

     processing

     (

    N

    LP

    ):

     N

    LP

     is

     the

     ability

     of

     AI

     systems

     to

     understand

     and

     interpret

     human

     language

    .

     This

     technology

     is

     expected

     to

     further

     improve

    



```python
llm.shutdown()
```
