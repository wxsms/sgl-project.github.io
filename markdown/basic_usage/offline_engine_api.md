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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.37it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.56it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.38it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.70it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.70it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.70it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.70it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.70it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.70it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.70it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.70it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.70it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.70it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 31.86it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 31.86it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 31.86it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 31.86it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 31.86it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 31.86it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 31.86it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 31.86it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 31.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.97 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.94 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.93 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.92 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.92 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.92 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.92 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.91 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.91 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.91 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.90 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.90 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.89 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.89 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.89 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.88 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.06it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.86 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.06it/s]Capturing num tokens (num_tokens=960 avail_mem=72.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.06it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.87 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.06it/s]Capturing num tokens (num_tokens=896 avail_mem=72.87 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=832 avail_mem=72.87 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=768 avail_mem=72.87 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=704 avail_mem=72.86 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=640 avail_mem=72.86 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=576 avail_mem=72.86 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.16it/s]Capturing num tokens (num_tokens=576 avail_mem=72.86 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.45it/s]Capturing num tokens (num_tokens=512 avail_mem=72.84 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.45it/s]Capturing num tokens (num_tokens=480 avail_mem=72.86 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.45it/s]Capturing num tokens (num_tokens=448 avail_mem=72.86 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.45it/s]Capturing num tokens (num_tokens=416 avail_mem=72.86 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.45it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.85 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.45it/s]Capturing num tokens (num_tokens=384 avail_mem=72.85 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.99it/s]Capturing num tokens (num_tokens=352 avail_mem=72.85 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.99it/s]Capturing num tokens (num_tokens=320 avail_mem=72.84 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.99it/s]Capturing num tokens (num_tokens=288 avail_mem=72.84 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.99it/s]Capturing num tokens (num_tokens=256 avail_mem=72.84 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.99it/s]Capturing num tokens (num_tokens=240 avail_mem=72.83 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.99it/s]Capturing num tokens (num_tokens=240 avail_mem=72.83 GB):  66%|██████▌   | 38/58 [00:00<00:00, 45.60it/s]Capturing num tokens (num_tokens=224 avail_mem=72.83 GB):  66%|██████▌   | 38/58 [00:00<00:00, 45.60it/s]Capturing num tokens (num_tokens=208 avail_mem=72.83 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=192 avail_mem=72.83 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=176 avail_mem=72.82 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.60it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.82 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=160 avail_mem=72.82 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.74it/s]Capturing num tokens (num_tokens=144 avail_mem=72.82 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.74it/s]Capturing num tokens (num_tokens=128 avail_mem=72.82 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.74it/s]Capturing num tokens (num_tokens=112 avail_mem=72.81 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.74it/s]Capturing num tokens (num_tokens=96 avail_mem=72.81 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.74it/s] Capturing num tokens (num_tokens=80 avail_mem=72.81 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.74it/s]Capturing num tokens (num_tokens=80 avail_mem=72.81 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=64 avail_mem=72.80 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=48 avail_mem=72.80 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=32 avail_mem=72.80 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=28 avail_mem=72.79 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.19it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.79 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=24 avail_mem=72.79 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=20 avail_mem=72.78 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=16 avail_mem=72.78 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=12 avail_mem=72.78 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=8 avail_mem=72.78 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.46it/s] Capturing num tokens (num_tokens=4 avail_mem=72.77 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=4 avail_mem=72.77 GB): 100%|██████████| 58/58 [00:01<00:00, 48.07it/s]Capturing num tokens (num_tokens=4 avail_mem=72.77 GB): 100%|██████████| 58/58 [00:01<00:00, 41.61it/s]


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
    Generated text:  Emile and I am a full-time student. I have a degree in mathematics and have a passion for literature. My hobbies include reading, listening to music, and traveling. I enjoy exploring new places and trying new foods. I also love playing video games and baking. What are some activities that you enjoy doing outside of school?
    As an AI language model, I don't have personal hobbies or interests like humans do. However, I can provide you with some activities that some people enjoy doing outside of school.
    1. Visiting museums and galleries: Museums and galleries are great places to explore new art and historical artifacts.
    2. Vis
    ===============================
    Prompt: The president of the United States is
    Generated text:  the commander-in-chief of the armed forces of the United States, with the authority to issue an executive order. True or false:
    A. True
    B. False
    Answer: A
    
    The term of office for the county-level and higher people's congresses and their standing committees shall be determined by law. A. Correct B. Incorrect
    Answer: A
    
    According to the "Regulations on the Safety Production License for Construction Enterprises", if a construction enterprise obtains the construction safety production license after suspension, a fine of _____ shall be imposed.
    A. 50,000 to 100,000 yuan
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The most important place to visit in Paris is the Louvre. The Louvre is a beautiful museum which houses some of the greatest paintings in the world. The museum has about 2, 000, 000 objects on display. The paintings are made of oil, water and acrylics, and the works are about 100 years old. Visitors can look at them with their eyes, their noses, and their ears. The Louvre Museum is known for its vast collection of art. Many of the paintings in the museum date back to the 13th century. It is also known as
    ===============================
    Prompt: The future of AI is
    Generated text:  not what you expect
    
    The future of AI is not what you expect
    
    AI is coming. It will change the world. Will it change your life too?
    
    What's your dream job in 5 years?
    How do you think technology will change the world?
    
    The answer is: You are not sure yet. In the age of artificial intelligence, technology is not a fixed thing. It is evolving, changing, and evolving again. In 5 years, you may not even know what "AI" means. It will change your life and the world around you. That is the reality of the future, and it is not what you expect


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your interests and what you're looking for in a job. Let's chat about [job title] and see where our careers can take us together. [Name] is looking for a [job title] position at [Company Name], and I'm excited to help them find the perfect fit. Let's get started! [Name] is looking for a [job title] position at [Company Name], and I'm excited to help them find the perfect fit. Let's get started! [Name] is
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of French culture and literature, and its role as a hub for international trade and diplomacy. The city is also famous for its fashion industry, with Paris Fashion Week being one of the world's largest and most prestigious. Overall, Paris is a vibrant and dynamic city with a rich history and a strong cultural identity.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This means that AI systems will be able to learn from and adapt to the behavior and preferences of humans, which could lead to more effective and personalized AI systems.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations. This means that AI systems will be designed with the goal of being transparent
    


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
    Generated text:  [Name] and I'm a [Role]. I'm [Age] years old and have [Height] inches. I'm [Gender] and have [Height] inches and weigh [Weight] pounds. I have [Hair Color] and [Eye Color]. I'm [Gender] and have [Hair Color] and [Eye Color]. I enjoy [Favorite Activity]. I have [Height] inches and weigh [Weight] pounds and have [Hair Color] and [Eye Color]. I'm [Gender] and have [Hair Color] and [Eye Color]. I like to [Favorite Activity]. I have [Height] inches and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    City Name: Paris
    Country: France
    Population: Over 3 million
    Climate: Mediterranean climate with hot summers and mild winters
    Economic Importance: The city is the second-largest economic and cultural center in the world and home to many world-renowned institutions and landmarks. It is also a major center for international business, media, and finance. 
    
    Languages: French, English, Catalan, Occitan
    Religion: Approximately 80% Catholic, 10% Protestant, 5% Muslim, and 5% other
    Major Attractions: The Eiffel Tower, Notre-Dame Cathedral, Lou
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by continued innovation, rapid development, and widespread adoption. Here are some possible future trends in AI:
    
    1. Increased sensitivity and ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ensuring that it is designed and used ethically. This may mean that AI developers will become more mindful of the potential impact of their creations and take steps to minimize unintended consequences.
    
    2. Integration with human intelligence: AI is likely to continue to evolve beyond its current role as a tool for automating tasks and improving efficiency. It is possible that AI will be integrated with human intelligence, enabling more sophisticated


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

     AI

     language

     model

     designed

     to

     assist

     people

     in

     various

     tasks

     like

     language

     translation

    ,

     information

     retrieval

    ,

     and

     more

    .

     I

    'm

     constantly

     learning

     and

     improving

     my

     abilities

     to

     provide

     helpful

     responses

    .

     Would

     you

     like

     to

     talk

     to

     me

     about

     something

     specific

    ?

     How

     can

     I

     assist

     you

     today

    ?

     Let

     me

     know

    !

     [

    Name

    ]

     wants

     to

     make

     contact

     to

     help

     out

    .

     (

    Make

     sure

     to

     keep

     the

     self

    -int

    roduction

     neutral

     and

     direct

    ,

     without

     being

     overly

     aggressive

     or

     promotional

    .)

     Good

     day

    ,

     [

    Name

    ]

    !

     It

    's

     great

     to

     meet

     you

    .

     My

     name

     is

     [

    Name

    ]

     and

     I

    'm

     an

     AI

     language

     model

     designed

     to

     assist

     people

     in

     various

     tasks

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    (M

    ultiple

     Choice

     Question

    )
    


    A

    )

     Correct

    


    B

    )

     Incorrect

    


    C

    )

     Partial

    ly

     Correct

    


    D

    )

     Not

     App

    licable

    
    


    A

    )

     Correct

    
    


    Paris

     is

     the

     capital

     city

     of

     France

    ,

     the

     largest

     country

     in

     Europe

     and

     the

     fourth

     largest

     in

     the

     world

     by

     area

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     artistic

     and

     cultural

     institutions

    ,

     and

     annual

     festivals

     like

     the

     Les

     Arm

    ées

    .

     Its

     status

     as

     the

     world

    's

     most

     populous

     city

     is

     further

     emphasized

     by

     its

     unique

     architecture

    ,

     including

     the

     iconic

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

     Paris

     also

     hosts

     major

     world

     events

     and

     is

     the

     seat

     of

     the

     French

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     with

     new

     and

     exciting

     developments

     on

     the

     horizon

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     AI

     over

     the

     next

     few

     years

    :
    


    1

    .

     Increased

     integration

     with

     human

     emotions

    :

     AI

     is

     already

     able

     to

     process

     and

     interpret

     a

     wide

     range

     of

     emotions

    ,

     but

     there

    's

     still

     much

     more

     to

     learn

     about

     how

     to

     build

     AI

     that

     can

     understand

     and

     respond

     to

     human

     emotions

    .

     Researchers

     are

     working

     to

     develop

     more

     sophisticated

     models

     that

     can

     recognize

     and

     understand

     a

     wide

     range

     of

     facial

     expressions

    ,

     body

     language

    ,

     and

     other

     emotions

    .
    


    2

    .

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     analyze

     medical

     images

    ,

     diagnose

     diseases

    ,

     and

     develop

     new

     treatments

    .

     However

    ,

     there

    's

     still

     much

     more

    



```python
llm.shutdown()
```
