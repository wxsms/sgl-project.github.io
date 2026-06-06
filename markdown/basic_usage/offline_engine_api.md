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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.60s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.60s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.60s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:21,  4.60s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.96it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  8.96it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.96it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.58it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.58it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.58it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.58it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.58it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.58it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.58it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.58it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.58it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.58it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.58it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.24it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 17.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 17.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 17.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 17.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.16it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=70.98 GB):  31%|███       | 18/58 [00:00<00:01, 30.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  31%|███       | 18/58 [00:00<00:01, 30.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  31%|███       | 18/58 [00:00<00:01, 30.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  31%|███       | 18/58 [00:00<00:01, 30.61it/s]Capturing num tokens (num_tokens=960 avail_mem=70.97 GB):  31%|███       | 18/58 [00:00<00:01, 30.61it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  31%|███       | 18/58 [00:00<00:01, 30.61it/s]Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.81it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.81it/s]Capturing num tokens (num_tokens=768 avail_mem=70.96 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.81it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.81it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.81it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.81it/s]

    Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.44it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.44it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.44it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.44it/s]Capturing num tokens (num_tokens=416 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.44it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.44it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.29it/s]

    Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s] Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s]

    Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.19it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.19it/s]

    Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 45.27it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 38.81it/s]


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
    Generated text:  Alex and I recently started working at a local brewery. I love craft beer, especially the Belgian style, which is light, refreshing and can be great for a cold day. I recently decided to try a new type of Belgian-style beer and bought this from the brewery. While I was enjoying the beer, I noticed that it was going through a process of aging in a cask. The cask was not labeled on the outside and I didn't know if it was fresh or not. 
    
    My question is, what would be a good way to tell if a beer is fresh or not? What would be a good way to tell if
    ===============================
    Prompt: The president of the United States is
    Generated text:  34 years older than the president of Brazil, and the president of Brazil is 2 times older than the president of Russia. How old is the president of Russia?
    
    To determine the age of the president of Russia, we need to follow the information given step by step.
    
    1. Let's denote the age of the president of Brazil as \( B \). According to the problem, the president of Brazil is 2 times older than the president of Russia. Therefore, we can express the age of the president of Brazil as \( B = 2R \), where \( R \) is the age of the president of Russia.
    
    
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. London
    C. Moscow
    D. Tokyo
    Answer:
    
    A
    
    When using the S shape method to analyze a current transformer, the gap between the bushings of the two transformer windings should be ___.
    A. 5-10mm
    B. 10-15mm
    C. 20-25mm
    D. 25-30mm
    Answer:
    
    C
    
    The primary focus of adolescent psychology research is ____
    A. Adolescent psychology
    B. Development psychology
    C. Educational psychology
    D. Social psychology
    Answer:
    
    A
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable. As data volumes and complexity increase, so do the challenges of AI development, deployment, and management. AI applications in healthcare, transportation, and customer service are among the fastest-growing areas of AI development. Due to the high volume of data, AI models trained using traditional machine learning approaches can be slow and not suitable for complex problems. Traditional approaches can have very high overhead, especially in terms of computing power, storage, and deployment.
    Other types of machine learning approaches include reinforcement learning, and they are more suited for complex problems. It’s important to note that AI is not a technology that solves all problems. While AI can solve problems


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] [Vehicle Name]. I'm [Favorite Hobby] [Favorite Food] [Favorite Book] [Favorite Movie]. I'm [Favorite Sport] [Favorite Music] [Favorite Movie]. I'm [Favorite Place] [Favorite Color]. I'm [Favorite Animal]. I'm [Favorite Movie]. I'm [Favorite Book]. I'm [Favorite Sport]. I'm [Favorite Music]. I'm [Favorite Movie]. I'm [Favorite Place]. I'm [Favorite Color]. I'm [Favorite Animal]. I'm [Favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic center, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. It is a popular tourist destination, with millions of visitors annually. Paris is also known for its cuisine, with its famous dishes such as croissants, boudin, and escargot. The city is also home to many museums, including the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to the needs of their users.
    
    2. Enhanced machine learning capabilities: As AI technology continues to advance, machine learning algorithms are likely to become even more sophisticated and capable. This could lead to more complex and sophisticated AI systems that can perform a
    


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
    Generated text:  [Name], and I am a [insert profession or role] who has always been fascinated by [insert something interesting about your field of work]. I enjoy learning about new technologies and products, and I'm always looking for ways to improve myself and others through knowledge sharing. I'm a problem solver who can think outside the box and come up with creative solutions to complex issues. And, I love hiking and exploring new places to explore new adventures. How about you? What are your interests and passions? What's your career path and what kind of work are you currently involved in? I'd love to hear about your experiences and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid advancements in technology and applications that further transform the way we live, work, and interact with each other. Here are some possible future trends in AI that could shape the future of the field:
    
    1. Improved Machine Learning and Deep Learning: AI systems will continue to improve their ability to learn and make decisions autonomously, improving the efficiency of their operations and reducing errors.
    
    2. Enhanced Personalization: AI will be able to better understand individual user preferences and behavior, allowing for more personalized experiences and services.
    
    3. Integration with the Physical World: AI will continue to become more integrated with the physical world, allowing for


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

     am

     an

     experienced

     freelance

     writer

     and

     digital

     marketer

     with

     a

     passion

     for

     crafting

     compelling

     content

     and

     creating

     a

     unique

     brand

     identity

    .

     I

     have

     a

     degree

     in

     marketing

     and

     have

     worked

     in

     various

     industries

    ,

     including

     advertising

     and

     public

     relations

    ,

     and

     I

     have

     a

     knack

     for

     turning

     ideas

     into

     successful

     campaigns

    .

     I

     am

     a

     strong

     communicator

    ,

     able

     to

     write

     engaging

     and

     informative

     content

     that

     reson

    ates

     with

     my

     audience

     and

     drives

     positive

     results

    .

     I

     am

     always

     looking

     for

     new

     opportunities

     to

     contribute

     to

     the

     field

     of

     marketing

     and

     I

     am

     excited

     to

     bring

     my

     skills

     to

     your

     team

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

     I

    'm

     reaching

     out

     to

     you

     on

     [

    medium

     platform

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     is

     fact

    ually

     accurate

     and

     con

    veys

     the

     essential

     information

     about

     the

     city

    's

     capital

     in

     English

    .

     It

     does

     not

     require

     any

     additional

     context

     or

     elabor

    ation

    .

     
    


    In

     the

     context

     of

     France

    's

     political

     and

     cultural

     landscape

    ,

     Paris

     is

     the

     second

    -largest

     city

     in

     the

     country

    ,

     behind

     only

     Paris

    ,

     and

     is

     the

     largest

     city

     by

     area

     in

     the

     European

     Union

    .

     It

     is

     the

     capital

     of

     France

     and

     the

     third

     most

     populous

     city

     in

     the

     country

    ,

     after

     Paris

     and

     London

    .

     Paris

     is

     known

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

     iconic

     status

     and

     love

     affairs

    ,

     and

     is

     the

     home

     of

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     several

     key

     trends

    :
    


     

     

    1

    .

     Increased

     integration

     with

     human

     AI

    :

     As

     AI

     systems

     continue

     to

     become

     more

     integrated

     with

     human

     AI

    ,

     we

     may

     see

     greater

     efficiency

     and

     productivity

    ,

     but

     also

     greater

     biases

     and

     vulnerabilities

    .


     

     

    2

    .

     Development

     of

     AI

     ethics

    :

     As

     more

     AI

     systems

     become

     integrated

     with

     human

     decision

    -making

    ,

     there

     will

     be

     increased

     scrutiny

     of

     the

     impact

     of

     AI

     on

     human

     society

    ,

     including

     issues

     related

     to

     privacy

    ,

     bias

    ,

     and

     accountability

    .


     

     

    3

    .

     AI

    -driven

     personal

    ization

    :

     AI

     is

     becoming

     increasingly

     capable

     of

     personal

    izing

     our

     lives

    ,

     from

     recommendations

     for

     products

     and

     services

     to

     personalized

     learning

     experiences

    .

     This

     trend

     is

     likely

     to

     continue

    



```python
llm.shutdown()
```
