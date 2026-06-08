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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.51it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.26it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.26it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.92it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 38.90it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 38.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:04, 13.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:04, 13.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:04, 13.27it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:04, 13.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:03, 16.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:03, 16.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:03, 16.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:03, 16.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.35it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.35it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.35it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.52it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.52it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.24it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.24it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.45it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.45it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.45it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.45it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.45it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.45it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:01<00:00, 37.19it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.27it/s]

    Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.30it/s]

    Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.29it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.29it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.29it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.29it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.29it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 35.97it/s]


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
    Generated text:  Len and I am a creative visual artist from Ontario, Canada. I currently live in Toronto, Ontario, Canada and have been since 2018. I have been active in the visual arts for over 10 years.
    I began drawing in my 4th grade, and have been drawing for my entire life. I like to create unique and interesting images and I love the freedom of the visual arts.
    My artwork is always very personal and unique to the people I work with. My goal is to create something that evokes a response in the viewer. I am always working on new projects and am always trying to create something
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ( ) position, holding ( ) office, which is the highest office in the United States, corresponding to the president of the United States.
    A. Senior, ceremonial
    B. Senior, official
    C. Deputy, ceremonial
    D. Deputy, official
    
    The president of the United States is a senior position, holding a ceremonial office, which is the highest office in the United States, corresponding to the president of the United States. Therefore, the correct answer is:
    
    A. Senior, ceremonial
    
    To summarize the answer in the context of the provided options:
    
    A. Senior, ceremonial
    
    Option A is the correct choice. The correct
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A) Paris
    B) London
    C) Madrid
    D) Berlin
    The capital of France is:
    
    A) Paris
    
    Paris is the capital city of France, located on the North Bank of the Seine in the Île de France. It is one of the most famous cities in Europe and is known for its rich history, beautiful architecture, and annual celebrations. London is the capital of the United Kingdom, while Madrid and Berlin are the capital cities of Spain and Germany, respectively.
    ===============================
    Prompt: The future of AI is
    Generated text:  clear. The technology has the potential to transform industries, change the way we live, and open up entirely new possibilities. However, as technology evolves, the risks and challenges it poses are also increasing.
    AI is a broad term that refers to a broad range of technologies that involve the use of computers to perform tasks that would previously have required human intervention. These tasks include tasks that are analytical, creative, or rule-based. Examples of AI technologies include natural language processing, computer vision, machine learning, and robotics.
    While AI has numerous potential benefits, it also poses significant risks and challenges. One of the most significant risks is the potential for AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that was founded in 987 AD and is the largest city in the European Union. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its fashion industry, with many famous fashion designers and boutiques. Paris is a popular tourist destination and is home to many museums, theaters, and restaurants. It is a city that is steeped in history and culture, and is a must-visit for anyone interested in French culture and history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased integration with other technologies: AI will continue to be integrated with other technologies such as blockchain, IoT, and autonomous vehicles, creating a more interconnected and integrated world.
    
    2. Enhanced privacy and security: As AI becomes more prevalent, there will be a need for increased privacy and security measures to protect user data and prevent misuse of AI systems.
    
    3. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations and the responsible use of AI systems
    


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
    Generated text:  [insert your name here]. I am [insert your profession, education level, or any other relevant information here]. I am a dedicated and passionate learner who always strives to learn new things, stay updated on the latest trends and technologies, and contribute to the community in any way I can. I am constantly seeking to improve myself and make a positive impact on the world. I am excited to meet and collaborate with anyone who shares my passion for learning and the desire to make the world a better place. Hello! My name is [insert your name here], and I am a dedicated and passionate learner who always strives to learn new things, stay
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Ville Lumière," and is the largest city in Europe by population. It is a major urban area with a rich history, culture, and cuisine, and is renowned for its Gothic architecture, opera, and iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral. France's capital is a popular tourist destination, with millions of visitors annually making it the second-largest urban center in the world by population and gross domestic product. It is also home to numerous notable museums, art galleries, and historical sites, including the Louvre Museum, the National Library, and the Musée d'Orsay
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of emerging trends, including:
    
    1. Increased focus on ethics and privacy: With the increasing amount of personal data that is being generated and stored, there is a growing concern about the ethical and privacy implications of AI. This means that there will be an increased focus on developing AI that is designed with privacy and data protection in mind.
    
    2. More reliance on AI in healthcare: AI is already being used in a wide range of healthcare applications, from diagnosis and treatment planning to drug discovery and personalized medicine. As AI technology improves and becomes more accessible, it is likely that it will play a more significant role in healthcare.
    
    


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

    ]

     and

     I

    'm

     a

     [

    insert

     your

     job

     or

     profession

     here

    ,

     e

    .g

    .,

     "

    digital

     marketer

    ",

     "

    software

     engineer

    ",

     "

    ped

    iatric

    ian

    "].

     I

    'm

     passionate

     about

     [

    insert

     your

     main

     interest

     or

     hobby

     here

    ,

     e

    .g

    .,

     "

    travel

    ing

    ",

     "

    reading

    ",

     "

    music

    ".

    ]

     and

     I

     believe

     that

     my

     skills

     and

     knowledge

     can

     be

     used

     to

     help

     people

     make

     the

     most

     of

     their

     time

     and

     improve

     their

     lives

    .
    


    Throughout

     my

     career

    ,

     I

    've

     gained

     valuable

     experience

     and

     knowledge

     in

     [

    insert

     the

     technology

     or

     field

     you

     specialize

     in

    ,

     e

    .g

    .,

     "

    the

     internet

    ",

     "

    AI

    ",

     "

    software

     development

    ".

    ]

     and

     I

    'm

     excited

     to

     bring

     my

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     known

     for

     its

     rich

     history

     and

     vibrant

     culture

    .

     
    


    Here

    's

     the

     same

     information

     in

     different

     formats

    :
    


    1

    .

     A

     single

     sentence

    :

     "

    Paris

    ,

     the

     capital

     of

     France

    ,

     is

     famous

     for

     its

     rich

     history

     and

     vibrant

     culture

    .

     "


    2

    .

     A

     sentence

     using

     a

     pre

    position

    :

     "

    In

     Paris

    ,

     the

     capital

     of

     France

    ,

     the

     rich

     history

     and

     vibrant

     culture

     shine

     through

    .

     "


    3

    .

     A

     sentence

     using

     a

     coordinating

     conjunction

    :

     "

    Paris

    ,

     the

     capital

     of

     France

    ,

     is

     recognized

     for

     its

     rich

     history

     and

     vibrant

     culture

    .

     "


    4

    .

     A

     sentence

     with

     a

     relative

     pron

    oun

    :

     "

    The

     capital

     of

     France

    ,

     Paris

    ,

     is

     famous

     for

     its

     rich

     history

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     one

     of

     revolution

    ,

     and

     the

     potential

     to

     transform

     industries

    ,

     society

    ,

     and

     our

     very

     way

     of

     life

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     that

     are

     likely

     to

     shape

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     In

     the

     past

    ,

     the

     development

     of

     AI

     has

     been

     largely

     driven

     by

     profit

     motives

    .

     However

    ,

     with

     ethical

     considerations

     becoming

     more

     prominent

    ,

     the

     AI

     industry

     is

     likely

     to

     prioritize

     the

     development

     of

     AI

     that

     is

     aligned

     with

     ethical

     guidelines

    .

     This

     could

     lead

     to

     the

     development

     of

     AI

     that

     is

     more

     transparent

    ,

     accountable

    ,

     and

     ethical

    .
    


    2

    .

     Enhanced

     AI

     transparency

    :

     As

     AI

     becomes

     more

     advanced

    ,

     the

     ability

     to

     understand

     and

    



```python
llm.shutdown()
```
