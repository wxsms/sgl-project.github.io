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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.26it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.26it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.26it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.26it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.26it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.26it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.26it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.26it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.26it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.26it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.28it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.28it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.28it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.28it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.28it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.28it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.28it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.28it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.28it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.28it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.28it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.41it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.41it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.41it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.41it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.41it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.41it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 23.41it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.41it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.36it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.36it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.36it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.36it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.36it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.36it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.36it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.36it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.36it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.19 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.18 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=54.18 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.18 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=54.18 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.17 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.14it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=54.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.14 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.14 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.14 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s]Capturing num tokens (num_tokens=960 avail_mem=54.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s] Capturing num tokens (num_tokens=896 avail_mem=54.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s]

    Capturing num tokens (num_tokens=832 avail_mem=54.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.65it/s]Capturing num tokens (num_tokens=832 avail_mem=54.08 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=768 avail_mem=54.07 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=704 avail_mem=54.07 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=640 avail_mem=54.07 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=576 avail_mem=54.07 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=512 avail_mem=54.05 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=512 avail_mem=54.05 GB):  50%|█████     | 29/58 [00:00<00:00, 37.86it/s]Capturing num tokens (num_tokens=480 avail_mem=54.07 GB):  50%|█████     | 29/58 [00:00<00:00, 37.86it/s]Capturing num tokens (num_tokens=448 avail_mem=54.07 GB):  50%|█████     | 29/58 [00:00<00:00, 37.86it/s]

    Capturing num tokens (num_tokens=416 avail_mem=54.06 GB):  50%|█████     | 29/58 [00:00<00:00, 37.86it/s]Capturing num tokens (num_tokens=384 avail_mem=54.06 GB):  50%|█████     | 29/58 [00:00<00:00, 37.86it/s]Capturing num tokens (num_tokens=352 avail_mem=54.06 GB):  50%|█████     | 29/58 [00:00<00:00, 37.86it/s]Capturing num tokens (num_tokens=352 avail_mem=54.06 GB):  59%|█████▊    | 34/58 [00:00<00:00, 40.52it/s]Capturing num tokens (num_tokens=320 avail_mem=54.05 GB):  59%|█████▊    | 34/58 [00:00<00:00, 40.52it/s]Capturing num tokens (num_tokens=288 avail_mem=54.05 GB):  59%|█████▊    | 34/58 [00:00<00:00, 40.52it/s]Capturing num tokens (num_tokens=256 avail_mem=54.05 GB):  59%|█████▊    | 34/58 [00:00<00:00, 40.52it/s]Capturing num tokens (num_tokens=240 avail_mem=54.04 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=224 avail_mem=54.04 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=224 avail_mem=54.04 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.45it/s]Capturing num tokens (num_tokens=208 avail_mem=54.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.45it/s]

    Capturing num tokens (num_tokens=192 avail_mem=54.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.45it/s]Capturing num tokens (num_tokens=176 avail_mem=54.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.45it/s]Capturing num tokens (num_tokens=160 avail_mem=54.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.45it/s]Capturing num tokens (num_tokens=144 avail_mem=54.02 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.45it/s]Capturing num tokens (num_tokens=144 avail_mem=54.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=128 avail_mem=54.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=112 avail_mem=54.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=96 avail_mem=54.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.72it/s] Capturing num tokens (num_tokens=80 avail_mem=54.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=64 avail_mem=54.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=64 avail_mem=54.01 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=48 avail_mem=54.00 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]

    Capturing num tokens (num_tokens=32 avail_mem=54.00 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=28 avail_mem=54.00 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=24 avail_mem=53.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=20 avail_mem=53.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=20 avail_mem=53.99 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.35it/s]Capturing num tokens (num_tokens=16 avail_mem=53.99 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.35it/s]Capturing num tokens (num_tokens=12 avail_mem=53.98 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.35it/s]Capturing num tokens (num_tokens=8 avail_mem=53.98 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.35it/s] Capturing num tokens (num_tokens=4 avail_mem=53.98 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.35it/s]Capturing num tokens (num_tokens=4 avail_mem=53.98 GB): 100%|██████████| 58/58 [00:01<00:00, 39.20it/s]


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
    Generated text:  Andrei and I'm a Student at MIT. I have been studying calculus for the last 3 years, so I'm quite comfortable with it. I'm also really good at improving my math skills and my math teacher has been really good to me. I have always loved math and math problems. I have taken math courses, including some calculus. I'm interested in pursuing a PhD in math in the future. I want to understand the workings of mathematics and learn how to apply math in real life situations. And I want to find a career that uses math in a practical way. Thank you for taking the time to read my application.
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the Vice President. Which of the following would be the correct sentence to use? The Vice President of the United States represents the President of the United States. To determine the correct sentence to use, let's break down the statement and its components.
    
    1. **Identify the key elements:**
       - President of the United States
       - Vice President of the United States
       - The relationship between them
    
    2. **Comprehend the relationship:**
       - The Vice President of the United States serves as the President’s primary spokesperson in the executive branch of the United States government.
       - The Vice President represents the
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lyon
    C. Saint-Étienne
    D. Lyon
    Answer:
    
    A
    
    The four circles inside a circle are called the circumcircle of the cyclic quadrilateral. In the figure, the circumcircle of the cyclic quadrilateral \(ABCD\) is a circle with a diameter equal to the diagonal \(AC\) of the quadrilateral. What is the measure of the angle \(A\)?
    (A) \(60^\circ\)
    (B) \(40^\circ\)
    (C) \(30^\circ\)
    (D) \(10^\circ\)
    (E) \(2
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting, with developments in the field promising to revolutionize various industries and bring new opportunities to society. The concept of AI can be traced back to the 1950s when Claude Shannon, a professor at the University of Illinois, proposed that computers could perform tasks that human beings currently do. Today, AI has advanced and evolved from these early ideas, with new developments in the field bringing about remarkable advancements in the modern day. With the rapid advancement of technology, the use of AI is becoming more common in different fields, making it easier and quicker for businesses to operate and accomplish their goals. The need for skilled AI professionals has increased


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field. What's your favorite hobby or activity? I'm a [job title] at [company name], and I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and economic center with a rich history dating back to the Roman Empire. Paris is a major hub for international trade and tourism, and is home to many renowned museums, art galleries, and restaurants. The city is also known for its vibrant nightlife and fashion scene. Paris is a city of contrasts, with its rich history and modern culture making it a unique and fascinating destination for visitors. The city is also home to many important institutions of higher education, including the University of Paris and the Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations, such as privacy, fairness, and accountability. This could lead to more stringent regulations and standards
    


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
    Generated text:  [Name] and I am [Age] years old. I come from [Current Location], and I am a [Occupation] who has always been [Why] and [Why not]. My best friend is [Name of Best Friend] and I am [Interests or Personality Traits], and I'm also [Why].
    
    [Name] is [Name of Profession] who believes in [Why]. I am [Why I Am What I Am]. I am here because [Why]. Whether you know me or not, I want you to know that I'm here to be your [Why].
    
    Thanks for taking the time to meet me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known as the "City of Light" due to its rich history and vibrant culture. The city is home to many world-renowned museums, artistic venues, and theaters, and is also renowned for its stunning architecture, including the iconic Eiffel Tower. Additionally, Paris is a popular tourist destination, with its rich history and culture making it a must-visit for visitors of all ages. The city is also home to a diverse and vibrant community of French people, who enjoy a wide range of cultural and social activities. Overall, Paris is a truly unique and unforgettable city that offers a glimpse into the rich history and culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a number of exciting and potentially transformative trends that will shape how we live, work, and interact with technology. Here are some of the key trends that may be poised to shape the future of AI:
    
    1. Increased Integration with Human Decision-Making: As AI becomes more advanced, it is likely to become more integrated with human decision-making processes. This may lead to a greater emphasis on human expertise and the ability to collaborate with AI systems in complex decision-making situations.
    
    2. Development of AI Ethics and Legal Guidelines: As AI systems become more sophisticated, there will likely be an increase in the development of guidelines and ethical


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

    ],

     and

     I

    'm

     a

     friendly

     AI

     assistant

    .

     I

    'm

     here

     to

     help

     you

     with

     any

     questions

     or

     tasks

     you

     need

     help

     with

    .

     Whether

     you

     need

     information

     about

     your

     schedule

    ,

     homework

     help

    ,

     or

     just

     general

     information

    ,

     I

    'm

     here

     to

     assist

     you

    .

     So

    ,

     what

    's

     your

     name

    ?

     And

     what

     can

     I

     help

     you

     with

     today

    ?

     Let

     me

     know

    .

     [

    Your

     Name

    ]

    
    
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

     both

     France

     and

     the

     European

     Union

     and

     has

     been

     the

     seat

     of

     government

     since

     

    1

    0

    7

    7

    .
    


    The

     French

     capital

     is

     located

     on

     the

     left

     bank

     of

     the

     Se

    ine

     River

    ,

     which

     flows

     through

     the

     center

     of

     the

     city

    .

     It

     is

     home

     to

     the

     Palace

     of

     Vers

    ailles

    ,

     the

     largest

     palace

     in

     the

     world

    ,

     and

     to

     the

     Ch

    amps

    -

    É

    lys

    ées

    ,

     a

     famous

     avenue

     with

     numerous

     monuments

     and

     landmarks

    .

     The

     city

     also

     has

     several

     museums

     and

     art

     galleries

    ,

     such

     as

     the

     Mus

    ée

     d

    '

    Or

    say

     and

     the

     Mus

    ée

     Rod

    in

    .

     Paris

     is

     a

     cultural

     and

     economic

     center

     with

     a

     rich

     history

     and

     a

     diverse

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     advancements

     in

     technology

     and

     the

     development

     of

     new

     algorithms

     and

     techniques

     that

     will

     make

     it

     more

     efficient

    ,

     accurate

    ,

     and

     versatile

    .

     Some

     possible

     future

     trends

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     there

     will

     be

     a

     growing

     emphasis

     on

     ensuring

     that

     they

     are

     used

     eth

    ically

     and

     responsibly

    .

     This

     may

     include

     considerations

     such

     as

     data

     privacy

    ,

     bias

    ,

     and

     safety

    .
    


    2

    .

     Integration

     of

     AI

     into

     human

     decision

    -making

    :

     AI

     will

     continue

     to

     become

     more

     integrated

     into

     human

     decision

    -making

     processes

    ,

     enabling

     more

     sophisticated

     and

     nuanced

     analyses

     of

     complex

     data

    .
    


    3

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     will

     play

     a

     more

    



```python
llm.shutdown()
```
