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

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:47,  5.04s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:47,  5.04s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:47,  5.04s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:47,  5.04s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:47,  5.04s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.87it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.22it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.22it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.22it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.22it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.22it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.22it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.22it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.22it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.22it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 26.58it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 26.58it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 26.58it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 26.58it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 26.58it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 26.58it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 26.58it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 26.58it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 26.58it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 33.54it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 33.54it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 33.54it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 33.54it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 33.54it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.91it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.91it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 20.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 20.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.24it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.49it/s] Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.32it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.32it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.32it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.32it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.32it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.32it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.72it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.72it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.72it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.72it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.72it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.72it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.29it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.96it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.96it/s] Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  81%|████████  | 47/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.49it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.49it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.49it/s] Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 36.58it/s]


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
    Generated text:  Maximilian, a fourth year medical student at the University of California, San Francisco. I am in the program to learn about the therapeutic potential of vaccines. I have a passion for treating people that suffer from chronic inflammatory disease like rheumatoid arthritis. If you are interested in learning about the science behind how vaccines work, read on. In my free time I am studying infectious diseases, testing new medications, and going for long walks in the park. To me, a great medicine is one that is good for me.
    
    How did you get started in the medical field? I have always been drawn to health care, and felt that helping people
    ===============================
    Prompt: The president of the United States is
    Generated text:  30 years older than the president of Brazil. The president of Brazil is taller than the president of France by 3 years. If the president of France is 30 years old, how old is the president of Brazil?
    To determine the age of the president of Brazil, we need to follow these steps:
    
    1. Identify the age of the president of France.
    2. Determine the age of the president of Brazil based on the information given.
    
    First, we know the age of the president of France:
    \[ \text{Age of the president of France} = 30 \text{ years} \]
    
    Next, we
    ===============================
    Prompt: The capital of France is
    Generated text:  located in which region?
    A) Region of Paris
    B) Region of Lyon
    C) Region of Bordeaux
    D) Region of Tours
    Answer: A
    
    The capital of France is located in the region of Paris. Therefore, the correct answer is: A) Region of Paris.
    
    So, the answer to the question "What is the capital of France? " is: Paris.  
    This question tests the knowledge of the capitals of major world countries, being a basic question. It requires students to know the capitals of major world countries.  
    In fact, the capital of France is located in the Region of Paris. Therefore, the
    ===============================
    Prompt: The future of AI is
    Generated text:  here
    
    The future of AI is here. It's growing, with real-world applications in a variety of industries. However, it's also in the hands of the founders of companies that make it. As these companies grow, so will the level of AI that's accessible to users.
    
    The rise of AI is a phenomenon in its own right, and its impact on every industry is vast. In many cases, it has already been on the rise for decades. The field has been largely developed and evolved in the United States by the last 50 years. It is expanding even further, but it is also being accelerated by many forces that


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a bustling metropolis with a rich cultural heritage and is a popular tourist destination. The city is known for its diverse cuisine, including French cuisine, and is home to many famous museums, including the Louvre and the Musée d'Orsay. Paris is a city of contrasts, with its historic architecture and modern art, and is a major hub for business and commerce in Europe. The city is also known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced interactions between machines and humans.
    
    2. Enhanced natural language processing: AI will continue to improve its ability to understand and respond to natural language, leading to more sophisticated and context-aware interactions.
    
    3. Improved machine learning algorithms: AI will continue to improve its ability to learn from data and make more accurate predictions and decisions.
    
    4. Increased use of AI in healthcare: AI will be used to improve the accuracy and efficiency of medical diagnosis and treatment, leading
    


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
    Generated text:  [Name] and I'm a [Age] year old [Occupation] with a [Skill] in [Field]. I have [Number] years of experience in [Field] and [Number] years of experience in [Other Field]. I love [Favorite Activity] and [Favorite Hobby]. I also have [Number] friends, [Number] family members, [Number] pets, and [Number] hobbies. My favorite thing about [Field] is [Favorite Activity], and I love [Favorite Hobby]. I can [Number] in [Time Frame], [Number] in [Time Frame], [Number] in [Time
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is located on the Seine River, in the northern part of the country, and is the largest city in Europe. It is also known as "the city of lights" and the "City of Love" due to its famous地标建筑，如埃菲尔铁塔、卢浮宫和巴黎圣母院。此外，巴黎是法国的首都，拥有悠久的历史和丰富的文化遗产，包括埃菲尔铁塔、卢浮宫、巴黎圣母院等著名地标建筑。除此之外，法国的巴黎也是一个充满活力和浪漫气息的城市，拥有独特的文化和美食。总之，巴黎是法国的文化、艺术和
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to develop and evolve, driven by the increasing complexity of our world, the increasing demand for artificial intelligence in various industries, and the increasing need for advanced algorithms and models to handle increasingly complex tasks.
    
    One potential future trend is the development of more advanced and sophisticated AI systems that can perform tasks that were previously considered too complex for traditional AI systems. This could include tasks that require more advanced natural language processing, machine learning, and deep learning algorithms, as well as tasks that require the ability to interpret and reason about complex social and political issues.
    
    Another potential trend is the development of more ethical and responsible AI systems that prioritize the well-being


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

     [

    age

    ],

     [

    gender

    ].

     I

    've

     always

     been

     fascinated

     by

     [

    field

     or

     interest

    ],

     [

    explain

     briefly

     why

     you

    're

     interested

     in

     it

    ].


    As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     a

     physical

     existence

     or

     a

     personal

     history

    ,

     so

     I

     can

    't

     provide

     information

     about

     my

     age

     or

     gender

    .

     However

    ,

     I

    'm

     happy

     to

     have

     a

     friendly

     conversation

     about

     anything

     you

    'd

     like

    !

     How

     can

     I

     assist

     you

     today

    ?

     [

    Name

    ]

     is

     an

     AI

     language

     model

    ,

     trained

     to

     assist

     with

     a

     wide

     range

     of

     tasks

     and

     inquiries

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

     you

     might

     have

     and

     answer

     your

     questions

     in

     a

     helpful

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     a

     historical

     and

     cultural

     center

     that

     has

     been

     a

     part

     of

     France

     since

     ancient

     times

    .
    


    France

    's

     capital

     city

     is

     Paris

    .

     It

     has

     a

     rich

     history

     and

     culture

     dating

     back

     to

     ancient

     times

    ,

     and

     it

     is

     a

     popular

     tourist

     destination

    .

     Its

     name

     is

     derived

     from

     the

     Latin

     word

     "

    par

    v

    ulus

    ,"

     meaning

     "

    small

    ,"

     as

     it

     was

     originally

     a

     small

     town

     that

     grew

     into

     the

     capital

     of

     France

    .

     Paris

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

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     also

     has

     a

     vibrant

     and

     diverse

     population

    ,

     which

     contributes

     to

     its

     economic

     growth

     and

     cultural

     diversity

    .

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

    ,

     and

     we

     can

     expect

     to

     see

     many

     different

     trends

     developing

     in

     the

     coming

     years

    .

     Here

     are

     a

     few

     of

     the

     most

     likely

     trends

     we

     may

     see

     in

     the

     AI

     field

    :
    


    1

    .

     More

     sophisticated

     and

     complex

     AI

     systems

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     sophisticated

     and

     complex

     AI

     systems

     that

     can

     handle

     a

     wider

     range

     of

     tasks

     and

     applications

    .

     These

     systems

     may

     incorporate

     natural

     language

     processing

    ,

     computer

     vision

    ,

     and

     other

     advanced

     technologies

     to

     perform

     tasks

     that

     were

     previously

     impossible

     for

     humans

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     many

     different

     areas

     of

     healthcare

    ,

     from

     diagnosis

     and

     treatment

     planning

     to

     patient

    



```python
llm.shutdown()
```
