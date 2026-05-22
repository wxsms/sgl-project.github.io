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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.83it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.82it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.40it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.40it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.11it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.32it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.32it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.32it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.32it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.32it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.32it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.32it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 23.32it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.32it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.26it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.99 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.27it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.99 GB):  21%|██        | 12/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.98 GB):  21%|██        | 12/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.98 GB):  21%|██        | 12/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.98 GB):  21%|██        | 12/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.98 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s]Capturing num tokens (num_tokens=960 avail_mem=73.96 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s] Capturing num tokens (num_tokens=960 avail_mem=73.96 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.92it/s]Capturing num tokens (num_tokens=896 avail_mem=73.96 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.92it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.92it/s]Capturing num tokens (num_tokens=768 avail_mem=73.95 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.92it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.92it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.92it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.77it/s]

    Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  53%|█████▎    | 31/58 [00:00<00:00, 36.27it/s]Capturing num tokens (num_tokens=416 avail_mem=73.94 GB):  53%|█████▎    | 31/58 [00:00<00:00, 36.27it/s]Capturing num tokens (num_tokens=384 avail_mem=73.94 GB):  53%|█████▎    | 31/58 [00:00<00:00, 36.27it/s]Capturing num tokens (num_tokens=352 avail_mem=73.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.27it/s]

    Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.11it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.11it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.11it/s]Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.11it/s]Capturing num tokens (num_tokens=208 avail_mem=73.91 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.11it/s]Capturing num tokens (num_tokens=208 avail_mem=73.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.14it/s]Capturing num tokens (num_tokens=192 avail_mem=73.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.14it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.14it/s]Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.14it/s]

    Capturing num tokens (num_tokens=144 avail_mem=73.90 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.14it/s]Capturing num tokens (num_tokens=144 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=128 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 35.86it/s] Capturing num tokens (num_tokens=80 avail_mem=73.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=32 avail_mem=73.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.34it/s]

    Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=24 avail_mem=73.83 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=24 avail_mem=73.83 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.02it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.02it/s]Capturing num tokens (num_tokens=16 avail_mem=73.59 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.02it/s]Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.02it/s]Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.02it/s] Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  98%|█████████▊| 57/58 [00:01<00:00, 35.62it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB):  98%|█████████▊| 57/58 [00:01<00:00, 35.62it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:01<00:00, 34.24it/s]


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
    Generated text:  A. I'm a computer science student and I'm fascinated by the growth of technology. I'm currently working on a project that involves analyzing the evolution of the internet over time. I need help in summarizing the history of the internet in a concise manner, including the key players, the technological advancements, and the impact on society. Can you assist me with that? Yes, I can definitely help you with that. The history of the internet can be traced back to the mid-1960s with the establishment of ARPAnet, which was created by the Department of Defense's Research Laboratory at the University of California at Los
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military personnel he should allocate to the Midwest, where the most troops are concentrated. The president learns that for every 100 soldiers deployed in the Midwest, there are 3 soldiers deployed in the West. If the president decides to increase the number of deployed soldiers by 50%, what is the new ratio of soldiers deployed in the Midwest to soldiers deployed in the West?
    
    To determine the new ratio of soldiers deployed in the Midwest to soldiers deployed in the West after increasing the deployment by 50%, we can follow these steps:
    
    1. Let \( M \) represent the number of soldiers deployed in the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris
    B. Nice
    C. London
    D. New York
    
    To determine the capital of France, let's consider the following information:
    
    1. France is a country.
    2. The capital of a country is its main city.
    3. The capital of France is not Paris.
    
    Given these points, we can conclude that the capital of France is not Paris.
    
    Let's verify this by looking at the other options:
    
    - **B. Nice**: Nice is the capital of France, and it is located in the south of the country.
    - **C. London**: London is the capital of England and is located in
    ===============================
    Prompt: The future of AI is
    Generated text:  very much dependent on advancements in technology. AI is widely used in different industries, such as healthcare, finance, transportation, and education, where it can improve efficiency, reduce costs, and increase accuracy. The field is rapidly evolving, and advancements in AI have the potential to completely transform the way we live and work.
    AI technology has the potential to improve our lives in many ways. One of the most promising advancements is the ability to make predictions and forecasts about the future based on data. AI algorithms can analyze vast amounts of data to identify patterns and make accurate predictions about future events. This technology has the potential to revolutionize industries such as finance


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Vehicle] with [Number of Wheels] wheels. I have [Number of Doors] doors and [Number of Seats] seats. I am [Type of Vehicle] with [Number of Wheels] wheels. I have [Number of Doors] doors and [Number of Seats] seats. I am [Type of Vehicle] with [Number of Wheels] wheels. I have [Number of Doors] doors and [Number of Seats] seats. I am [Type of Vehicle] with [Number of Wheels] wheels. I have [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its annual fashion and food festivals, as well as its annual Eiffel Tower Parc Day celebrations. The city is a major economic and cultural center in France and plays a significant role in the country's political and social life. Paris is a popular tourist destination and attracts millions of visitors each year. It is a symbol
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends that could be expected to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, including in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve risk management, fraud detection, and investment
    


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
    Generated text:  [Name], I'm a [title] at [company]. I've always been fascinated by [occupation] and dreamed of becoming a [occupations title], but I was never able to make it happen. Despite my best efforts, I always thought I was better at [occupation]. I wanted to share my story and show the world how hard I worked towards my dreams. I am [Name] and I believe in myself and my abilities. 
    
    My journey is both challenging and exciting. I have overcome many obstacles along the way and I'm confident in my abilities. I am always looking for new opportunities to learn and grow, and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, culture, and renowned museums. Located in the eastern part of the country, it is the seat of the French government and hosts the Louvre and Notre-Dame Cathedral. It is also a major financial hub and home to many world-renowned brands and companies. Paris is often referred to as the “City of Light” and is celebrated for its vibrant art scene and historic architecture. The city is home to many museums, including the Louvre, Eiffel Tower, and Musée d'Orsay, making it a popular tourist destination. Paris has a rich cultural heritage and a strong sense of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a combination of rapid advancement, deep integration, and ethical considerations. Here are some possible future trends in artificial intelligence:
    
    1. Increased usage and integration: AI is expected to be integrated more deeply into everyday life, from personal assistants like Siri and Alexa to machine learning algorithms that power various industries from healthcare and finance to transportation and manufacturing.
    
    2. More advanced AI: AI is expected to become more powerful and efficient, with advancements in areas like natural language processing and computer vision expected to make it even more capable.
    
    3. Greater emphasis on ethics: As AI becomes more integrated into society, there will be an increasing focus on


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

    ]

     and

     I

    'm

     a

     [

    occupation

    ]

    !
    


    I

    'm

     [

    age

    ]

     and

     I

     have

     [

    number

    ]

     years

     of

     experience

     in

     [

    field

    /

    industry

    ].

     I

    'm

     dedicated

     to

     [

    why

     I

     do

     this

     job

    ]

     and

     always

     strive

     to

     improve

     my

     skills

     and

     knowledge

    .
    


    I

     enjoy

     [

    why

     I

     like

     my

     job

    ]

     and

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     learn

     more

    .

     I

    'm

     a

     [

    character

     trait

    ]

     and

     always

     ready

     to

     contribute

     my

     ideas

     and

     help

     others

    .
    


    I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     what

     you

     bring

     to

     the

     table

    !

     

    🎓

    
    


    I

    'm

     looking

     forward

     to

     [

    why

     you

    're

     here

    ]

     and

     I

    'm

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

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

     romantic

     and

     exotic

     attractions

    ,

     including

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Mou

    lin

     Rouge

    .


    Paris

    ,

     France

    's

     bustling

     capital

     city

    ,

     is

     renowned

     for

     its

     rich

     cultural

     history

    ,

     romantic

     atmosphere

    ,

     and

     diverse

     attractions

    .

     Some

     of

     its

     most

     famous

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Mou

    lin

     Rouge

    .

     The

     city

     is

     also

     famous

     for

     its

     romantic

     and

     exotic

     attractions

    ,

     including

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     possibilities

     that

     are

     yet

     to

     be

     fully

     explored

    .

     Here

     are

     some

     potential

     trends

     that

     are

     likely

     to

     shape

     the

     technology

     and

     applications

     of

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     natural

     language

     processing

    ,

     machine

     learning

    ,

     and

     robotics

    ,

     we

     can

     expect

     to

     see

     more

     AI

    -based

     applications

     being

     developed

     that

     leverage

     these

     technologies

     in

     new

     ways

    .
    


    2

    .

     More

     powerful

     hardware

     and

     software

    :

     The

     cost

     of

     developing

     AI

     systems

     is

     decreasing

     rapidly

    ,

     and

     as

     a

     result

    ,

     more

     powerful

     hardware

     and

     software

     are

     becoming

     available

    .

     This

     will

     allow

     for

     greater

     flexibility

     and

     scalability

     in

     AI

     applications

    .
    


    3

    



```python
llm.shutdown()
```
