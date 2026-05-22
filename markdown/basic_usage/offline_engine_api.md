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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.17it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.17it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.17it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.17it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.17it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.17it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.17it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.17it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 13.50it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 13.50it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 13.50it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 13.50it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 13.50it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 13.50it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 13.50it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 13.50it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 13.50it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 19.75it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 19.75it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 28.08it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 28.08it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 28.08it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 28.08it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 28.08it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 28.08it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 28.08it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 28.08it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 28.08it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 28.08it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 28.08it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 38.14it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 38.14it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 38.14it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 38.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.75 GB):   3%|▎         | 2/58 [00:00<00:03, 17.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:03, 17.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:03, 17.17it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=73.74 GB):   7%|▋         | 4/58 [00:00<00:03, 15.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.74 GB):   7%|▋         | 4/58 [00:00<00:03, 15.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.46 GB):   7%|▋         | 4/58 [00:00<00:03, 15.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.46 GB):  10%|█         | 6/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.75 GB):  10%|█         | 6/58 [00:00<00:03, 14.82it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.75 GB):  10%|█         | 6/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.75 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.75 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.87it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.74 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.87it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.74 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.74 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.59it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.73 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.73 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.73 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.73 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.73 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.72 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.72 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.71it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=72.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.71it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.69 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.71it/s]Capturing num tokens (num_tokens=960 avail_mem=72.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.71it/s] Capturing num tokens (num_tokens=960 avail_mem=72.70 GB):  38%|███▊      | 22/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=896 avail_mem=72.70 GB):  38%|███▊      | 22/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=832 avail_mem=72.70 GB):  38%|███▊      | 22/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=768 avail_mem=72.69 GB):  38%|███▊      | 22/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=704 avail_mem=72.69 GB):  38%|███▊      | 22/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=704 avail_mem=72.69 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.01it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.01it/s]

    Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.01it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.01it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.01it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=416 avail_mem=72.25 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  60%|██████    | 35/58 [00:01<00:00, 36.72it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  60%|██████    | 35/58 [00:01<00:00, 36.72it/s]

    Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  60%|██████    | 35/58 [00:01<00:00, 36.72it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  60%|██████    | 35/58 [00:01<00:00, 36.72it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  60%|██████    | 35/58 [00:01<00:00, 36.72it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  60%|██████    | 35/58 [00:01<00:00, 36.72it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=128 avail_mem=72.21 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=128 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.29it/s]

    Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.29it/s] Capturing num tokens (num_tokens=80 avail_mem=72.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=32 avail_mem=72.19 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.36it/s]

    Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=12 avail_mem=71.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=8 avail_mem=72.14 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.86it/s] Capturing num tokens (num_tokens=4 avail_mem=72.14 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=4 avail_mem=72.14 GB): 100%|██████████| 58/58 [00:01<00:00, 30.35it/s]


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
    Generated text:  Gina. I am a twenty-five-year-old female. I was born in the United States. I've lived in many places, such as Los Angeles, San Francisco and Chicago. I was born with a rare condition. I have the condition known as congenital heart disease. A heart disease that affects the heart and blood vessels of a baby. It can happen in the baby before birth or after birth. The baby's heart isn't formed yet, so it's a baby heart. The heart and the blood vessels of a baby heart are different from those of a regular person. One of the major problems with congenital heart disease is that
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military officers to allocate to each branch of the military. He has 100 military officers and needs 100 soldiers, 50 sailors, and 50 civilians. The president has a budget that only allows him to buy 50 units of additional equipment for the military. If he allocates each unit of additional equipment equally between the three branches, how many units of additional equipment should he allocate to each branch?
    To determine how many units of additional equipment the president should allocate to each branch, we need to follow these steps:
    
    1. Calculate the total number of units of additional equipment required
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city in the European Union and the third-largest city in the European continent. It is a major metropolis and is located in the 4th department of the Loire Valley, with its northern and western suburbs in the Loire Valley. The city is situated on the banks of the Seine River, which flows from the eastern to the western bank, from the eastern end of which point to the western end of the city. The Seine is one of the longest rivers in the world and the second-longest in Europe. Paris is the capital of the department of the Seine-et-Oise.
    
    Choose
    ===============================
    Prompt: The future of AI is
    Generated text:  not a dark mystery
    
    This is a special report from the board of directors of the European Artificial Intelligence Chatham House, a forum of the European Academy for Artificial Intelligence.
    
    There are five main areas of AI development and they are: robotics, machine learning, natural language understanding and understanding, the Internet of Things, and AI ethics. These are the five major areas that are driving the development of AI. The other areas of development are the software developers, the users, the distributors, the users of the tools, and the users of the results. We want to start with robotics, and the way it is developing is interesting.
    
    There are two


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and the country's cultural and political capital. Paris is a major tourist destination and a popular destination for international business and diplomacy. The city is known for its rich history, art, and cuisine, and is a major center for the arts, science, and technology. It is also home to many famous museums, including the Musée d'Orsay, the Musée Rodin, and the Musée Rodin. Paris is a vibrant and diverse city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread adoption in healthcare.
    
    2. AI in finance: AI is already being used in finance to improve risk management, fraud detection, and trading. As AI technology continues to improve, we can expect to see even more widespread adoption in finance.
    
    3. AI in manufacturing
    


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
    Generated text:  [Name], and I'm a [Title] at [Company]. I've been [number of years] years in the field of [profession], and I've always been passionate about [why you love what you do]. My background is [any relevant education, training or experience], and I've always been dedicated to [why you're passionate about your work]. What excites me the most is [exciting thing about your work that makes you want to keep doing it].
    Hello, my name is [Name], and I'm a [Title] at [Company]. I've been [number of years] years in the field of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Choose your answer from:
     *yes;
     *no;
    Is the question answered in the same general way in the second sentence? yes;
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be a complex one, with many potential developments and applications. Here are some possible trends in AI in the next few years:
    
    1. Increased automation and specialization: The development of AI-driven robots and machines will continue to automate routine tasks and increase the specialization of AI systems. This could lead to more efficient production processes, improved customer service, and increased productivity.
    
    2. Personalization and artificial intelligence: AI will become more sophisticated and will be able to analyze and understand user behavior and preferences. This will lead to more personalized experiences, such as recommendations and targeted advertisements.
    
    3. Enhanced privacy and security: As AI systems become more sophisticated,


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

     [

    character

     type

    ]

     with

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     the

     industry

    .

     I

    'm

     passionate

     about

     [

    job

     title

    ]

     and

     I

     enjoy

     [

    what

     you

     enjoy

     most

     about

     your

     job

    ].

     I

     thrive

     on

     [

    what

     motiv

    ates

     you

     most

     about

     your

     job

    ].

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

     your

     journey

    .

     Can

     you

     tell

     me

     a

     bit

     about

     yourself

    ?

     I

    'd

     love

     to

     get

     to

     know

     you

     better

    .

     Hello

    !

     My

     name

     is

     [

    Your

     Name

    ].

     I

    ’m

     a

     [

    character

     type

    ]

     with

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     the

     industry

    .

     I

    ’m

     passionate

     about

     [

    job

     title

    ]

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Paris

    ienne

    ."
    


    A

    .

     Correct

    


    B

    .

     Incorrect

    
    


    B

    .

     Incorrect

    
    


    The

     capital

     of

     France

     is

     actually

     Paris

    ,

     not

     "

    La

     Paris

    ienne

    ."

     The

     city

    's

     official

     name

     is

     not

     "

    La

     Paris

    ienne

    ,"

     but

     rather

     "

    Paris

    ,"

     which

     is

     the

     historic

     name

     for

     the

     city

    .

     The

     official

     name

     "

    La

     Paris

    ienne

    "

     was

     once

     widely

     used

    ,

     but

     has

     been

     replaced

     by

     "

    Paris

    ."

     While

     Paris

     is

     the

     largest

     city

     in

     France

     by

     population

    ,

     "

    La

     Paris

    ienne

    "

     is

     the

     general

     term

     for

     the

     city

    .

     The

     French

     name

     is

     more

     widely

     recognized

     in

     international

     communications

    .

     Therefore

    ,

     the

     correct

     answer

     is

     B

    ,

     "

    Incorrect

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     technological

     advancements

     and

     a

     shift

     towards

     more

     diverse

     and

     ethical

     applications

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     AI

     becomes

     more

     prevalent

    ,

     the

     focus

     will

     shift

     towards

     ensuring

     that

     AI

     systems

     are

     developed

     and

     used

     eth

    ically

     and

     responsibly

    .

     This

     includes

     designing

     AI

     systems

     that

     are

     transparent

    ,

     fair

    ,

     and

     accountable

    ,

     and

     ensuring

     that

     the

     ethical

     implications

     of

     AI

     use

     are

     considered

     throughout

     the

     development

     process

    .
    


    2

    .

     More

     diverse

     and

     representative

     AI

    :

     The

     current

     AI

     landscape

     is

     largely

     dominated

     by

     technology

     developed

     by

     the

     West

    ,

     with

     contributions

     from

     other

     cultures

     and

     languages

    .

     There

     is

     a

     growing

     trend

     towards

     more

     diverse

     and

    



```python
llm.shutdown()
```
