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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.62it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:19,  4.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:19,  4.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:19,  4.54s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:19,  4.54s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:19,  4.54s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.27it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.31it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.31it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.16it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 17.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 17.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:03, 17.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:03, 17.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.00 GB):   9%|▊         | 5/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):   9%|▊         | 5/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.79it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.79it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.43it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.43it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.98 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.43it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.57it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.57it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.57it/s]Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.57it/s] Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=768 avail_mem=70.95 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.38it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]

    Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=416 avail_mem=70.94 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=416 avail_mem=70.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.36it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.36it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.36it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.36it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.36it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.36it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]

    Capturing num tokens (num_tokens=208 avail_mem=70.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=192 avail_mem=70.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=128 avail_mem=70.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.02it/s] Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  81%|████████  | 47/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=80 avail_mem=70.89 GB):  81%|████████  | 47/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  81%|████████  | 47/58 [00:01<00:00, 44.49it/s]

    Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  81%|████████  | 47/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=32 avail_mem=70.88 GB):  81%|████████  | 47/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  81%|████████  | 47/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=8 avail_mem=70.86 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s] Capturing num tokens (num_tokens=8 avail_mem=70.86 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 39.24it/s]


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
    Generated text:  Anna. I have a twin sister, Victoria. We both love animals very much. We both like to go to the zoo and visit the petting zoo. We both like to see the animals' faces. We also love to ride in the hot air balloon at the zoo. On vacation we all have a special trip. On Christmas Day, we go to the zoo and take the lion for a ride in the hot air balloon. On New Year's Day, we go to the zoo and take the lion for a ride in the hot air balloon. We always ride with Victoria. Then, we ride with Anna when we go to the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He or she has a very important job, but you can't get into the White House unless you are very rich. The president is the head of the government. They are also the leaders of many states in the country. 
    
    The president also has a lot of jobs. He or she may be in charge of a military, a police force, or even of the army. He or she also has other jobs, like being the person who takes care of the people's lives and making sure that everyone has the right to vote. 
    
    The president is like a king or queen, but he or
    ===============================
    Prompt: The capital of France is
    Generated text:  located in what valley?
    The answer to this question is: the Loire Valley. The capital of France, Paris, is situated in the Loire Valley, a valley of the River Loire. The Loire Valley is known for its rich history, beautiful landscapes, and historical sites, making it a popular destination for tourists and locals alike. The river plays a crucial role in the economy of the region, supporting industries such as agriculture, wine production, and tourism. The Loire Valley is also home to the famous Château de Chambord, a masterpiece of French Baroque architecture, which attracts thousands of visitors each year.
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the individuals and businesses who create and deploy it. The following three cases show how they are used in different industries and how they are changing.
    
    AI is everywhere.
    
    This means that AI is everywhere, and the future of AI is here, and it will have a powerful impact on all industries.
    
    At a recent conference, my colleague said that in the coming years, the AI industry will have a tremendous impact on how we work. But if we don’t start now, we may have a hard time keeping up with this change. If we don’t change our minds, we won’t be able to adapt to the new AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character, such as "funny, witty, and always up for a good laugh"]. I enjoy [insert a short description of your character's interests, such as "reading, cooking, or playing sports"]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity, such as "playing
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its cuisine, fashion, and art, and is a major tourist destination. The city is home to many important institutions such as the French Academy of Sciences and the Louvre Museum. It is also home to many cultural and artistic institutions, including the Musée d'Orsay and the Musée Rodin. Paris is a city of contrasts, with its modern
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from and adapt to new situations.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations and the responsible use of AI. This could lead to more stringent
    


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
    Generated text:  [Name], and I am a [Career] at [Company]. I'm a [Reason for Success] person who thrives on [Reason for Success]. I have a passion for [Reason for Success] and I'm always looking to learn new things and expand my knowledge. I enjoy [Reason for Success] and I'm always eager to try new things. I'm a [Reason for Success] person who is always up for a challenge and I'm always looking for opportunities to grow and improve myself. I'm a [Reason for Success] person who is always looking for ways to make a difference in the world and I'm always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. 
    
    Please let me know if you would like to see more information about it, such as when it is open, where it is located, or any other interesting facts about the city. Also, can you provide the capital of Spain? The capital of Spain is Madrid, known for its beautiful architecture, vibrant nightlife, and rich cultural heritage. 
    
    Please let me know if you would like to see more information about it, such as when it is open, where it is located, or any other interesting facts about the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a variety of trends and developments, including:
    
    1. Greater Personalization: As AI becomes more sophisticated, it will be able to learn and adapt to individual preferences, providing more personalized experiences for users.
    
    2. Improved Natural Language Processing: With the development of deep learning, AI will become even more adept at understanding and generating human language, making it possible for machines to understand and respond to human emotions and speech patterns.
    
    3. Autonomous Vehicles: AI will continue to evolve, leading to the development of self-driving vehicles, which could revolutionize transportation and delivery services.
    
    4. Predictive Analytics: AI will become even more capable of


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

     am

     a

     [

    Your

     Profession

    ]

     with

     [

    Your

     Years

     of

     Experience

    ]

     years

     of

     experience

     in

     [

    Your

     Role

    ].

     I

     have

     a

     passion

     for

     [

    Your

     Hobby

    /

    Interest

    ]

     and

     I

     enjoy

     [

    Your

     Area

     of

     Expert

    ise

    /

    Interest

    ].

     I

     am

     a

     dedicated

     [

    Your

     Leadership

     Qual

    ification

    ]

     and

     I

     thrive

     on

     [

    Your

     Way

     of

     Working

    /

    Goals

    ].

     I

     am

     a

     [

    Your

     Character

     Trait

    ]

     and

     I

     strive

     to

     always

     do

     [

    Your

     Task

    ].

     I

     am

     [

    Your

     Personality

    ]

     and

     I

     am

     always

     ready

     to

     learn

     and

     grow

    .

     [

    Your

     Name

    ]

     is

     dedicated

    ,

     driven

    ,

     and

     passionate

     about

     what

     they

     do

    .

     They

     take

     pride

     in

     their

     work

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     known

     for

     its

     iconic

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     rich

     history

     and

     diverse

     population

    .

     
    


    Paris

     is

     often

     referred

     to

     as

     "

    the

     city

     of

     a

     thousand

     sights

    "

     and

     is

     considered

     one

     of

     the

     most

     famous

     cities

     in

     the

     world

    .

     It

     is

     home

     to

     over

     

    2

    0

     million

     people

     and

     is

     the

     economic

    ,

     cultural

    ,

     and

     political

     center

     of

     France

    .

     Paris

     is

     also

     known

     for

     its

     wine

     industry

    ,

     fashion

     industry

    ,

     and

     art

     scene

    ,

     which

     attract

     visitors

     from

     around

     the

     world

    .

     
    


    Paris

     is

     also

     famous

     for

     its

     romantic

     architecture

    ,

     such

     as

     the

     Lou

    vre

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     exciting

     and

     will

     likely

     continue

     to

     evolve

     in

     unique

     ways

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

     AI

     in

     the

     years

     ahead

    :
    


    1

    .

     Machine

     learning

     will

     become

     more

     sophisticated

    :

     Machine

     learning

     is

     already

     making

     significant

     strides

     in

     improving

     the

     accuracy

     and

     efficiency

     of

     AI

     applications

    .

     As

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

     even

     more

     sophisticated

     algorithms

     that

     can

     learn

     from

     data

     and

     adapt

     to

     new

     situations

    .
    


    2

    .

     AI

     will

     be

     more

     integrated

     into

     everyday

     life

    :

     AI

     is

     already

     integrated

     into

     many

     aspects

     of

     our

     daily

     lives

    ,

     from

     self

    -driving

     cars

     to

     virtual

     assistants

     like

     Siri

     or

     Alexa

    .

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     routines

    ,

     we

     can

     expect

    



```python
llm.shutdown()
```
