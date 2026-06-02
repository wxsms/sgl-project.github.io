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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.18it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.07it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.07it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.07it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.07it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.07it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.07it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.07it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.07it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:05<00:02, 13.53it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 20.70it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 20.70it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 20.70it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 20.70it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 20.70it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 20.70it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 20.70it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 20.70it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 20.70it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 20.70it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:05<00:01, 20.70it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 41.51it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.00it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:03, 16.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:03, 16.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:03, 16.94it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.21 GB):   7%|▋         | 4/58 [00:00<00:02, 18.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.21 GB):   7%|▋         | 4/58 [00:00<00:02, 18.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.20 GB):   7%|▋         | 4/58 [00:00<00:02, 18.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):   7%|▋         | 4/58 [00:00<00:02, 18.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.19 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.19 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.36it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.18 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.18 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.18 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.90it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.68it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.13 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.68it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.14 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.68it/s] Capturing num tokens (num_tokens=960 avail_mem=74.14 GB):  38%|███▊      | 22/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  38%|███▊      | 22/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  43%|████▎     | 25/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  43%|████▎     | 25/58 [00:00<00:01, 27.32it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.32it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.32it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.34it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.34it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.34it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.34it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:01<00:01, 25.57it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:01<00:01, 25.57it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:01<00:01, 25.57it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  53%|█████▎    | 31/58 [00:01<00:01, 25.57it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.16it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.16it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.16it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.16it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.16it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  71%|███████   | 41/58 [00:01<00:00, 25.69it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  71%|███████   | 41/58 [00:01<00:00, 25.69it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 25.69it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 25.69it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 25.69it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.87it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.87it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.87it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.87it/s]Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.05it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.05it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.05it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.05it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.05it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 28.35it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 28.35it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.35it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.35it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.35it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.35it/s] Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.29it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.29it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:02<00:00, 27.17it/s]


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
    Generated text:  Ralph. I'm a writer, I write fiction. I'd like to take a short trip to the United States and travel around, seeing all of the beautiful places.
    I'm in my 40s now, but when I'm younger, I was 17. I moved to the USA from New York City a few years ago to stay away from the life I was used to. I live in New York City now, and live with a family I've been with for about a year. I have a family of three children. I have been writing since I was 20 years old. I have been an editor
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the government. The term of office for the president is 4 years. The current president of the United States is Donald Trump. He took office in January 2017 and will be retiring in January 2021. During which year of his presidential career did he take office? To determine during which year of his presidential career Donald Trump took office, we need to understand the term of office for the president. The term of office for the president is 4 years. Therefore, we need to find the year when Trump took office within the range of 2017 to 2021
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is located at the mouth of the Seine River. Is this statement true or false?
    
    To determine whether the statement "Paris is located at the mouth of the Seine River" is true or false, we need to analyze the geographical location of Paris.
    
    1. **Identify the key points:**
       - Paris is the capital of France.
       - Paris is located in the western part of France, specifically in the region of western Paris.
       - The Seine River is a major waterway in the Paris region.
       - The mouth of the Seine River is a specific location where the river enters the ocean
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be a very interesting, and unpredictable one. From the advances in computer vision to the development of neural networks, these are the three main advancements of the field. In this article, we will discuss the future of AI and how it will impact the world. We will also discuss the potential consequences of AI on society, including job loss and privacy concerns. Finally, we will provide some suggestions for how we can use AI to create a safer and more equitable society.
    The future of AI will be a very exciting and rapidly evolving field. It will continue to transform our lives and impact how we interact with technology. As the field advances,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? As an AI language model, I'm here to assist you with any questions or tasks you may have. How can I help you today? What's your background, education, and experience? I'm always here to help you with any questions you may have. What's your favorite hobby or activity? As an AI language model, I'm here to assist you with any questions or tasks you may have. How can I help you today? What's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a cultural and historical center with a rich history dating back to ancient times. Paris is a major transportation hub, with the iconic Eiffel Tower serving as a symbol of the city's importance in global affairs. It is also a popular tourist destination, attracting millions of visitors each year. The city is known for its cuisine, fashion, and art, and is home to many renowned museums and galleries. Paris is a vibrant and dynamic city, with a diverse population and a rich cultural heritage. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased automation: AI is already being used to automate a wide range of tasks, from manufacturing to customer service. As technology continues to advance, we can expect to see even more automation in areas such as healthcare, finance, and transportation.
    
    2. Enhanced cognitive abilities: AI is likely to continue to improve its ability to process and analyze large amounts of data, which will enable it to make more accurate and nuanced decisions. This could lead to breakthroughs in fields such as medicine, engineering, and finance.
    
    3. Improved privacy and security: As AI becomes more integrated
    


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
    Generated text:  [Name] and I'm a [Career] with [Experience] years of experience. I have a passion for [Interest or hobby], and I'm a [Occupation] who believes in [Value Proposition]. I'm constantly learning and evolving as a result, and I'm looking forward to continuing to make a positive impact on the world through my work. What can I say about myself? [Tell about your personal qualities, strengths, and goals]. [Tell about your career journey and any notable achievements or experiences]. [Tell about your current projects, any challenges, and the impact you're making]. [Tell about any challenges you've
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its rich history, stunning architecture, and vibrant cultural scene. The city is home to the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, among other notable landmarks. Paris also has a long-standing tradition of art and culture, with museums like the Musée de l'Orangerie and the Centre Pompidou, and festivals like the Musée du Nord and Festival de la Chanson. With its strategic location on the River Seine, Paris is a cultural hub with a rich heritage of Parisian culture, history, and art. According to the United Nations, Paris is the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be driven by four key trends:
    
    1. AI will become more prevalent in the everyday lives of people. The more people are aware of AI's capabilities, the more they will use it. Artificial intelligence will become more integrated into our everyday lives, making it easier for us to interact with machines and systems.
    
    2. AI will become more collaborative and complementary to humans. AI will increasingly assist humans in their work and decision-making processes, helping them to accomplish tasks more efficiently and effectively. This will create a more collaborative and harmonious relationship between humans and machines.
    
    3. AI will become more ethical and transparent. As the development of AI


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

     I

    'm

     a

     [

    Age

    ]

     year

     old

     female

    .

     I

    'm

     an

     artist

    .

     I

     love

     to

     paint

    ,

     draw

    ,

     and

     sculpt

    .

     I

    've

     been

     painting

     since

     I

     was

     a

     child

     and

     sculpt

    ing

     for

     a

     long

     time

    .

     I

     love

     to

     travel

    ,

     travel

     to

     different

     parts

     of

     the

     world

    ,

     experience

     different

     cultures

    ,

     and

     learn

     from

     them

    .

     I

    'm

     always

     on

     the

     lookout

     for

     new

     ideas

    ,

     fresh

     perspectives

     and

     new

     things

     to

     see

     and

     explore

    .

     I

     love

     to

     think

     outside

     of

     the

     box

    ,

     to

     try

     new

     things

    ,

     and

     always

     strive

     for

     excellence

    .

     I

     believe

     in

     using

     my

     creativity

     to

     make

     the

     world

     a

     better

     place

    .

     Thanks

     for

     asking

    ,

     I

    'll

     be

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     capital

     of

     France

    .

     It

     is

     also

     home

     to

     some

     of

     the

     world

    's

     most

     famous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Notre-D

    ame

     Cathedral

    ,

     Lou

    vre

     Museum

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     Paris

     is

     known

     for

     its

     fashion

     industry

    ,

     cuisine

    ,

     and

     music

     scene

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     economic

     hub

    .

     The

     city

     is

     also

     home

     to

     many

     influential

     French

     artists

    ,

     writers

    ,

     and

     scholars

    .

     Paris

     is

     a

     vibrant

     and

     dynamic

     city

     that

     continues

     to

     inspire

     and

     capt

    ivate

     people

     around

     the

     world

    .

     It

     is

     also

     known

     for

     its

     unique

     cultural

     and

     artistic

     traditions

    ,

     including

     the

     Renaissance

    
    
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

     areas

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     robotics

    ,

     and

     quantum

     computing

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     we

     can

     expect

     to

     see

     a

     continued

     focus

     on

     improving

     healthcare

     delivery

     through

     the

     use

     of

     AI

    .

     This

     could

     include

     developing

     more

     effective

     diagnostic

     tools

    ,

     predicting

     patient

     outcomes

    ,

     and

     optimizing

     healthcare

     delivery

    .
    


    2

    .

     Autonomous

     vehicles

    :

     With

     the

     development

     of

     more

     advanced

     AI

     technology

    ,

     autonomous

     vehicles

     are

     likely

     to

     become

     a

     more

     common

     feature

     of

     our

     daily

     lives

    .

     These

     vehicles

     could

     be

     designed

     to

     follow

     traffic

     rules

    ,

     navigate

     roads

    ,

     and

     make

    



```python
llm.shutdown()
```
