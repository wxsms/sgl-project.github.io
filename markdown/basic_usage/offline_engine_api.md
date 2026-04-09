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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.64it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.63it/s]


    2026-04-09 03:32:51,922 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 03:32:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.11it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.11it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.11it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.11it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.11it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.11it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.11it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.11it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.25it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.25it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.25it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.25it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.25it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.25it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.25it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 23.78it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 23.78it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 23.78it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 23.78it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 23.78it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 23.78it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 23.78it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 34.68it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 34.68it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 34.68it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 34.68it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 34.68it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 34.68it/s]

    Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 34.68it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 38.16it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 46.70it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 46.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.81it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.48it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=131.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  31%|███       | 18/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  31%|███       | 18/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  31%|███       | 18/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  31%|███       | 18/58 [00:00<00:01, 35.85it/s]

    Capturing num tokens (num_tokens=960 avail_mem=131.58 GB):  31%|███       | 18/58 [00:00<00:01, 35.85it/s] Capturing num tokens (num_tokens=960 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.12it/s]Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.12it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.12it/s]Capturing num tokens (num_tokens=768 avail_mem=131.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.12it/s]Capturing num tokens (num_tokens=704 avail_mem=131.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.12it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.12it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.19it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.19it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.19it/s]Capturing num tokens (num_tokens=480 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.19it/s]

    Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.19it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.19it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=224 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.30it/s]

    Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.42it/s] Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  81%|████████  | 47/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  81%|████████  | 47/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=64 avail_mem=131.51 GB):  81%|████████  | 47/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=48 avail_mem=131.51 GB):  81%|████████  | 47/58 [00:01<00:00, 43.96it/s]

    Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  81%|████████  | 47/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=28 avail_mem=131.50 GB):  81%|████████  | 47/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=28 avail_mem=131.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.17it/s] Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.87it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 39.34it/s]


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
    Generated text:  Alice. I’m 25 years old and I love to travel. I have been to many countries, and my travels have enriched my life. When I travel, I try to experience different cultures, and I always have a question that I want to ask to the people I meet. Here are some questions that I have been asking myself. 1. What is your favorite place to visit? 2. What is your favorite food to eat? 3. What is your favorite type of music to listen to? 4. What is your favorite kind of drink? 5. What is your favorite type of book? 
    ===============================
    Prompt: The president of the United States is
    Generated text:  a(n) ______. A. president B. leader C. head of state D. prime minister C. head of state
    
    The president of the United States is a(n) head of state. The president is the head of government and is the commander-in-chief of the armed forces, but they do not serve as a leader in the traditional sense. Instead, they are the head of the executive branch of the government and are the highest ranking official in the United States. The other options are not correct as they do not accurately describe the function of the president in the United States government.
    
    Answer: C. head of state
    
    Mr.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is a bustling city and the largest in Europe. In the city of Paris, the main tourist attractions include the Eiffel Tower, the Louvre, and the Champs-Élysées. The Paris metro system also connects these attractions, linking the city center to the outer suburbs.
    
    Now, imagine you are a student in the 10th grade and you want to travel to Paris, France. You decide to start your journey at the main metro station, which is located in the heart of the city. You plan to use the metro to travel to the Eiffel Tower and then to the Louvre,
    ===============================
    Prompt: The future of AI is
    Generated text:  changing as the company takes the lead in revolutionizing healthcare with its latest tech.
    
    AI in healthcare can be defined as a complex process where machines can perform tasks that require human intelligence. In the realm of healthcare, AI is being used in a wide variety of fields, including drug discovery, personalized treatment, and diagnosis. Here are some of the key ways in which AI is revolutionizing the healthcare industry.
    
    AI in Drug Discovery: AI is being used to predict drug candidates, identify potential targets, and even simulate drug interactions. This helps researchers to identify the right candidates for clinical trials, reducing the time and cost associated with the process. AI algorithms


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your job or experience here]. How can I help you today? I look forward to meeting you! [Name] [Company name] [Job title] [Company website or LinkedIn profile] [Phone number] [Email address] [Company address] [Company logo or company name] [Company mission or values] [Company values] [Company culture] [Company policies] [Company policies] [Company policies]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and vibrant French culture. 
    
    This statement encapsulates the key facts about Paris, including its historical significance, notable landmarks, and cultural importance. It provides a brief overview of the city's significance in French society and its global reputation. 
    
    For a more detailed and comprehensive answer, consider including additional information such as the city's population, economic importance, or notable events that have taken place there. 
    
    For example, you could add that Paris is the largest city in France by area, with a population of over 2.1 million people. It is also home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we can expect to see more automation and artificial intelligence in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for innovation and creativity.
    
    2. AI ethics and privacy: As AI technology becomes more advanced, there will be a need to address ethical and privacy concerns. This could lead to new regulations and standards for AI development and use, as
    


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
    Generated text:  [insert character's name], and I'm [insert character's age]. I'm a [insert a profession] and I've been here since [insert date or time]. I have a passion for [insert something that interests you, like food, travel, music, or fitness]. I enjoy spending time with my family and friends and have a deep sense of [insert something that describes your personality, like being independent, adventurous, or friendly]. I have [insert a skill or hobby] that I enjoy, and I value [insert something that describes your character trait, like honesty, kindness, or perseverance]. What's something you're proud
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks, rich history, and diverse cultural scene.
    
    What is the capital city of Italy? Rome is the capital city of Italy. It is known for its ancient ruins, stunning Romanesque architecture, and rich history. Rome is also famous for its colorful streets, vibrant markets, and delicious cuisine. The city has a rich culture and is home to many famous landmarks, including the Colosseum and the Vatican City. Rome is also home to many world-renowned museums and art galleries.
    
    How many languages are spoken in Paris? Paris is a major city in France where many languages are spoken, including French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  diverse and rapidly evolving, with significant trends shaping its development. Here are some possible future trends in AI:
    
    1. Integration of AI into all aspects of daily life: One of the most promising areas for AI in the future is the integration of AI into everyday life. This could include applications such as smart cities, self-driving cars, and personalized virtual assistants.
    
    2. Increased use of AI in healthcare: AI is being used to improve healthcare outcomes, such as by analyzing medical images, diagnosing diseases, and predicting patient outcomes.
    
    3. AI in manufacturing: AI is being used to automate production processes, improve quality control, and predict equipment failures


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

    insert

     character

    's

     name

    ],

     and

     I

    'm

     a

     [

    insert

     character

    's

     profession

     or

     role

    ].

     I

     specialize

     in

     [

    insert

     a

     specific

     area

     or

     skill

     that

     you

     excel

     in

    ].

     I

     enjoy

     [

    insert

     a

     hobby

    ,

     activity

    ,

     or

     interest

     that

     you

    're

     passionate

     about

    ].

     I

    'm

     excited

     to

     learn

     about

     you

    ,

     and

     I

    'm

     looking

     forward

     to

     our

     conversation

    .

     Can

    't

     wait

     to

     meet

     you

    !

     [

    insert

     your

     name

     and

     how

     you

     met

     the

     other

     person

    ].

     [

    insert

     your

     experience

    ,

     any

     awards

    ,

     or

     any

     other

     relevant

     information

     that

     you

     can

     provide

     about

     yourself

    ].

     I

     look

     forward

     to

     meeting

     you

    ,

     and

     I

    'm

     ready

     to

     learn

     more

     about

     you

    .

     I

    'm

     [

    insert

     your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    .

     It

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     is

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

     Paris

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

    ,

     including

     the

     Roman

    ,

     Gothic

    ,

     and

     Renaissance

     er

    as

    ,

     and

     remains

     a

     city

     of

     contrasts

    ,

     cultural

     diversity

    ,

     and

     artistic

     innovation

    .

     The

     city

     is

     home

     to

     over

     a

     million

     people

    ,

     is

     home

     to

     many

     of

     the

     world

    ’s

     most

     famous

     landmarks

    ,

     and

     is

     an

     important

     hub

     for

     business

     and

     commerce

    ,

     as

     well

     as

     culture

     and

     entertainment

    .

     Paris

     is

     known

     for

     its

     art

    ,

     fashion

    ,

     food

    ,

     and

     wine

    ,

     and

     is

     home

     to

     numerous

     museums

    ,

     galleries

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     growing

     convergence

     of

     technology

     across

     different

     sectors

    ,

     as

     well

     as

     an

     increasing

     emphasis

     on

     ethical

     considerations

     and

     social

     impact

    .

     Some

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     various

     sectors

    :

     AI

     will

     continue

     to

     become

     more

     integrated

     into

     various

     industries

    ,

     including

     healthcare

    ,

     finance

    ,

     transportation

    ,

     manufacturing

    ,

     and

     retail

    .

     As

     AI

     becomes

     more

     prevalent

    ,

     we

     may

     see

     increased

     integration

     of

     AI

     into

     existing

     systems

     and

     workflows

    ,

     leading

     to

     more

     efficient

     and

     effective

     operations

    .
    


    2

    .

     Improved

     AI

     safety

     and

     ethical

     considerations

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     there

     will

     be

     increasing

     concerns

     about

     its

     safety

     and

     ethical

     implications

    .

     AI

     developers

     and

     users

     will

    



```python
llm.shutdown()
```
