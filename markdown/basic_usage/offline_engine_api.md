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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.38it/s]


    2026-05-20 09:21:34,864 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 09:21:34] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.80it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.80it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.80it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.80it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.80it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.80it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.80it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.80it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.80it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 21.22it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 21.22it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 21.22it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 21.22it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 21.22it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 21.22it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 21.22it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 21.22it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 21.22it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 21.22it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 29.56it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 29.56it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 29.56it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 29.56it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 29.56it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 39.63it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 39.63it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 39.63it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.06it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.06it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.69 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.06it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.06it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.08it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.08it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.08it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.08it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.08it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.08it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.91it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.91it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.91it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.91it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.91it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.91it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=288 avail_mem=76.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.96it/s]

    Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=224 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.57it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=160 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.93it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=64 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  81%|████████  | 47/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.91it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.91it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=4 avail_mem=76.60 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=4 avail_mem=76.60 GB): 100%|██████████| 58/58 [00:01<00:00, 38.53it/s]


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
    Generated text:  Yen and I am a journalist. I have been a journalist for over 10 years. My passion for journalism is to provide an honest, transparent, and factual reporting to the public.
    I love to travel and travel for the love of travel. I like to travel to different parts of the world and do so for several reasons. One is the opportunity to meet people from different cultures. I love to meet people from different cultures and learn about their ways of life and their stories. Secondly, I find it incredibly interesting and exciting to document the cultural differences between people.
    I like to travel to different parts of the world because it is
    ===============================
    Prompt: The president of the United States is
    Generated text:  30 years older than the president of Central America. The president of Central America is half the age of the president of Asia. If the president of Asia is 30 years old, how old is the president of Central America? To determine the age of the president of Central America, we need to follow the given information step by step.
    
    1. Identify the age of the president of Asia.
       The president of Asia is given as 30 years old.
    
    2. Determine the age of the president of Central America.
       According to the problem, the president of Central America is half the age of the president of Asia.
    
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris, and the largest city in Europe. It is a highly cultural, historic, and tourist city, and one of the world's most popular cities, with a population of about 2. 4 million as of 2017. The capital has a population of 785,000 and 60% of the city is in the Paris Region. The main universities of the city are the University of Paris 12 and the Paris Institute of Technology. The capital of France is known for its significant cultural and historical importance, and is home to many famous museums and art galleries, such
    ===============================
    Prompt: The future of AI is
    Generated text:  here and it’s not going to be like it looks like now. If you were to ask me what the future of AI will look like, the most likely answer would be what it will be like when you read this. If you’re one of the AI-driven company executives, you would have to have a different view. It is going to be the future of AI is this: it will be the future of AI is based on the understanding that AI is an emergent phenomenon. This means that it is evolving and changing continuously. It is going to be a multi-disciplinary field where multiple knowledge domains are fused together. As for how


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What do you enjoy
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and the Renaissance. Paris is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to many other notable landmarks and attractions, including the Champs-Élysées, the Eiffel
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives through smart home devices, self-driving cars, and virtual assistants. As AI technology continues to advance, we can expect to see even more integration into our daily lives, such as in healthcare, finance, and transportation.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical and responsible AI. This will involve developing AI that is
    


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
    Generated text:  Sarah and I am a 25-year-old freelance graphic designer who specializes in creating visual identities for brands. I have a talent for creating designs that are both creative and effective. I enjoy working with a variety of clients and always strive to make my clients' projects a success. Thank you for considering me for the position! What other information would you like to know about Sarah? Of course! Please let me know if you have any additional questions or if there are any other areas you would like me to provide more information about Sarah. I am looking forward to the opportunity to meet with you! 
    Answer in complete sentences. I'm thrilled
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the heart of the country's cultural and economic life. Known for its Notre-Dame Cathedral and Eiffel Tower, it is a popular tourist destination and a cultural center that hosts the world-renowned Paris Opera. Paris is also home to numerous museums, including the Louvre, which houses the world's largest collection of art and artifacts.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright with many possibilities and potential applications. Here are some of the trends and advancements we might see in the AI field in the next few years:
    
    1. AI will become more advanced: As technology continues to advance, so will AI. We may see advancements in areas such as natural language processing, computer vision, and robotics, leading to more advanced AI systems that can perform complex tasks with greater accuracy and speed.
    
    2. AI will become more accessible: With the increasing use of AI in everyday life, we may see a greater focus on making AI more accessible. This could mean making AI systems more accessible to people with disabilities, or making them


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

     a

     [

    role

    ]

     at

     [

    Company

    ].

     What

     can

     you

     tell

     me

     about

     yourself

     and

     your

     expertise

     in

     the

     field

    ?


    [

    Name

    ]:

     Hi

    ,

     my

     name

     is

     [

    Name

    ].

     I

    'm

     a

     [

    role

    ]

     at

     [

    Company

    ].

     I

     have

     a

     deep

     understanding

     of

     [

    relevant

     field

    ]

     and

     have

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     [

    relevant

     field

    ].

     I

     thrive

     on

     learning

     new

     things

     and

     am

     always

     looking

     for

     opportunities

     to

     grow

     my

     skills

    .

     What

     can

     you

     tell

     me

     about

     yourself

     and

     your

     expertise

     in

     the

     field

    ?


    I

     am

     a

     [

    role

    ]

     at

     [

    Company

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

     [

    relevant

     field

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     as

     the

     city

     of

     light

     and

     is

     home

     to

     many

     of

     France

    's

     major

     cultural

     institutions

     and

     landmarks

    .

     The

     city

    's

     skyline

     is

     a

     testament

     to

     its

     status

     as

     a

     world

    -class

     met

    ropolis

    ,

     with

     towering

     buildings

     and

     a

     vibrant

     atmosphere

    .

     Paris

     is

     a

     major

     hub

     of

     commerce

    ,

     finance

    ,

     and

     education

    ,

     attracting

     a

     large

     number

     of

     visitors

     each

     year

    .

     The

     city

     is

     also

     home

     to

     a

     diverse

     population

     of

     French

     people

    ,

     who

     are

     known

     for

     their

     traditional

     clothing

    ,

     music

    ,

     and

     cuisine

    .

     France

    's

     capital

     city

     has

     played

     an

     important

     role

     in

     shaping

     the

     country

    's

     cultural

     identity

     and

     has

     been

     a

     major

     hub

     of

     international

     trade

     and

     diplomacy

     for

     centuries

    .

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     with

     many

     possibilities

     and

     applications

     in

     various

     fields

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

     focus

     on

     ethical

     AI

    :

     As

     concerns

     about

     AI

    's

     impact

     on

     society

     become

     more

     prevalent

    ,

     there

     will

     be

     increasing

     pressure

     to

     make

     sure

     that

     AI

     is

     developed

     eth

    ically

     and

     responsibly

    .

     This

     could

     mean

     implementing

     stricter

     ethical

     guidelines

     and

     standards

     for

     AI

    ,

     including

     rules

     around

     transparency

    ,

     accountability

    ,

     and

     fairness

    .
    


    2

    .

     AI

    -powered

     healthcare

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     play

     a

     more

     significant

     role

     in

     healthcare

    ,

     from

     diagn

    osing

     diseases

     to

     developing

     personalized

     treatment

     plans

    .

     This

     could

     lead

     to

     more

     precise

     diagnosis

     and

     more

     effective

     drug

     development

    ,

     among

     other

     benefits

    .
    


    



```python
llm.shutdown()
```
