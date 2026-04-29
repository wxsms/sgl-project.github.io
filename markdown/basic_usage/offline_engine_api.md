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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.02it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.01it/s]


    2026-04-29 15:06:09,413 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 15:06:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.23it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.51it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.51it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.25it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.25it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.25it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.25it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.25it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.25it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.25it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.25it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.25it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.25it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.25it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.13it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.13it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.13it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.13it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.13it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.13it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.13it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.92it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.92it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.92it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.92it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.92it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.55it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.55it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.55it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.55it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.66it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.66it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.66it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.66it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.66it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.66it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  50%|█████     | 29/58 [00:00<00:00, 43.14it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 43.14it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 43.14it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 43.14it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 43.14it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 43.14it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.38it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.38it/s]Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.38it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.75it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.75it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.75it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.75it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.75it/s]Capturing num tokens (num_tokens=48 avail_mem=74.54 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.75it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=32 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=28 avail_mem=74.53 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=24 avail_mem=74.53 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=20 avail_mem=74.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=16 avail_mem=74.50 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=16 avail_mem=74.50 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.98it/s]Capturing num tokens (num_tokens=12 avail_mem=73.93 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.98it/s]Capturing num tokens (num_tokens=8 avail_mem=73.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.98it/s] Capturing num tokens (num_tokens=4 avail_mem=73.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.98it/s]

    Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 38.84it/s]


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
    Generated text:  Alan and I am a PhD student studying phonetics at the University of Birmingham. My area of interest is the phonology of English and particularly the phonological structure of the verb.
    
    I have been doing some research about the phonological structure of the verb in English in collaboration with the University of Birmingham with Professor Andrew Marlow. We have been using a big corpus of English corpora to explore the phonological structure of verbs, taking into account features such as aspect, tense, and aspect, aspect markers, and negative markers, among others.
    
    We have found that there are a number of features which are important for the phonology of the verb
    ===============================
    Prompt: The president of the United States is
    Generated text:  getting a new electric vehicle. If he drives to the beach, he will have to pay a $20 maintenance fee and an additional $10 service fee for each trip. If he drives to the city, he will have to pay a $30 maintenance fee and an additional $15 service fee for each trip. Calculate his total maintenance and service fees for 60 trips to the beach and 45 trips to the city. To calculate the total maintenance and service fees for 60 trips to the beach and 45 trips to the city, we can break down the costs for each type of trip and then
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and it is also the largest city of the country. France is a constitutional monarchy. The President of the French Republic is the head of state, and the President is elected by the members of the National Assembly. The president takes up his or her seat in the National Assembly on the first day of the month of January and leaves it on the last day of the month of December. If the last day of the month of December falls on a Wednesday, the President is inaugurated on the first day of the month of January. How many times will the President of the French Republic take his seat in the National Assembly in a year?
    ===============================
    Prompt: The future of AI is
    Generated text:  still uncertain. However, in the next few years, AI will become more and more ubiquitous, with the potential to help us solve some of the world's biggest problems. As it goes, companies will be scrambling to develop and deploy AI systems, while governments will be tasked with overseeing the use of AI in their nation's public and private sectors.
    The development of AI is a complex and rapidly evolving field, with new technologies constantly emerging and being refined. As such, it is important for individuals to stay up-to-date on the latest developments and developments in the field. This will allow them to make informed decisions about the use of AI in their


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [country]. I have a [job title] at [company name], and I'm always looking for ways to improve my skills and knowledge. I'm always eager to learn and to help others. What do you do for a living? I'm a [job title] at [company name]. I'm always looking for ways to improve my skills and knowledge. I'm always eager to learn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the annual Eiffel Tower Festival. It is also the seat of the French government and the country's cultural and political capital. Paris is a major tourist destination and a popular tourist destination, with many famous landmarks and museums. The city is also known for its rich history and cultural heritage, including the Louvre Museum and the Notre-Dame Cathedral. Paris is a vibrant and diverse city with a rich cultural scene, and is a popular destination for tourists and locals alike. The city is also home to many important institutions, including the French Academy of Sciences and the French National Library
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be a need to address privacy and security concerns. This may involve developing new algorithms and techniques to protect user data and prevent unauthorized access.
    
    3. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs.
    


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
    Generated text:  [Name] and I am a [年龄段/性别] with [number] years of experience. I have [number of years of experience in [field of expertise]]. I have always been a [character trait] person, and I have been very [character trait] in my career. I have always [verb to describe a trait, such as "had, " "had, " "had, " "had, " etc.]. I am passionate about [job title], and I love [occupation]. I have a love for [activity or hobby], and I often [verb to describe a behavior, such as "shares,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its iconic Eiffel Tower and is a bustling center of culture and daily life. 
    
    (25 words) France's capital city, Paris, is renowned for its iconic Eiffel Tower and a thriving culture that hosts daily life activities. It's a bustling metropolis that is a popular tourist destination. 
    
    The capital city is the seat of the French government and is home to numerous cultural institutions, including the Louvre Museum and the Notre-Dame Cathedral. It is also a major transportation hub, with the iconic Eiffel Tower serving as a symbol of France's international status. 
    
    The French capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be increasingly shaped by the rapid advancements in the technology sector. There are several trends that are expected to shape the landscape of AI in the coming years, including:
    
    1. Increased focus on ethical AI: As more and more people become concerned about the ethical implications of AI, there is a growing emphasis on developing AI systems that are not only effective but also transparent and accountable. This includes developing ethical guidelines for AI development, designing AI systems that can adapt to changing circumstances, and ensuring that AI systems are not used in ways that promote bias or discrimination.
    
    2. Integration of AI with other technologies: As AI becomes more integrated with other technologies


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

    ].

     I

    'm

     a

     [

    Age

    ]

     year

     old

    ,

     [

    Occup

    ation

    ]

     with

     [

    Degree

    ]

     from

     [

    University

     Name

    ].

     I

     enjoy

     [

    My

     Favorite

     Activity

    ],

     [

    My

     Favorite

     Book

    /

    Album

    /

    Video

    ],

     and

     [

    My

     Favorite

     Food

    ].

     What

    's

     your

     favorite

     thing

     about

     yourself

    ?
    


    [

    Name

    ]:

     As

     an

     AI

     language

     model

    ,

     I

     am

     not

     physically

     present

     to

     answer

     this

     question

     directly

    ,

     but

     I

     am

     programmed

     to

     respond

     to

     prompts

     and

     assist

     with

     tasks

    .

     So

    ,

     my

     favorite

     thing

     about

     myself

     is

     that

     I

     am

     always

     learning

     and

     growing

    ,

     and

     I

     strive

     to

     be

     a

     valuable

     asset

     to

     the

     community

    .

     I

     am

     also

     dedicated

     to

     providing

     the

     best

     assistance

     and

     support

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     and

     it

     is

     the

     largest

     city

     in

     Europe

     and

     one

     of

     the

     most

     populous

     cities

     in

     the

     world

    .

     It

     is

     the

     cultural

     and

     political

     center

     of

     France

     and

     is

     known

     for

     its

     iconic

     landmarks

    ,

     museums

    ,

     and

     food

     scene

    .

     Paris

     is

     home

     to

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     attractions

    .

     The

     city

     is

     also

     famous

     for

     its

     fashion

     industry

    ,

     art

    ,

     and

     music

     scenes

    ,

     attracting

     many

     visitors

     each

     year

    .

     Its

     rich

     history

     and

     diverse

     culture

     make

     Paris

     an

     important

     city

     for

     many

     people

     around

     the

     world

    .

     Paris

     is

     also

     known

     for

     its

     rich

     food

     scene

    ,

     including

     popular

     French

     cuisine

    ,

     such

     as

     cro

    iss

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     one

     of

     rapid

     development

     and

     widespread

     adoption

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     this

     future

    :
    


    1

    .

     Increased

     automation

     and

     robotics

    :

     AI

     will

     continue

     to

     become

     more

     sophisticated

    ,

     with

     more

     advanced

     algorithms

     and

     machine

     learning

     techniques

     being

     applied

     to

     automate

     routine

     tasks

     and

     processes

    .

     This

     could

     lead

     to

     widespread

     automation

     in

     industries

     such

     as

     manufacturing

    ,

     transportation

    ,

     and

     healthcare

    .
    


    2

    .

     Enhanced

     data

     analysis

    :

     AI

     will

     continue

     to

     become

     more

     sophisticated

    ,

     with

     the

     ability

     to

     analyze

     large

     amounts

     of

     data

     to

     extract

     insights

     and

     make

     predictions

    .

     This

     could

     lead

     to

     a

     more

     informed

     and

     data

    -driven

     society

    ,

     with

     better

     decision

    -making

     abilities

     and

     improved

     public

     services

    .
    


    3

    .

     Integration

     of

     AI

    



```python
llm.shutdown()
```
