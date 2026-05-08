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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.08it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.08it/s]


    2026-05-08 09:14:38,405 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 09:14:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.15it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.15it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.15it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.15it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.15it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.15it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.15it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.15it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.15it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.15it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:01, 19.91it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:01, 19.91it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:01, 19.91it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 27.70it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 37.61it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 37.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.62it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.62it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.62it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.62it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.49it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.49it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.49it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  21%|██        | 12/58 [00:00<00:01, 28.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 28.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 28.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 28.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 28.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.66 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.60it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.60it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.60it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.60it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.96it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.96it/s] Capturing num tokens (num_tokens=896 avail_mem=74.64 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.96it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.96it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.96it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.96it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.11it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.11it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.11it/s]Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.11it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.11it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.11it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.90it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.90it/s]Capturing num tokens (num_tokens=384 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.90it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.90it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.90it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.90it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  71%|███████   | 41/58 [00:01<00:00, 43.47it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 43.47it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 43.47it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 43.47it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 43.47it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  71%|███████   | 41/58 [00:01<00:00, 43.47it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.50it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.50it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.33it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.33it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 38.61it/s]


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
    Generated text:  Cameron and I'm a 17-year-old college student. I'm studying for my computer science course, and I'm really interested in the topic of privacy and encryption. I've been looking for some information about it, but I'm having trouble figuring out how to do so. Can you provide me with some information on privacy and encryption? And if possible, any tips on how to create a strong password? Also, could you provide me with some examples of how to use encryption to protect personal information on the internet? Lastly, could you give me some advice on how to use GitHub to contribute to open source projects?
    Certainly, I
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a country in need of relief. The president uses a special program that can grant funds to hospitals. If the program has 500 days until a new year and the president wants to ensure that the hospital receives 100% of the funding available to him, how much money must the president contribute if he receives 100 funds from the program per day? To determine how much money the president must contribute, we need to follow these steps:
    
    1. Calculate the total amount of funding available to the president.
    2. Determine how much funding the president needs to ensure 100% of the funding is available
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is located at 44°49'47" latitude and 2°48'37" longitude. The capital of Brazil is Brasilia, which is located at 22°30'27" latitude and 45°13'28" longitude. 
    
    If you travel from Paris to Brasilia, how many degrees of longitude will you travel? 
    
    This problem is similar to the one you encountered in the math content but with a different country, different latitude and longitude, and different units of measurement. It requires calculating the difference between the two locations in terms of a
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the people. We are just the future for the future. Is it true that AI will completely replace humans one day, and we cannot continue to pursue knowledge and improve ourselves? Please provide a rationale for your answer.
    
    The statement "The future of AI is in the hands of the people" implies that AI will continue to develop and evolve independently without human intervention. However, this statement is not entirely accurate. AI is not an entity that can simply "create" or "replace" humans, but rather it is the result of human effort and creativity. It is a technology that relies on human intelligence, algorithms, and data


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character]. I enjoy [insert a short description of your character's interests or hobbies]. I'm always looking for new experiences and challenges, and I'm always eager to learn and grow. What's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a popular tourist destination and a major economic center in France. The city is also known for its cuisine, fashion, and music. It is a cultural and political center of France and a major hub for international trade and diplomacy. The city is home to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This means that AI systems will be able to learn from and adapt to human behavior, making them more capable of understanding and responding to human emotions and motivations.
    
    2. Greater use of machine learning: Machine learning is expected to become more prevalent in AI, with more sophisticated algorithms and models being developed to improve the accuracy and efficiency of AI systems.
    
    3. Increased focus on ethical considerations
    


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
    Generated text:  [insert character name], and I'm an [insert occupation or title] who has always been passionate about [insert passion or hobby that you enjoy]. I'm always eager to learn and always willing to share my knowledge with others. In my free time, I enjoy playing sports, reading books, and spending time with my furry friend. I'm a very down-to-earth, friendly person who is always ready to help others. If you need help with anything, don't hesitate to ask me. I'm [insert a short, enthusiastic greeting or nickname]. [Insert character's signature]. 
    [Insert character's character description or summary here,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and one of the largest cities in Europe. It is a major economic, cultural, and political center. The city includes the Île de la Cité, an island in the Seine River. Paris is known for its architecture, cuisine, music, and fashion. It is also the cultural and artistic capital of France, hosting major events such as the Eiffel Tower parades and the World Cup. Paris is home to a large population and is one of the most visited cities in the world. The city is also home to a diverse population with a mix of French and immigrant communities.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and evolving, but several trends are likely to shape its direction:
    
    1. Increased automation: AI will continue to automate routine tasks, freeing up more human beings to focus on higher-level work. This could lead to a decrease in the need for humans in certain fields, such as healthcare, finance, and retail.
    
    2. Democratization: AI will continue to democratize access to technology, making it more accessible to people from all backgrounds. This will lead to a greater willingness to engage with AI technologies, and more people will become engaged in AI research and development.
    
    3. Ethical concerns: As AI becomes more prevalent, there will be


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

    Your

     Age

    ]

     year

     old

     [

    Your

     Occupation

    ].

     I

    'm

     always

     ready

     to

     learn

     and

     I

     enjoy

     sharing

     my

     knowledge

     with

     others

    .
    


    If

     you

     could

     say

     anything

     about

     yourself

    ,

     what

     would

     it

     be

    ?

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     my

     skills

    ,

     whether

     it

    's

     through

     education

    ,

     self

    -im

    pro

    vement

    ,

     or

     just

     enjoying

     life

    .

     I

    'm

     always

     willing

     to

     learn

     and

     grow

    ,

     no

     matter

     what

    .
    


    What

    's

     your

     favorite

     hobby

    ?

     I

     love

     spending

     time

     with

     my

     family

     and

     playing

     games

     like

     board

     games

    ,

     video

     games

    ,

     and

     board

     games

    .

     I

     also

     enjoy

     hiking

     and

     taking

     long

     walks

     in

     nature

    .
    


    Describe

     a

     time

     when

    
    
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

     France

     and

     one

     of

     the

     largest

     in

     Europe

    ,

     with

     a

     population

     of

     over

     

    2

    .

     

    5

     million

     people

    .

     Its

     most

     famous

     landmark

     is

     the

     E

    iff

    el

     Tower

    .

     It

     is

     also

     the

     seat

     of

     France

    ’s

     government

    ,

     government

     offices

    ,

     and

     many

     of

     its

     institutions

    .

     The

     city

     is

     known

     for

     its

     architecture

    ,

     art

    ,

     and

     culture

    .

     It

     is

     a

     major

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     Paris

     is

     also

     a

     cultural

     and

     artistic

     center

    ,

     known

     for

     its

     world

    -ren

    owned

     museums

    ,

     art

     galleries

    ,

     and

     festivals

    .

     It

     is

     home

     to

     many

     famous

     landmarks

    ,

     such

     as

     the

     Lou

    vre

    ,

     Notre

    -D

    ame

     Cathedral

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     driven

     by

     a

     combination

     of

     technological

     advancements

    ,

     changes

     in

     society

    ,

     and

     the

     increasing

     complexity

     of

     the

     tasks

     that

     AI

     systems

     will

     need

     to

     perform

    .
    


    One

     trend

     that

     is

     likely

     to

     continue

     is

     the

     integration

     of

     AI

     into

     various

     industries

    ,

     including

     healthcare

    ,

     finance

    ,

     and

     transportation

    ,

     to

     help

     improve

     efficiency

     and

     accuracy

     in

     these

     areas

    .

     AI

     will

     also

     be

     used

     to

     enhance

     the

     user

     experience

     by

     providing

     personalized

     recommendations

     and

     insights

    .
    


    Another

     trend

     is

     the

     increasing

     use

     of

     AI

     to

     automate

     routine

     tasks

    ,

     freeing

     up

     workers

     for

     more

     complex

     and

     creative

     work

    .

     This

     could

     result

     in

     increased

     productivity

     and

     reduced

     costs

    ,

     but

     it

     may

     also

     lead

     to

     job

     displacement

     for

     some

     workers

    .
    


    The

     pace

    



```python
llm.shutdown()
```
