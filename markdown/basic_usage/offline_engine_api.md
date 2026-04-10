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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.53it/s]


    2026-04-10 18:53:56,309 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 18:53:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:44,  2.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:44,  2.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:44,  2.89s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:44,  2.89s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:31,  1.74it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:31,  1.74it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.37it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.37it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.37it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.37it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.37it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.37it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.37it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.37it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.37it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.43it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.43it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.43it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.43it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.43it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.43it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.43it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.43it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.33it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.33it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.33it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.33it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.33it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.33it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.33it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.33it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 23.86it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 23.86it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 23.86it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 23.86it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 23.86it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 23.86it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 23.86it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.87it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.30it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.30it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.30it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.30it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.30it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.30it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.30it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.74it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.74it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.74it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.74it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.74it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.74it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.74it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.74it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.01it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.55it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=131.64 GB):   7%|▋         | 4/58 [00:00<00:03, 17.05it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   7%|▋         | 4/58 [00:00<00:03, 17.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):   7%|▋         | 4/58 [00:00<00:03, 17.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.64 GB):   7%|▋         | 4/58 [00:00<00:03, 17.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.64 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.73it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.61 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.02it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s]Capturing num tokens (num_tokens=960 avail_mem=131.59 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s] Capturing num tokens (num_tokens=896 avail_mem=131.59 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s]Capturing num tokens (num_tokens=768 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s]Capturing num tokens (num_tokens=704 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s]Capturing num tokens (num_tokens=704 avail_mem=131.58 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.27it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.27it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.27it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.27it/s]

    Capturing num tokens (num_tokens=480 avail_mem=131.58 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.27it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.27it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=224 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]

    Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=160 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.84it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.84it/s] Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.84it/s]

    Capturing num tokens (num_tokens=64 avail_mem=131.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.84it/s]Capturing num tokens (num_tokens=48 avail_mem=131.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.84it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.84it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=28 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.52it/s] Capturing num tokens (num_tokens=4 avail_mem=131.48 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.52it/s]

    Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 38.57it/s]


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
    Generated text:  Xiao Ming, and my ID number is 2003101028. How would you recommend I improve my English proficiency? As an AI language model, I would suggest the following:
    
    1. Read widely: It's important to read in English and practice reading comprehension. You can find many good resources online, such as BBC Learn English, Oxford Dictionary, and BBC Learning English.
    
    2. Use a language learning app: There are many language learning apps available, such as Duolingo, Memrise, and Babbel. These apps can be very helpful in improving your English proficiency.
    
    3. Listen to English:
    ===============================
    Prompt: The president of the United States is
    Generated text:  attending a conference where he is visiting several cities. He is in city A for 10 hours, city B for 12 hours, and city C for 15 hours. After his trip, he is scheduled to visit city D, which is 4 hours away from city C and 3 hours away from city B. If the president's meeting with each city takes 1 hour, how long will the total duration of his visit be?
    To determine the total duration of the president's visit, we need to calculate the total time he spends in each city and then sum these times.
    
    1. The president spends 1
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the city of the Eiffel Tower, and the food here is excellent. The food is very good and I have tried several restaurants. For a nice lunch it is really affordable. I have been there a couple of times in the past and have had a great time. I have to say that I do not think my taste is anything like the food here. I really like the French people and the food, but I believe the food in France is better. That is one of the main differences between France and the United States. It is not surprising that French food has a reputation for being so delicious. The reputation for French
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s accelerating with a new wave of AI-Driven initiatives. These initiatives are leading the way in the world of AI development, with a focus on improving the efficiency and accuracy of AI systems.
    What is AI?
    AI is a subset of computer science that focuses on designing machines that can perform tasks that normally require human intelligence. These tasks include tasks like recognizing faces, making decisions, understanding speech, and playing games.
    AI is being used in a variety of applications, from self-driving cars and robotics to natural language processing and medical diagnostics.
    AI has the potential to revolutionize industries and improve the quality of life. However, it


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am a [job title] at [company name], and I have been with the company for [number of years] years. I am passionate about [job title] and I am always looking for ways to [job title] my skills and abilities. I am a [job title] at [company name], and I have been with the company for [number of years] years. I am passionate about [job title] and I am always looking for ways to [job title] my skills and abilities. I am a [job title] at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major financial center and a major tourist destination, with many famous museums, theaters, and restaurants. The city is home to many important institutions of higher education, including the University of Paris and the Paris School of Design. Paris is a city of contrasts, with its modern architecture and cultural heritage blending with its historic charm. It is a city that has played a significant role in French history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the potential trends that are likely to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there is a growing recognition of the need to consider the ethical implications of its use. This will likely lead to increased focus on ethical considerations, including issues such as bias, transparency, and accountability.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As the technology continues to advance
    


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
    Generated text:  [Your Name], and I'm a [Your profession], [Your job title] at [Your company name]. I'm currently the [Your current position], and I am a [Your position title], [Your full title] at [Your company name], where I have been working for [Your length of service]. I am a big fan of [Your hobbies, interests, or activities], and I am always looking for ways to grow my skills and knowledge. I'm also a strong communicator, and I enjoy writing and speaking in front of groups, helping people understand complex topics. I'm a member of the [Your club or group
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, and it is the largest and most populous city in the country. It is known for its medieval architecture, museums, and annual celebrations, such as the Eiffel Tower and the world-renowned annual "Carnival of Lights" festival. Paris is also known for its culinary traditions, including famous dishes such as boudin and croissants. The city's historical significance dates back to the Roman Empire and continues to be a cultural and commercial hub in Europe. As of 2021, Paris had a population of around 2.8 million people. Paris is the capital of France and is the largest city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several key trends that are emerging in different ways around the world. Here are some of the most promising trends in AI that are likely to shape the future:
    
    1. Increased Use of AI for Advancing Science and Medicine: AI can be used to enhance medical diagnosis, drug discovery, and personalized medicine. AI can also help researchers develop new treatments and therapies.
    
    2. Increased Use of AI for Autonomous Vehicles: AI is being used to develop autonomous vehicles that can navigate the roads safely and efficiently. This will have a significant impact on the transportation industry and will improve public safety.
    
    3. Increased Use of AI for Financial Services


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

     am

     a

     [

    position

    ]

     at

     [

    organization

    ].

     I

     have

     a

     background

     in

     [

    field

    ],

     and

     I

     am

     passionate

     about

     [

    interest

    s

    ].

     I

     am

     here

     to

     [

    explain

     why

     you

    'd

     be

     a

     good

     fit

     for

     the

     position

    ],

     and

     I

    'm

     excited

     to

     help

     you

     achieve

     your

     goals

    .

     And

     if

     you

     want

     to

     hear

     about

     my

     work

    ,

     I

    'll

     let

     you

     know

     once

     I

    'm

     done

    .

     [

    Name

    ]

     is

     a

     dedicated

    ,

     passionate

    ,

     and

     experienced

     team

     player

     who

     thr

    ives

     in

     a

     fast

    -paced

    ,

     collaborative

     environment

    .

     I

    'm

     a

     strong

     communicator

    ,

     detail

    -oriented

    ,

     and

     have

     a

     passion

     for

     helping

     others

     succeed

    .

     Whether

     you

    're

     looking

     to

     grow

     your

    
    
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

     Europe

     and

     one

     of

     the

     most

     important

     cities

     in

     the

     world

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     thriving

     cultural

     scene

    .

     It

     is

     home

     to

     many

     famous

     landmarks

     and

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     E

    iff

    el

     Tower

    .

     Paris is

     a

     popular

     tourist

     destination

     and

     the

     center

     of

     many

     important

     cultural

     and

     political

     events

    .

     The

     city

     is

     also

     home

     to

     many

     famous

     French

     composers

     and

     writers

    ,

     as

     well

     as

     the

     headquarters

     of

     many

     major

     French

     companies

     and

     organizations

    .

     The

     city

     is

     also

     known

     for

     its

     vibrant

     nightlife

    ,

     with

     many

     excellent

     bars

     and

     clubs

    .

     Overall

    ,

     Paris

     is

     a

     fascinating

     and

     diverse

     city

     that

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     dynamic

    ,

     and

     it

     is

     impossible

     to

     predict

     what

     the

     next

     generation

     of

     AI

     will

     look

     like

    .

     However

    ,

     here

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

     considerations

    :

     As

     concerns

     about

     AI

    's

     potential

     to

     be

     used

     in

     ways

     that

     harm

     individuals

     and

     society

     grow

    ,

     there

     will

     be

     greater

     emphasis

     on

     ethical

     considerations

     and

     accountability

     for

     AI

     development

     and

     deployment

    .
    


    2

    .

     Deep

     learning

     models

     will

     become

     more

     sophisticated

    :

     With

     the

     increasing

     availability

     of

     large

     datasets

     and

     powerful

     computing

     power

    ,

     deep

     learning

     models

     will

     become

     more

     sophisticated

     and

     capable

     of

     handling

     increasingly

     complex

     tasks

    .
    


    3

    .

     Increased

     focus

     on

     mini

    atur

    ization

    :

     As

     AI

     becomes

     more

     complex

    ,

     it

     will

     require

     smaller

    



```python
llm.shutdown()
```
