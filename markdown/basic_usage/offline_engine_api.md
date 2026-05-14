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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.21it/s]


    2026-05-14 01:42:25,720 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 01:42:25] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  7.94it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  7.94it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  7.94it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  7.94it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  7.94it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  7.94it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  7.94it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  7.94it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  7.94it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 12.60it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 12.60it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 12.60it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 12.60it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 12.60it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 12.60it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 12.60it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 12.60it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 12.60it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 18.31it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 18.31it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 25.87it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 25.87it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 25.87it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 25.87it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 25.87it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 25.87it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 25.87it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 25.87it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 25.87it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 25.87it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 25.87it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 35.32it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 35.32it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:06<00:00, 35.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 14.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 14.87it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 14.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.34 GB):   7%|▋         | 4/58 [00:00<00:03, 14.96it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   7%|▋         | 4/58 [00:00<00:03, 14.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):   7%|▋         | 4/58 [00:00<00:03, 14.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):  10%|█         | 6/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):  10%|█         | 6/58 [00:00<00:03, 15.57it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):  10%|█         | 6/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):  10%|█         | 6/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  21%|██        | 12/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:02, 21.71it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  21%|██        | 12/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  21%|██        | 12/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.02it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.04it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.04it/s]Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.04it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.04it/s]Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:01<00:01, 21.97it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:01<00:01, 21.97it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:01<00:01, 21.97it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  40%|███▉      | 23/58 [00:01<00:01, 21.97it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.86it/s]

    Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.57it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.57it/s]Capturing num tokens (num_tokens=416 avail_mem=72.25 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.57it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.57it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.57it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.79it/s]

    Capturing num tokens (num_tokens=256 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=208 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=192 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=144 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.11it/s]

    Capturing num tokens (num_tokens=128 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.11it/s] Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  81%|████████  | 47/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=80 avail_mem=72.20 GB):  81%|████████  | 47/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  81%|████████  | 47/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=48 avail_mem=72.19 GB):  81%|████████  | 47/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=32 avail_mem=72.19 GB):  81%|████████  | 47/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  81%|████████  | 47/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.29it/s]Capturing num tokens (num_tokens=24 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.29it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.29it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.29it/s]Capturing num tokens (num_tokens=12 avail_mem=72.17 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.29it/s]Capturing num tokens (num_tokens=8 avail_mem=72.17 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.29it/s] Capturing num tokens (num_tokens=8 avail_mem=72.17 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 29.14it/s]


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
    Generated text:  Yana. I'm from the United States. I've been studying abroad for a while now. I've been living in Sweden for the last five years. My friends and I have been on an educational tour in Paris. We've been to the Louvre Museum, the Notre Dame Cathedral, and so many other places. Paris has a lot of things to see and do. I think that I've learned so much. How do I go about learning to speak other languages? I have some basic knowledge of English, but I really don't understand how to use it very much. I can't speak English well, but I think that
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide between two schools for his children. School A offers a tuition of $600 per year with no fees for the first 15 years, and a $100 per year fee for each additional year. School B offers a tuition of $700 per year with a $50 per year fee for each additional year. The president wants to know which school will be better in the long run. To determine which school is better in the long run, we need to compare the tuition costs for each school over a period of many years, considering the student's lifetime expenses. The president should choose the school
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A: Paris B: London C: New York D: Sydney C: New York
    
    The capital of France is Paris. Paris is the largest city in France and also the capital of France. Other options are not capitals of France: London is the capital of the United Kingdom, New York is the capital of the United States, and Sydney is the capital of Australia. Therefore, the correct answer is A: Paris. 
    
    To verify this understanding:
    
    1. Paris is the largest city in France.
    2. It serves as the capital of France.
    3. Other options are not capitals of France.
    
    So, the final answer is
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the AI people. 
    
    All 700,000 professionals in the AI industry are passionate about this technology. They are passionate about the potential to have a profound impact in society, and the opportunity to make a difference in the lives of others. But they also know that the industry will have its own problems, risks, and challenges in the coming years.
    
    When asked about the future of AI, many of those same individuals feel strongly that it will continue to advance. They want to see it reach new heights and bring benefits to society. But they also understand that, to achieve that, the technology will


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a few details about your personality or background]. I enjoy [insert a few details about your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the European Union. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also known for its rich history, art, and culture, and is a popular tourist destination. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. Its status as the capital of France has made it a major economic and political center, and it continues to be an important cultural and artistic center in the world. Paris is also known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into our daily lives, from manufacturing to customer service. Automation will likely become more prevalent as AI becomes more advanced and efficient.
    
    2. AI ethics: As AI becomes more integrated into our daily lives, there will likely be a growing concern about its ethical implications. This will likely lead to a greater focus on AI ethics and the development of new ethical guidelines and standards.
    
    3. AI for healthcare: AI is already
    


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
    Generated text:  [Name] and I am a [fill in your profession] from [location] with [number of years of experience]. I am a passionate [reason why you are passionate about your career] and I am always looking for new and exciting opportunities to grow and learn. In my free time, I enjoy spending time with my friends, reading books, and playing outdoor games like fishing or hiking. I am always seeking out ways to improve myself and expand my horizons. I am confident in my abilities and I am excited to share my knowledge with others. How can I be a good resource for you? I'm really looking to expand my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris, the historical and cultural heart of the country, is the most populous city in France and the seat of the French government, capital of the country. It is the second largest city in the European Union, with an estimated population of over 2. 3 million residents. The city is known for its rich history, art, fashion, cuisine, and gastronomy, as well as its iconic landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. Paris is also a significant center for business, finance, education, and the arts, and has been
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of possibilities and potential, but it's not clear what exactly these possibilities will be. However, some of the most likely trends that experts are predicting include:
    
      1. Increased use of AI in healthcare: AI is already being used to diagnose and treat a variety of medical conditions, and experts are optimistic that it will continue to become more advanced and accessible in the future.
      2. Advancements in robotics and automation: AI is already being used to automate a variety of tasks, from manufacturing to customer service, and experts are excited about the potential for even greater automation in the future.
      3. Increased focus on ethical


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

     Alex

     and

     I

    'm

     a

     friendly

     and

     knowledgeable

     writer

     who

     loves

     to

     read

    ,

     learn

    ,

     and

     travel

    .

     I

    'm

     confident

     and

     always

     eager

     to

     share

     my

     ideas

     with

     anyone

     who

     will

     listen

    .

     I

     enjoy

     writing

     short

     stories

     and

     essays

    ,

     and

     I

    'm

     always

     eager

     to

     give

     my

     thoughts

     on

     important

     issues

    .

     I

     believe

     in

     using

     language

     to

     express

     ideas

     clearly

     and

     vivid

    ly

    ,

     and

     I

    'm

     always

     willing

     to

     learn

     from

     my

     readers

    .

     I

     hope

     you

     can

     meet

     me

     at

     any

     moment

    !

     Can

     you

     elaborate

     on

     your

     hobbies

     and

     interests

     outside

     of

     writing

    ?


    Certainly

    !

     Besides

     writing

    ,

     I

     enjoy

     reading

     and

     learning

     new

     things

    .

     I

    'm

     also

     an

     avid

     reader

     and

     love

     to

     spend

     time

     exploring

     new

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     northern

     bank

     of

     the

     Se

    ine

     river

    ,

     overlooking

     the

     beautiful

     Lou

    vre

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

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

    ,

     and

     is

     home

     to

     a

     diverse

     population

     of

     about

     

    2

     million

     people

    .

     France

    's

     cultural

    ,

     intellectual

    ,

     and

     economic

     center

    ,

     Paris

     is

     also

     an

     important

     hub

     for

     diplomacy

    ,

     trade

    ,

     and

     tourism

    .

     The

     city

     is

     a

     major

     financial

     center

     and

     is

     a

     significant

     player

     in

     the

     world

    's

     financial

     markets

    .

     The

     French

     government

     has

     long

     been

     known

     for

     its

     commitment

     to

     education

     and

     cultural

     enrichment

    ,

     and

     Paris

     is

     a

     well

    -known

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     rapidly

     evolving

    ,

     with

     several

     trends

     shaping

     its

     direction

     and

     impact

     on

     society

    .
    


    One

     potential

     trend

     is

     the

     increase

     in

     the

     development

     and

     deployment

     of

     AI

     in

     different

     sectors

    ,

     such

     as

     healthcare

    ,

     transportation

    ,

     finance

    ,

     and

     manufacturing

    .

     As

     AI

     becomes

     more

     integrated

     into

     these

     industries

    ,

     it

     is

     expected

     to

     drive

     significant

     improvements

     in

     efficiency

    ,

     productivity

    ,

     and

     accuracy

    .
    


    Another

     trend

     is

     the

     increasing

     reliance

     on

     AI

     for

     decision

    -making

     in

     all

     areas

     of

     life

    .

     While

     AI

     can

     augment

     human

     decision

    -making

    ,

     it

     is

     unlikely

     to

     completely

     replace

     humans

    ,

     who

     are

     highly

     skilled

     in

     complex

     problem

    -solving

     and

     decision

    -making

    .
    


    In

     the

     next

     few

     years

    ,

     AI

     is

     expected

     to

     continue

     to

     advance

     in

    



```python
llm.shutdown()
```
