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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:26,  1.99it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:26,  1.99it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:26,  1.99it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:26,  1.99it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:26,  1.99it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:11,  4.05it/s]

    Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:05,  8.03it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:05,  8.03it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:05,  8.03it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:05,  8.03it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:05<00:05,  8.03it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:05<00:05,  8.03it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:05<00:05,  8.03it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:05<00:05,  8.03it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 13.70it/s]

    Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:02, 13.70it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 20.12it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 28.57it/s]

    Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 28.57it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 28.57it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 38.16it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 38.16it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 38.16it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 38.16it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 38.16it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 38.16it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 38.16it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 38.16it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 38.16it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 38.16it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 38.16it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 49.89it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 49.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.38 GB):   2%|▏         | 1/58 [00:00<00:06,  8.82it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.16 GB):   2%|▏         | 1/58 [00:00<00:06,  8.82it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:06,  8.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.37 GB):   3%|▎         | 2/58 [00:00<00:06,  8.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.36 GB):   3%|▎         | 2/58 [00:00<00:06,  8.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.36 GB):   7%|▋         | 4/58 [00:00<00:04, 11.14it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.36 GB):   7%|▋         | 4/58 [00:00<00:04, 11.14it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.35 GB):   7%|▋         | 4/58 [00:00<00:04, 11.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.35 GB):  10%|█         | 6/58 [00:00<00:03, 13.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.33 GB):  10%|█         | 6/58 [00:00<00:03, 13.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.33 GB):  10%|█         | 6/58 [00:00<00:03, 13.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.32 GB):  10%|█         | 6/58 [00:00<00:03, 13.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.32 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.16it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.32 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.16it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.31 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.30 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.30 GB):  21%|██        | 12/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.30 GB):  21%|██        | 12/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.29 GB):  21%|██        | 12/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.28 GB):  21%|██        | 12/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.28 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.28 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.79it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.27 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.79it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.25 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.26 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.24 GB):  33%|███▎      | 19/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=960 avail_mem=74.25 GB):  33%|███▎      | 19/58 [00:01<00:01, 26.79it/s] Capturing num tokens (num_tokens=896 avail_mem=74.24 GB):  33%|███▎      | 19/58 [00:01<00:01, 26.79it/s]Capturing num tokens (num_tokens=832 avail_mem=74.23 GB):  33%|███▎      | 19/58 [00:01<00:01, 26.79it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.15it/s]Capturing num tokens (num_tokens=768 avail_mem=74.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.15it/s]Capturing num tokens (num_tokens=704 avail_mem=74.21 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.15it/s]Capturing num tokens (num_tokens=640 avail_mem=74.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.15it/s]Capturing num tokens (num_tokens=576 avail_mem=74.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.15it/s]Capturing num tokens (num_tokens=512 avail_mem=74.20 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.15it/s]Capturing num tokens (num_tokens=512 avail_mem=74.20 GB):  50%|█████     | 29/58 [00:01<00:00, 34.52it/s]Capturing num tokens (num_tokens=480 avail_mem=74.22 GB):  50%|█████     | 29/58 [00:01<00:00, 34.52it/s]Capturing num tokens (num_tokens=448 avail_mem=74.21 GB):  50%|█████     | 29/58 [00:01<00:00, 34.52it/s]Capturing num tokens (num_tokens=416 avail_mem=74.20 GB):  50%|█████     | 29/58 [00:01<00:00, 34.52it/s]Capturing num tokens (num_tokens=384 avail_mem=74.20 GB):  50%|█████     | 29/58 [00:01<00:00, 34.52it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.19 GB):  50%|█████     | 29/58 [00:01<00:00, 34.52it/s]Capturing num tokens (num_tokens=352 avail_mem=74.19 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.93it/s]Capturing num tokens (num_tokens=320 avail_mem=74.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.93it/s]Capturing num tokens (num_tokens=288 avail_mem=74.17 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.93it/s]Capturing num tokens (num_tokens=256 avail_mem=74.17 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.93it/s]Capturing num tokens (num_tokens=240 avail_mem=74.16 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.93it/s]Capturing num tokens (num_tokens=224 avail_mem=74.16 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.93it/s]Capturing num tokens (num_tokens=224 avail_mem=74.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.24it/s]Capturing num tokens (num_tokens=208 avail_mem=74.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.24it/s]Capturing num tokens (num_tokens=192 avail_mem=74.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.24it/s]Capturing num tokens (num_tokens=176 avail_mem=74.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.24it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.24it/s]Capturing num tokens (num_tokens=144 avail_mem=74.13 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.24it/s]Capturing num tokens (num_tokens=144 avail_mem=74.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=128 avail_mem=74.12 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=112 avail_mem=74.12 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=96 avail_mem=74.11 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.03it/s] Capturing num tokens (num_tokens=80 avail_mem=74.10 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=80 avail_mem=74.10 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=64 avail_mem=74.09 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=48 avail_mem=74.09 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=32 avail_mem=74.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.18it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=24 avail_mem=74.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=24 avail_mem=74.07 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.84it/s]Capturing num tokens (num_tokens=20 avail_mem=74.06 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.84it/s]Capturing num tokens (num_tokens=16 avail_mem=74.06 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.84it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.84it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.84it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.84it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 41.48it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 30.71it/s]


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
    Generated text:  Liv and I am a Visual Art student at the University of Maryland College Park. I have always loved using different mediums to express myself. At the moment, I am interested in making my own comics and doing a book. I also enjoy learning about art history.
    I have always loved reading and writing. I like to write about interesting things and people that have a lot of meaning. I love that people can read my stories, and even if they don't, I have a feeling that they could understand.
    I have a lot of knowledge about art history and I enjoy exploring new things through art. I am really excited to learn about different mediums
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person, not an animal. Which of the following statements is correct about the president of the United States being an animal? Options: A) He or she is a plant. B) He or she is a reptile. C) He or she is a mammal. D) He or she is a fish. E) He or she is a bird.
    
    To determine which statement is correct about the president of the United States being an animal, let's analyze the information given in the prompt:
    
    1. The president of the United States is a person.
    2. The president of the United States is not an animal.
    
    Since the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, where the Eiffel Tower is located. Which country has France as its capital?
    A. Switzerland
    B. Germany
    C. Japan
    D. Italy
    
    To determine which country has France as its capital, let's first consider the capital cities of all the countries mentioned in the options.
    
    1. **Switzerland**: The capital of Switzerland is Berne.
    2. **Germany**: The capital of Germany is Berlin.
    3. **Japan**: The capital of Japan is Tokyo.
    4. **Italy**: The capital of Italy is Rome.
    
    From the above list, we can see that none of these cities is located in France
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, with the potential to solve problems and improve people’s lives. However, the barriers to AI adoption are high, particularly in developing countries. This is where governments and other organizations can make a big impact. Here are 4 ways governments and other organizations can improve the adoption of AI in developing countries:
    1. Provide funding and infrastructure
    Governments and other organizations can provide funding and infrastructure to help build the necessary skills and resources to support the development and use of AI. This could include funding for training programs, infrastructure such as cloud computing and data storage, and access to resources for researchers and developers.
    2. Establish regulations and guidelines
    


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


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its vibrant arts scene and culinary delights. Paris is a popular tourist destination and a cultural hub for France and the world. It is also home to many important institutions such as the French Academy of Sciences and the Paris Opera. The city is known for its fashion industry, with many famous designers and boutiques. Overall, Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some possible future trends in AI include:
    
    1. Increased use of AI in healthcare: AI is already being used to improve patient care, from personalized treatment plans to automated diagnostic tools. As AI technology continues to advance, we can expect to see even more sophisticated and accurate AI systems being used in healthcare to improve patient outcomes and reduce costs.
    
    2. Increased use of AI
    


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
    Generated text:  [insert character name] and I'm a [insert character's age, gender, or personality traits]. I'm the [insert character's job or occupation]. I enjoy [insert activities, interests, or hobbies]. What's your name, and what's your profession or role in the world? I'm excited to meet you! Let's get to know each other better! [Your name here]. (Type in your real name if you wish to continue this conversation.) [Your name here]. (Type in your real name if you wish to continue this conversation.) [Your name here]. (Type in your real name if you wish to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, with many possibilities for advancements in various areas. Here are some possible trends in AI:
    
    1. Natural Language Processing (NLP): AI systems that can understand and generate human-like language, including human speech, text, and images. This could lead to more advanced language translation, virtual assistants, and chatbots.
    
    2. Deep Learning: The development of neural networks with more layers and complexity, which can learn from large datasets and generalize to new data. This could revolutionize fields such as image recognition, natural language processing, and autonomous systems.
    
    3. Autonomous Systems: AI systems that can operate without human intervention, such as drones


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

     and

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ]

     [

    Occup

    ation

     or

     Character

    ]

     who

     is

     passionate

     about

     [

    Your

     Passion

     or

     Interest

    ].

     I

    'm

     a

     [

    Your

     Inter

    ests

     or

     Skills

    ]

     person

     who

     is

     always

     [

    Your

     Job

     or

     Skill

    ]

     and

     strive

     to

     achieve

     my

     goals

    .

     I

    'm

     always

     [

    Your

     Personality

     or

     Character

    ]

     and

     am

     always

     [

    Your

     Mot

    ivation

     or

     Goal

    ].

     I

     enjoy

     [

    Your

     Hobby

     or

     Activity

    ]

     and

     I

    'm

     always

     [

    Your

     Ded

    ication

     or

     Commit

    ment

    ].

     I

    'm

     [

    Your

     Personality

     or

     Character

    ]

     and

     [

    Your

     Inter

    ests

     or

     Skills

    ]

     person

     who

     is

     always

     [

    Your

     Mot

    ivation

     or

     Goal

    ].

     I

    'm

     [

    Your

    
    
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

     second

     largest

     in

     the

     European

     Union

    .

     It

     is

     known

     as

     "

    The

     City

     of

     Love

    "

     and

     is

     home

     to

     many

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     Lou

    vre

     Museum

    ,

     and

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     is

     also

     known

     for

     its

     diverse

     cuisine

     and

     music

     scene

    ,

     with

     numerous

     festivals

     and

     events

     throughout

     the

     year

    .

     The

     city

     has

     a

     rich

     cultural

     history

     dating

     back

     over

     

    2

    ,

    0

    0

    0

     years

     and

     continues

     to

     be

     an

     important

     cultural

     and

     economic

     center

     in

     Europe

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     with

     new

     technologies

     and

     applications

     constantly

     emerging

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

     Use

     of

     AI

     in

     Healthcare

    :

     With

     the

     increasing

     use

     of

     AI

     in

     medicine

    ,

     the

     development

     of

     new

     AI

     algorithms

     and

     techniques

     will

     be

     crucial

     to

     improve

     diagnosis

     and

     treatment

     of

     diseases

    .

     AI

    -powered

     diagnostic

     tools

    ,

     such

     as

     chat

    bots

     and

     virtual

     assistants

    ,

     will

     help

     doctors

     make

     better

     decisions

     and

     provide

     better

     care

     to

     patients

    .
    


    2

    .

     Integration

     of

     AI

     with

     Other

     Technologies

    :

     AI

     is

     becoming

     more

     integrated

     into

     other

     technologies

     such

     as

     smart

     homes

    ,

     self

    -driving

     cars

    ,

     and

     augmented

     reality

    .

     This

     integration

     will

     enhance

     the

     usability

     of

     these

     technologies

     and

     improve

     their

     efficiency

    .
    


    3

    .

    



```python
llm.shutdown()
```
