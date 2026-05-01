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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.76it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.76it/s]


    2026-05-01 23:44:19,700 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-01 23:44:19] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.53it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.53it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.53it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.53it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.53it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.53it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.53it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.53it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.53it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.53it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.41it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.41it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.41it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.41it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.41it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.41it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.41it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.41it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.41it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.41it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s] 

    Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.59it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.59it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.59it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.59it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.59it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.59it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.59it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.59it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.59it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.59it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 17.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 17.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 17.37it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:03, 17.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:03, 17.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   7%|▋         | 4/58 [00:00<00:03, 17.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   7%|▋         | 4/58 [00:00<00:03, 17.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.16it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.39it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.31it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.31it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.31it/s] Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.82it/s]

    Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.41it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.41it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.41it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.41it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.41it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.41it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.72it/s]

    Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.52it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.52it/s]

    Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=16 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.40it/s]

    Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.40it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 35.57it/s]


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
    Generated text:  Cj, I am a 22 year old male with a History of a Chronic Illness. Please type out the most effective words that can be used to create a personalized website about my illness.
    Creating a personalized website for someone with a chronic illness requires a detailed understanding of their medical condition, lifestyle, and treatment history. Here are some key elements to consider when creating an effective website:
    
    1. **Personalized Information**:
       - Include a clear, concise biography or personal statement detailing the person's life story, diagnosis, and how it has affected their life.
       - Provide contact information, including a phone number and a website
    ===============================
    Prompt: The president of the United States is
    Generated text:  expected to make a speech at the end of the day, and a poll of 1,000 adults conducted a week after the speech revealed that 585 thought the speech was good. Among the adults who were upset with the speech, 44% said the speech was very good. What is the sample size of the poll? To determine the sample size of the poll, we need to identify the number of adults who were polled and the percentage of those polled who were upset with the speech. The sample size is the total number of adults surveyed, and the percentage of those polled who were upset with the speech is
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Which of the following cities in Europe is the capital of France? A. New Delhi B. Stockholm C. Berlin D. London
    Answer:
    
    C
    
    In the experiment of boiling a liquid to observe its boiling point, the temperature at which the liquid begins to boil is ____
    A. The temperature at which water boils
    B. The temperature at which a liquid begins to boil
    C. The temperature at which the liquid turns into a gas
    D. The temperature at which the liquid begins to evaporate
    Answer:
    
    B
    
    When 1 mol of CH4 is fully burned to produce CO2 and H2O,
    ===============================
    Prompt: The future of AI is
    Generated text:  bleak - or is it?
    
    In the years ahead, whether we see the worst of it is up to you. That’s what’s at stake in the latest auction of intellectual property rights at the Edinburgh International AI Exhibition, held in Edinburgh, Scotland, in early April.
    
    The auction will see two companies that are currently under pressure for patents granted by the European Commission, leading to a flurry of requests to the Intellectual Property Office for extensions, and re-granting patents, including a few that are not even in the public domain.
    
    The patent auction is part of a series of events that are set to come to an end in the year 


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other cultural institutions. Paris is a popular tourist destination and is known for its rich history, art, and cuisine. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. The city is known for its vibrant nightlife and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the impact of AI on society.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in even more areas, including diagnosis
    


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
    Generated text:  [Name] and I'm a [occupation] with [number] years of experience in [specific field]. I've always been passionate about [reason why I enjoy my work]. And I love [reason why I love my job]. If you ever need any help or advice, just reach out! [Name] [Phone Number] [Email Address] [Social Media Links] [LinkedIn Profile]
    [Name] is a [occupation] with [number] years of experience in [specific field]. In her career, she's always been passionate about [reason why she enjoys her work]. She's also passionate about [reason why she loves
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is also known as the City of Light. It is the largest and most populous city in France, and it is known for its rich history, art, architecture, and cuisine. The city is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris also has a diverse and multicultural population, with many different nationalities and cultures living there. The city is an important center for trade and diplomacy, and it is home to many of the world's top universities and institutions of higher learning. Paris is known for its lively nightlife, arts and entertainment scene,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid and significant advancements in many different areas, including:
    
    1. AI in healthcare: AI will play an increasingly important role in the healthcare industry, as AI is being used to improve patient outcomes, reduce costs, and increase efficiency.
    
    2. AI in transportation: AI is becoming increasingly important in transportation, with vehicles equipped with sensors and AI-powered navigation systems that help to reduce congestion and improve safety.
    
    3. AI in entertainment: AI is already being used in entertainment, with voice-activated assistants like Siri and Alexa that assist with tasks like scheduling appointments, controlling the TV, and playing music.
    
    4. AI in manufacturing


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

    /an

     [

    Occup

    ation

    ].

     I

    'm

     currently

     [

    Age

    ]

     years

     old

    .

     My

     favorite

     hobby

     is

     [

    Favorite

     Hobby

    ].

     I

    'm

     an

     [

    Occup

    ation

    ]

     who

     enjoys

     [

    Favorite

     Activity

    ].

     I

     love

     [

    Favorite

     Cause

    /

    Event

    ].

     How

     do

     you

     like

     [

    Favorite

     People

    /

    Places

    ]?

     I

    'm

     [

    Favorite

     Person

    /

    Place

    ].

     I

    'm

     a

    /an

     [

    Occup

    ation

    ].

     I

    'm

     [

    Age

    ]

     years

     old

    .

     My

     favorite

     hobby

     is

     [

    Favorite

     Hobby

    ].

     I

    'm

     an

     [

    Occup

    ation

    ]

     who

     enjoys

     [

    Favorite

     Activity

    ].

     I

     love

     [

    Favorite

     Cause

    /

    Event

    ].

     How

     do

     you

     like

     [

    Favorite

     People

    /

    Places

    ]?

     I

    'm

     [

    Favorite

     Person

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     lively

     cultural

     scene

    .

     Paris

     is

     also

     home

     to

     many

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Visitors

     can

     explore

     the

     city

    's

     unique

     neighborhoods

    ,

     such

     as

     the

     Mont

    mart

    re

     and

     Mar

    ais

     areas

    ,

     and

     take

     in

     the

     city

    's

     rich

     cultural

     offerings

    .

     Paris

     is

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     French

     culture

    ,

     history

    ,

     and

     cuisine

    .

     Its

     elegant

     architecture

    ,

     charming

     streets

    ,

     and

     welcoming

     atmosphere

     make

     it

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     there

     are

     many

     potential

     areas

     where

     it

     is

     expected

     to

     evolve

    .

     Some

     possible

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

     with

     other

     technologies

    :

     AI

     is

     becoming

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

    ,

     data

     analytics

    ,

     and

     machine

     learning

    ,

     to

     improve

     efficiency

    ,

     accuracy

    ,

     and

     safety

     in

     various

     applications

    .
    


    2

    .

     AI

    -driven

     healthcare

    :

     AI

     is

     being

     used

     in

     the

     healthcare

     industry

     to

     improve

     diagnosis

    ,

     treatment

    ,

     and

     patient

     care

    .

     For

     example

    ,

     AI

    -powered

     imaging

     systems

     can

     analyze

     medical

     images

     with

     greater

     accuracy

     and

     speed

     than

     human

     radi

    ologists

    .
    


    3

    .

     AI

    -powered

     autonomous

     vehicles

    :

     AI

     is

     becoming

     more

     widely

     used

     in

     autonomous

     vehicles

     to

     improve

     safety

    



```python
llm.shutdown()
```
