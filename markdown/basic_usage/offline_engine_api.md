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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.81it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:24,  2.16it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:24,  2.16it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:24,  2.16it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:24,  2.16it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:24,  2.16it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:10,  4.39it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:04<00:04,  8.70it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:04<00:04,  8.70it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 14.78it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 14.78it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 14.78it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 14.78it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 14.78it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 14.78it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 14.78it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 14.78it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 14.78it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 22.70it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 22.70it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 22.70it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 22.70it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 22.70it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 22.70it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 22.70it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 22.70it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 22.70it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 22.70it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 32.61it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 32.61it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 32.61it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 32.61it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 32.61it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 32.61it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 42.05it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.74 GB):   3%|▎         | 2/58 [00:00<00:05, 11.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.73 GB):   3%|▎         | 2/58 [00:00<00:05, 11.15it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=54.73 GB):   3%|▎         | 2/58 [00:00<00:05, 11.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=54.73 GB):   7%|▋         | 4/58 [00:00<00:04, 12.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.73 GB):   7%|▋         | 4/58 [00:00<00:04, 12.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.71 GB):   7%|▋         | 4/58 [00:00<00:04, 12.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.71 GB):  10%|█         | 6/58 [00:00<00:03, 14.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.70 GB):  10%|█         | 6/58 [00:00<00:03, 14.42it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=54.67 GB):  10%|█         | 6/58 [00:00<00:03, 14.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.67 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.57 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.56 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.56 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.56 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.56 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.11it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=54.56 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.60 GB):  31%|███       | 18/58 [00:00<00:01, 25.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.58 GB):  31%|███       | 18/58 [00:00<00:01, 25.95it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=54.58 GB):  31%|███       | 18/58 [00:00<00:01, 25.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.58 GB):  31%|███       | 18/58 [00:00<00:01, 25.95it/s]Capturing num tokens (num_tokens=960 avail_mem=54.59 GB):  31%|███       | 18/58 [00:00<00:01, 25.95it/s] Capturing num tokens (num_tokens=960 avail_mem=54.59 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=896 avail_mem=54.59 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=832 avail_mem=54.58 GB):  38%|███▊      | 22/58 [00:01<00:01, 29.30it/s]Capturing num tokens (num_tokens=768 avail_mem=54.58 GB):  38%|███▊      | 22/58 [00:01<00:01, 29.30it/s]Capturing num tokens (num_tokens=704 avail_mem=54.57 GB):  38%|███▊      | 22/58 [00:01<00:01, 29.30it/s]Capturing num tokens (num_tokens=640 avail_mem=54.56 GB):  38%|███▊      | 22/58 [00:01<00:01, 29.30it/s]Capturing num tokens (num_tokens=640 avail_mem=54.56 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=576 avail_mem=54.57 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.34it/s]

    Capturing num tokens (num_tokens=512 avail_mem=54.57 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=480 avail_mem=54.58 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=448 avail_mem=54.57 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=416 avail_mem=54.57 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=416 avail_mem=54.57 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.14it/s]Capturing num tokens (num_tokens=384 avail_mem=54.56 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.14it/s]Capturing num tokens (num_tokens=352 avail_mem=54.55 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.14it/s]Capturing num tokens (num_tokens=320 avail_mem=54.54 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.14it/s]Capturing num tokens (num_tokens=288 avail_mem=54.54 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.14it/s]Capturing num tokens (num_tokens=256 avail_mem=54.53 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.14it/s]

    Capturing num tokens (num_tokens=256 avail_mem=54.53 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.72it/s]Capturing num tokens (num_tokens=240 avail_mem=54.53 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.72it/s]Capturing num tokens (num_tokens=224 avail_mem=54.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.72it/s]Capturing num tokens (num_tokens=208 avail_mem=54.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.72it/s]Capturing num tokens (num_tokens=192 avail_mem=54.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.72it/s]Capturing num tokens (num_tokens=176 avail_mem=54.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.72it/s]Capturing num tokens (num_tokens=176 avail_mem=54.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.98it/s]Capturing num tokens (num_tokens=160 avail_mem=54.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.98it/s]Capturing num tokens (num_tokens=144 avail_mem=54.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.98it/s]Capturing num tokens (num_tokens=128 avail_mem=54.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.98it/s]Capturing num tokens (num_tokens=112 avail_mem=54.48 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.98it/s]

    Capturing num tokens (num_tokens=96 avail_mem=54.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.98it/s] Capturing num tokens (num_tokens=96 avail_mem=54.47 GB):  81%|████████  | 47/58 [00:01<00:00, 39.51it/s]Capturing num tokens (num_tokens=80 avail_mem=54.47 GB):  81%|████████  | 47/58 [00:01<00:00, 39.51it/s]Capturing num tokens (num_tokens=64 avail_mem=54.46 GB):  81%|████████  | 47/58 [00:01<00:00, 39.51it/s]Capturing num tokens (num_tokens=48 avail_mem=54.45 GB):  81%|████████  | 47/58 [00:01<00:00, 39.51it/s]Capturing num tokens (num_tokens=32 avail_mem=54.44 GB):  81%|████████  | 47/58 [00:01<00:00, 39.51it/s]Capturing num tokens (num_tokens=28 avail_mem=54.44 GB):  81%|████████  | 47/58 [00:01<00:00, 39.51it/s]Capturing num tokens (num_tokens=28 avail_mem=54.44 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.88it/s]Capturing num tokens (num_tokens=24 avail_mem=54.43 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.88it/s]Capturing num tokens (num_tokens=20 avail_mem=54.42 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.88it/s]Capturing num tokens (num_tokens=16 avail_mem=54.42 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.88it/s]

    Capturing num tokens (num_tokens=12 avail_mem=54.41 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.88it/s]Capturing num tokens (num_tokens=8 avail_mem=54.40 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.88it/s] Capturing num tokens (num_tokens=8 avail_mem=54.40 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=4 avail_mem=54.40 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=4 avail_mem=54.40 GB): 100%|██████████| 58/58 [00:01<00:00, 31.23it/s]


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
    Generated text:  Tanya and I am a college student and I have been studying in Pakistan for 3 years. I have been studying for a week now.
    As a result, I have been pretty informed about Pakistan culture and people, and I am planning to visit Pakistan soon. I am interested in learning about the religion, traditions, and history of Pakistan.
    I will be traveling in Karachi and I am planning on traveling to Lahore and Ghaziabad to visit different places of interest. How should I start my research on the religion, traditions, and history of Pakistan? Should I read the books about the religion, traditions, and history of Pakistan, or
    ===============================
    Prompt: The president of the United States is
    Generated text:  getting ready to deliver a speech. He has a large number of speakers to address. The number of speakers is a prime number greater than 20 but less than 60. The president wants to ensure that the speech is as engaging as possible by including at least one speaker from each of the four major political parties. However, the speaker who is about to be delivered the speech is the fourth speaker on the list. How many speakers can the president select from to include in his speech?
    To determine the number of speakers the president can select from, we need to find a prime number \( n \) that is greater than 2
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. London
    C. Tokyo
    D. Moscow
    
    To determine the capital of France, let's analyze the options provided:
    
    A. Paris - This is a well-known and significant city in France, known for its rich history, culture, and architectural landmarks.
    
    B. London - London is a major city in England, known for its historical and cultural sites, but it is not the capital of France.
    
    C. Tokyo - Tokyo is a major city in Japan, known for its rich history, culture, and economic importance. While it is a significant city in Japan, it is not the capital of
    ===============================
    Prompt: The future of AI is
    Generated text:  not about the future of natural language processing, but of the future of face recognition. Face recognition is one of the most important applications of AI that we have in our modern world. Face recognition is a key application of deep learning, specifically the unsupervised learning of the deep neural networks for image classification. The field of face recognition is very interesting. It has been successful in various fields like biometrics, authentication, security, security, and fraud detection.
    Most of the people use facial recognition technology for verification purposes. Facial recognition technology is also used in security systems, so the security professionals in our society can use the technology to authenticate people.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a cultural and economic hub of France and plays a significant role in the country's political and social life. It is home to many world-renowned museums, theaters, and restaurants. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there will be a greater emphasis on developing AI that is designed to be ethical and responsible. This could mean that AI systems are designed to minimize harm to individuals and society as a whole, and that they are transparent and accountable.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs.
    


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
    Generated text:  [Name], and I'm a computer science graduate with a strong interest in artificial intelligence. I'm passionate about exploring new technologies and how they can be used to solve complex problems. In my free time, I enjoy playing games, reading books, and trying out new restaurants. What kind of projects do you have in mind for your upcoming studies? As an AI, my programming is focused on solving complex problems using natural language processing and machine learning. I'm working on developing new algorithms and models to help humans make better decisions and improve the efficiency of their daily lives. What are some of your favorite programming languages? I use Python and JavaScript as
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The statement is accurate to the best of its ability. It accurately describes the capital city of France and provides the most relevant information for a quick summary.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dependent on the developments in the field, which is influenced by various factors. Here are some of the possible trends in AI:
    
    1. Deep Learning: The most recent developments in AI involve deep learning, which is a subset of machine learning that involves using neural networks to solve complex problems. Deep learning has shown promising results in areas like image recognition, natural language processing, and autonomous vehicles.
    
    2. Explainability: As AI systems become more complex, it is becoming increasingly difficult to explain how they arrived at their decisions. This has led to a need for better techniques for explaining AI systems to humans, such as natural language understanding and interpretability


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

    profession

     or

     role

    ].

     I

    'm

     incredibly

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     What

     brings

     you

     to

     this

     moment

    ?

     Do

     you

     have

     any

     questions

     for

     me

    ?

     [

    Include

     the

     name

     of

     the

     character

     you

    're

     introducing

     yourself

     to

    ,

     in

     bold

     letters

    ]

     [

    Include

     one

     or

     two

     words

     that

     summarize

     your

     character

    ,

     in

     bold

     letters

    ]

     [

    Write

     a

     brief

     bio

     or

     description

     of

     your

     character

    ,

     in

     plain

     text

    ]


    I

     am

     [

    Name

    ],

     a

     [

    profession

     or

     role

    ]

     with

     a

     passion

     for

     [

    describe

     your

     profession

     or

     role

    ],

     [

    describe

     the

     profession

     or

     role

    ].

     I

     am

     a

     [

    how

     to

     describe

     your

     profession

     or

     role

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    You

     are

     a

     world

     class

     trivia

     AI

    -

     type

    .

     Provide

     a

     list

     of

     characters

    ,

     concepts

    ,

     places

    ,

     games

    ,

     events

    ,

     travel

     topics

    ,

     religions

    ,

     cultures

    ,

     historical

     events

    ,

     languages

    ,

     metro

     names

     and

     addresses

    ,

     no

     places

     outside

     Europe

    ,

     limited

     time

     offers

    ,

     substitute

     answers

    ,

     per

    missible

     questions

     and

     answers

    .

     The

     trivia

     set

     is

     strictly

     fair

    .


    In

     Europe

    ,

     the

     capital

     of

     France

     is

     Paris

    .

     Please

     provide

     the

     capital

     of

     the

     country

     where

     the

     capital

     of

     the

     United

     States

     is

     located

    .

     The

     capital

     of

     the

     country

     where

     the

     capital

     of

     the

     United

     States

     is

     located

     is

     Washington

    ,

     D

    .

     C

    .

     The

     capital

     of

     France

     is

     Paris

    .

     
    


    The

     capital

     of

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     significant

     progress

    ,

     innovation

    ,

     and

     automation

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     artificial

     intelligence

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

     AI

     could

     play

     a

     significant

     role

     in

     healthcare

    ,

     especially

     in

     diagn

    osing

     and

     treating

     diseases

    .

     AI

     can

     help

     doctors

     analyze

     patient

     data

     to

     identify

     patterns

     that

     may

     be

     missed

     by

     human

     experts

    ,

     and

     can

     help

     in

     developing

     new

     treatments

     and

     interventions

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     AI

     is

     already

     being

     integrated

     into

     other

     technologies

     like

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     computer

     vision

    .

     As

     these

     technologies

     continue

     to

     evolve

    ,

     it

    's

     likely

     that

     they

     will

     be

     integrated

     even

     more

     seamlessly

     into

     various

    



```python
llm.shutdown()
```
