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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.77it/s]


    2026-05-07 00:00:40,390 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 00:00:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  5.85it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  5.85it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  5.85it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  5.85it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  5.85it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  5.85it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:05<00:07,  5.85it/s]

    Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:05<00:07,  5.85it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:05<00:07,  5.85it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03, 11.19it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03, 11.19it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03, 11.19it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03, 11.19it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03, 11.19it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03, 11.19it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03, 11.19it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03, 11.19it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03, 11.19it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 17.51it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 17.51it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 17.51it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 17.51it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 17.51it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 17.51it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 17.51it/s]

    Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 17.51it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 17.51it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 24.76it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 24.76it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 24.76it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 24.76it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 24.76it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 24.76it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 24.76it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 24.76it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 24.76it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 24.76it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 33.75it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 33.75it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 33.75it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 33.75it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 33.75it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 33.75it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 33.75it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 33.75it/s]

    Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 33.75it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 33.75it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 33.75it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 44.71it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 44.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.05 GB):   2%|▏         | 1/58 [00:00<00:06,  8.79it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.01 GB):   2%|▏         | 1/58 [00:00<00:06,  8.79it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=72.01 GB):   3%|▎         | 2/58 [00:00<00:06,  8.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.00 GB):   3%|▎         | 2/58 [00:00<00:06,  8.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.00 GB):   5%|▌         | 3/58 [00:00<00:06,  8.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.99 GB):   5%|▌         | 3/58 [00:00<00:06,  8.39it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.99 GB):   7%|▋         | 4/58 [00:00<00:06,  8.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.97 GB):   7%|▋         | 4/58 [00:00<00:06,  8.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.86 GB):   7%|▋         | 4/58 [00:00<00:06,  8.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.86 GB):  10%|█         | 6/58 [00:00<00:05,  9.84it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.97 GB):  10%|█         | 6/58 [00:00<00:05,  9.84it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=71.96 GB):  10%|█         | 6/58 [00:00<00:05,  9.84it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.96 GB):  14%|█▍        | 8/58 [00:00<00:04, 12.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.95 GB):  14%|█▍        | 8/58 [00:00<00:04, 12.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.95 GB):  14%|█▍        | 8/58 [00:00<00:04, 12.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.88 GB):  14%|█▍        | 8/58 [00:00<00:04, 12.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.88 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.93 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.08it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.93 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.90 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.90 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.91 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.89 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.84it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.88 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.88 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.84it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=71.88 GB):  31%|███       | 18/58 [00:01<00:01, 21.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.87 GB):  31%|███       | 18/58 [00:01<00:01, 21.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.88 GB):  31%|███       | 18/58 [00:01<00:01, 21.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.86 GB):  31%|███       | 18/58 [00:01<00:01, 21.81it/s]Capturing num tokens (num_tokens=960 avail_mem=71.87 GB):  31%|███       | 18/58 [00:01<00:01, 21.81it/s] Capturing num tokens (num_tokens=960 avail_mem=71.87 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.31it/s]Capturing num tokens (num_tokens=896 avail_mem=71.86 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.31it/s]Capturing num tokens (num_tokens=832 avail_mem=71.87 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.31it/s]Capturing num tokens (num_tokens=768 avail_mem=71.86 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.31it/s]

    Capturing num tokens (num_tokens=704 avail_mem=71.86 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.31it/s]Capturing num tokens (num_tokens=704 avail_mem=71.86 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.15it/s]Capturing num tokens (num_tokens=640 avail_mem=71.85 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.15it/s]Capturing num tokens (num_tokens=576 avail_mem=71.85 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.15it/s]Capturing num tokens (num_tokens=512 avail_mem=71.83 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.15it/s]Capturing num tokens (num_tokens=480 avail_mem=71.84 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.15it/s]Capturing num tokens (num_tokens=480 avail_mem=71.84 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.83it/s]Capturing num tokens (num_tokens=448 avail_mem=71.84 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.83it/s]Capturing num tokens (num_tokens=416 avail_mem=71.83 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.83it/s]Capturing num tokens (num_tokens=384 avail_mem=71.82 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.83it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.82 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.83it/s]Capturing num tokens (num_tokens=352 avail_mem=71.82 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=320 avail_mem=71.81 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=288 avail_mem=71.80 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=256 avail_mem=71.80 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=240 avail_mem=71.79 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.73it/s]Capturing num tokens (num_tokens=240 avail_mem=71.79 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=224 avail_mem=71.76 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=208 avail_mem=71.76 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.34it/s]

    Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=176 avail_mem=71.71 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=176 avail_mem=71.71 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.48it/s]Capturing num tokens (num_tokens=160 avail_mem=71.73 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.48it/s]Capturing num tokens (num_tokens=144 avail_mem=71.72 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.48it/s]Capturing num tokens (num_tokens=128 avail_mem=71.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.48it/s]

    Capturing num tokens (num_tokens=112 avail_mem=71.71 GB):  72%|███████▏  | 42/58 [00:02<00:00, 30.48it/s]Capturing num tokens (num_tokens=112 avail_mem=71.71 GB):  79%|███████▉  | 46/58 [00:02<00:00, 27.38it/s]Capturing num tokens (num_tokens=96 avail_mem=71.70 GB):  79%|███████▉  | 46/58 [00:02<00:00, 27.38it/s] Capturing num tokens (num_tokens=80 avail_mem=71.68 GB):  79%|███████▉  | 46/58 [00:02<00:00, 27.38it/s]Capturing num tokens (num_tokens=64 avail_mem=71.67 GB):  79%|███████▉  | 46/58 [00:02<00:00, 27.38it/s]Capturing num tokens (num_tokens=64 avail_mem=71.67 GB):  84%|████████▍ | 49/58 [00:02<00:00, 25.94it/s]Capturing num tokens (num_tokens=48 avail_mem=71.66 GB):  84%|████████▍ | 49/58 [00:02<00:00, 25.94it/s]

    Capturing num tokens (num_tokens=32 avail_mem=71.66 GB):  84%|████████▍ | 49/58 [00:02<00:00, 25.94it/s]Capturing num tokens (num_tokens=28 avail_mem=71.64 GB):  84%|████████▍ | 49/58 [00:02<00:00, 25.94it/s]Capturing num tokens (num_tokens=28 avail_mem=71.64 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.83it/s]Capturing num tokens (num_tokens=24 avail_mem=71.64 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.83it/s]Capturing num tokens (num_tokens=20 avail_mem=71.53 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.83it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.83it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.54it/s]Capturing num tokens (num_tokens=12 avail_mem=71.61 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.54it/s]

    Capturing num tokens (num_tokens=8 avail_mem=71.60 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.54it/s] Capturing num tokens (num_tokens=4 avail_mem=71.60 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.54it/s]Capturing num tokens (num_tokens=4 avail_mem=71.60 GB): 100%|██████████| 58/58 [00:02<00:00, 22.34it/s]Capturing num tokens (num_tokens=4 avail_mem=71.60 GB): 100%|██████████| 58/58 [00:02<00:00, 22.35it/s]


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
    Generated text:  Amelia, I'm a girl. My favorite subject is music. My favorite singer is Taylor Swift. I have a sister named Caroline. I live in New York City. What kind of music do you like best?
    
    Amelia: I like Taylor Swift, but I also like the music of Prince. I really like the music of David Bowie, too. But I don't like pop music very much. My favorite rock band is The Kinks. Their music is very loud and the lyrics are very poetic.
    
    What is Amelia's favorite song? Write a short paragraph describing your answer. This week, I will be playing a concert for my
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. There are a lot of people who are very important in the United States. There are 50 states in the United States, and there are people who are in each state. Every state has a governor. There are also a lot of people who are in the House of Representatives. The President is elected to run for a second term. They are elected from the people who voted in the 2016 election. They have to have a running mate. The president can be from any party. It can be the Republican Party or the Democratic Party. The president is in charge of the country
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. In a mathematics class, students are studying the properties of square matrices and their eigenvalues. Inspired by the concept of diagonalizing a matrix, they decide to explore a simpler, yet related, idea. They consider a 2x2 matrix $A$ defined as follows:
    $$
    A = \begin{bmatrix}
    3 & 4 \\
    1 & 2
    \end{bmatrix}
    $$
    The students are asked to find the eigenvalues of matrix $A$ and then calculate the sum of these eigenvalues. What is the sum of the eigenvalues of matrix $A$?
    To find the eigenvalues
    ===============================
    Prompt: The future of AI is
    Generated text:  about the security of data and personal privacy. With AI and blockchain, the next generation of security technologies can enable more data to be stored securely, and the tools of AI and blockchain can enable more secure and efficient transactions.
    The future of AI is about the security of data and personal privacy. With AI and blockchain, the next generation of security technologies can enable more data to be stored securely, and the tools of AI and blockchain can enable more secure and efficient transactions.
    Decentralized financial systems are a new kind of financial system with very low transaction costs, low overhead and low energy consumption, and have the potential to revolutionize the financial industry


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also home to many world-renowned museums, theaters, and restaurants. Paris is a cultural and historical center that has played a significant role in French history and continues to be a major economic and political center of the country. The city is also known for its fashion industry, which has produced many famous designers and brands. Paris is a vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions between the two.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI becomes more advanced, it is likely to be used in even more complex and personalized ways, with the potential to revolutionize the field of medicine.
    
    3. Greater use of AI in education: AI is already being used in education to personalize learning experiences, improve student engagement, and
    


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
    Generated text:  [Name] and I'm a [age] year old [nationality] with [number of years] years of experience. My background is [briefly describe my career or education]. I'm passionate about [why you love this field or subject]. I'm also a [specialization or interest] enthusiast. I believe I have a [strength or weakness] with [mention a skill or skill set you have]. I enjoy [reason for being interested in this topic]. My [strength or weakness] is [mention the thing that makes you shine in this field]. I'm someone you can trust because I'm always [something you do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and is known for its architecture, art, and cuisine. Paris is also a major transportation hub and a popular tourist destination. The city is home to many prestigious institutions and organizations, such as the Louvre Museum and the Notre-Dame Cathedral. Paris is a city of diverse cultures and is home to over 2 million people. It is the cultural, economic, and political center of France. Paris is also a symbol of France and a tourist destination for millions of people around the world. The city is a popular tourist destination and is known for its beautiful architecture, museums, and food and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid and significant advancements. Some potential future trends include:
    
    1. Enhanced AI capabilities: As AI technology continues to evolve, we can expect to see further improvements in capabilities such as self-learning, multitasking, and problem-solving.
    
    2. Integration with human cognitive abilities: AI systems are expected to become more integrated with human cognitive abilities, allowing for more natural and effective communication between humans and AI.
    
    3. Greater use of AI in healthcare: AI is already being used in medical diagnostics and treatment planning, but as the technology advances, we can expect to see greater integration with the healthcare system.
    
    4. Enhanced privacy and data


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

    职

    别

    ]

     with

     [

    公司

    名

    ].

     I

    'm

     currently

     working

     on

     [

    Project

    /

    Position

    ]

     and

     have

     been

     [

    short

     introduction

     to

     your

     personal

     interests

     or

     passions

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     As

     an

     AI

     language

     model

    ,

     I

     can

     provide

     information

     and

     insights

     on

     my

     capabilities

     and

     limitations

    .

     However

    ,

     my

     primary

     focus

     is

     to

     assist

     and

     provide

     useful

     information

     to

     people

    .

     What

     can

     you

     tell

     me

     about

     yourself

     and

     what

     makes

     you

     special

     to

     me

    ?

     As

     an

     AI

     language

     model

    ,

     I

     can

     provide

     information

     and

     insights

     on

     my

     capabilities

     and

     limitations

    .

     However

    ,

     my

     primary

     focus

     is

     to

     assist

     and

     provide

     useful

     information

     to

     people

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Vil

    lette

    "

     or

     "

    La

     Vict

    oire

    "

     and

     is

     located

     on

     the

     River

     Se

    ine

    .
    


    Here

    's

     a

     concise

     factual

     statement

     about

     Paris

    ,

     France

    :
    


    Paris

     is

     the

     capital

     and

     most

     populous

     city

     of

     France

    ,

     located

     on

     the

     River

     Se

    ine

     in

     the

     center

     of

     the

     country

    .
    


    This

     statement

     encaps

    ulates

     the

     key

     information

     about

     Paris

    ,

     including

    :
    


    1

    .

     Its

     status

     as

     the

     capital

    


    2

    .

     Its

     current

     location

    


    3

    .

     Its

     population

     (

    most

     populous

     city

     in

     Europe

    )


    4

    .

     Its

     significance

     as

     a

     major

     city

     in

     France

    
    


    This

     statement

     provides

     a

     brief

     overview

     of

     Paris

    's

     key

     facts

    ,

     which

     are

     typically

     contained

     in

     a

     

    5

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     key

     trends

    :
    


    1

    .

     Increased

     sophistication

    :

     AI

     will

     continue

     to

     evolve

     and

     become

     more

     sophisticated

    ,

     with

     breakthrough

    s

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

     and

     computer

     vision

    .
    


    2

    .

     Extended

     use

    :

     AI

     will

     increasingly

     be

     integrated

     into

     everyday

     life

    ,

     from

     smart

     homes

     to

     self

    -driving

     cars

    ,

     and

     beyond

    .
    


    3

    .

     Eth

    ical

     concerns

    :

     AI

     will

     likely

     face

     greater

     scrutiny

     from

     regulators

     and

     the

     public

    ,

     as

     concerns

     about

     bias

    ,

     transparency

    ,

     and

     accountability

     emerge

    .
    


    4

    .

     Personal

    ization

    :

     AI

     will

     allow

     for

     more

     personalized

     and

     adaptive

     experiences

    ,

     as

     machines

     learn

     from

     user

     behavior

     and

     preferences

     over

     time

    .
    


    5

    .

     Autonomous

     vehicles

    :

     AI

    



```python
llm.shutdown()
```
