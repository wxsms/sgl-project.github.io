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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.42it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.41it/s]


    2026-05-04 02:16:46,762 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 02:16:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:08,  1.25s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:08,  1.25s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:08,  1.25s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:08,  1.25s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:26,  1.97it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:26,  1.97it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:26,  1.97it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:26,  1.97it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:26,  1.97it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:11,  4.01it/s]

    Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:05,  7.95it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:05,  7.95it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:05,  7.95it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:05,  7.95it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:05<00:05,  7.95it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:05<00:05,  7.95it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:03, 11.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:03, 11.81it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:03, 11.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:03, 11.81it/s]

    Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:03, 11.81it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:02, 15.00it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:02, 15.00it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:02, 15.00it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:02, 15.00it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:02, 15.00it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 17.73it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 17.73it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 17.73it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 17.73it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 17.73it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 20.46it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 20.46it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 20.46it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 20.46it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 20.46it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 20.46it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 20.46it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 27.40it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 27.40it/s]

    Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 27.40it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 27.40it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 27.40it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 27.40it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 27.40it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 27.40it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 34.78it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 34.78it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 34.78it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 34.78it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 34.78it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 34.78it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 34.78it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 34.78it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 42.50it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 42.50it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 42.50it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 42.50it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 42.50it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 42.50it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.61 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.61 GB):   3%|▎         | 2/58 [00:00<00:03, 15.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.60 GB):   3%|▎         | 2/58 [00:00<00:03, 15.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.58 GB):   3%|▎         | 2/58 [00:00<00:03, 15.40it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.58 GB):   7%|▋         | 4/58 [00:00<00:03, 17.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.58 GB):   7%|▋         | 4/58 [00:00<00:03, 17.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.56 GB):   7%|▋         | 4/58 [00:00<00:03, 17.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.55 GB):   7%|▋         | 4/58 [00:00<00:03, 17.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.55 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.56 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.56 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.57it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=53.55 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.55 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.54 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.54 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.53 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.53 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.53 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.52 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.32it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=53.50 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.49 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.49 GB):  31%|███       | 18/58 [00:00<00:01, 28.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.49 GB):  31%|███       | 18/58 [00:00<00:01, 28.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.49 GB):  31%|███       | 18/58 [00:00<00:01, 28.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.46 GB):  31%|███       | 18/58 [00:00<00:01, 28.19it/s]Capturing num tokens (num_tokens=960 avail_mem=53.48 GB):  31%|███       | 18/58 [00:00<00:01, 28.19it/s] Capturing num tokens (num_tokens=896 avail_mem=53.48 GB):  31%|███       | 18/58 [00:00<00:01, 28.19it/s]Capturing num tokens (num_tokens=896 avail_mem=53.48 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=832 avail_mem=53.47 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=768 avail_mem=53.47 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.65it/s]

    Capturing num tokens (num_tokens=704 avail_mem=53.47 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=640 avail_mem=53.46 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=576 avail_mem=53.46 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=576 avail_mem=53.46 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.16it/s]Capturing num tokens (num_tokens=512 avail_mem=53.45 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.16it/s]Capturing num tokens (num_tokens=480 avail_mem=53.46 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.16it/s]Capturing num tokens (num_tokens=448 avail_mem=53.46 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=416 avail_mem=53.46 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=384 avail_mem=53.46 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=384 avail_mem=53.46 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=352 avail_mem=53.45 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.73it/s]

    Capturing num tokens (num_tokens=320 avail_mem=53.44 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=288 avail_mem=53.44 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=256 avail_mem=53.44 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=240 avail_mem=53.44 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=240 avail_mem=53.44 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=224 avail_mem=53.43 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=208 avail_mem=53.43 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.41it/s]

    Capturing num tokens (num_tokens=192 avail_mem=53.43 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=176 avail_mem=53.42 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=160 avail_mem=53.42 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.41it/s]

    Capturing num tokens (num_tokens=160 avail_mem=53.42 GB):  74%|███████▍  | 43/58 [00:01<00:00, 24.26it/s]Capturing num tokens (num_tokens=144 avail_mem=53.42 GB):  74%|███████▍  | 43/58 [00:01<00:00, 24.26it/s]Capturing num tokens (num_tokens=128 avail_mem=53.42 GB):  74%|███████▍  | 43/58 [00:01<00:00, 24.26it/s]

    Capturing num tokens (num_tokens=112 avail_mem=53.41 GB):  74%|███████▍  | 43/58 [00:01<00:00, 24.26it/s]Capturing num tokens (num_tokens=96 avail_mem=53.41 GB):  74%|███████▍  | 43/58 [00:01<00:00, 24.26it/s] Capturing num tokens (num_tokens=96 avail_mem=53.41 GB):  81%|████████  | 47/58 [00:01<00:00, 17.61it/s]Capturing num tokens (num_tokens=80 avail_mem=53.41 GB):  81%|████████  | 47/58 [00:01<00:00, 17.61it/s]

    Capturing num tokens (num_tokens=64 avail_mem=53.40 GB):  81%|████████  | 47/58 [00:02<00:00, 17.61it/s]Capturing num tokens (num_tokens=48 avail_mem=53.40 GB):  81%|████████  | 47/58 [00:02<00:00, 17.61it/s]Capturing num tokens (num_tokens=48 avail_mem=53.40 GB):  86%|████████▌ | 50/58 [00:02<00:00, 16.00it/s]Capturing num tokens (num_tokens=32 avail_mem=53.40 GB):  86%|████████▌ | 50/58 [00:02<00:00, 16.00it/s]

    Capturing num tokens (num_tokens=28 avail_mem=53.39 GB):  86%|████████▌ | 50/58 [00:02<00:00, 16.00it/s]Capturing num tokens (num_tokens=24 avail_mem=53.39 GB):  86%|████████▌ | 50/58 [00:02<00:00, 16.00it/s]Capturing num tokens (num_tokens=24 avail_mem=53.39 GB):  91%|█████████▏| 53/58 [00:02<00:00, 15.26it/s]Capturing num tokens (num_tokens=20 avail_mem=53.39 GB):  91%|█████████▏| 53/58 [00:02<00:00, 15.26it/s]

    Capturing num tokens (num_tokens=16 avail_mem=53.39 GB):  91%|█████████▏| 53/58 [00:02<00:00, 15.26it/s]Capturing num tokens (num_tokens=16 avail_mem=53.39 GB):  95%|█████████▍| 55/58 [00:02<00:00, 14.36it/s]Capturing num tokens (num_tokens=12 avail_mem=53.38 GB):  95%|█████████▍| 55/58 [00:02<00:00, 14.36it/s]

    Capturing num tokens (num_tokens=8 avail_mem=53.38 GB):  95%|█████████▍| 55/58 [00:02<00:00, 14.36it/s] Capturing num tokens (num_tokens=8 avail_mem=53.38 GB):  98%|█████████▊| 57/58 [00:02<00:00, 13.54it/s]Capturing num tokens (num_tokens=4 avail_mem=53.37 GB):  98%|█████████▊| 57/58 [00:02<00:00, 13.54it/s]Capturing num tokens (num_tokens=4 avail_mem=53.37 GB): 100%|██████████| 58/58 [00:02<00:00, 20.20it/s]


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
    Generated text:  Vann and I am 25 years old. My name is a word I have been looking for for the last few years. I am a self-proclaimed New York City native. I have traveled to a number of countries but never to the United States. I have been to Paris, Tokyo, Mexico, and New York City. What is the word that best matches my search for the last few years? My name is a word I have been searching for for the last few years. I am a self-proclaimed New York City native. My name is a word I have been searching for for the last few years. I am
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. The president of the United States is from the United States. The president of the United States is a person. Therefore, the president of the United States is a person.
    This logic argument is flawed because:
    A. it introduces a new category that it did not define
    B. it lacks an internal logical link
    C. it violates the principle of an infinite regress
    D. it involves circular reasoning
    E. it involves a definition that is ambiguous
    To determine the type of logical fallacy in the given argument, let's analyze each option step by step:
    
    1. **Option A: It introduces a new category that
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is an important historical center of Europe with a long history of outstanding culture and architectural works. Paris has a rich culture that is reflected in its many museums and galleries. It is known for the Louvre and the Eiffel Tower. Some of the world's most famous films and plays are also filmed here.
    
    Based on that paragraph can we conclude that this sentence is true?
    Paris has a rich history.
    
    pick from the following. (A). yes (B). no (A). yes
    The paragraph states that Paris has a rich history of outstanding culture and architectural works. It also mentions that it has many museums and galleries
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but what will it be like? This is a very important question, especially given the rapidly changing nature of the world. In the past, people have been warned that the future of AI would be bleak. We could have a world where AI takes over our jobs, and a world where it does not. However, today, the view is changing. In fact, people are already beginning to expect a bright future of AI.
    The internet has provided a perfect platform for AI research, and it has enabled us to explore a wide range of possibilities. For example, AI has been used to create intelligent assistants, chatbots, and virtual


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that is home to the Eiffel Tower and the Louvre Museum. It is also the seat of the French government and the country's cultural and political capital. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is also home to many famous landmarks and attractions, including the Notre-Dame Cathedral, the Champs-Élysées, and the Montmartre neighborhood. Paris is a vibrant and diverse city with a rich cultural heritage that continues to attract visitors from around the world. The city is also known for its fashion industry, with many famous designers and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations and tasks.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI becomes more advanced, it is likely to be used in even more areas, including personalized medicine,
    


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
    Generated text:  Jane and I'm a 30-year-old software engineer with a degree in computer science. I love to code and develop new applications. I have a passion for learning new technologies and trying to stay up-to-date with the latest trends. I'm always looking for ways to improve my skills and stay relevant in my field. I also enjoy working with people and collaborating with other engineers to solve problems. I'm very dedicated to my work and try to work long hours to stay productive. I'm not afraid to take risks and try new things, but I try to do so responsibly and ethically. I'm very collaborative and enjoy working in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also home to the French Parliament and the United Nations Headquarters. Paris is known for its elegant architecture and cultural heritage, as well as its reputation for being a welcoming and livable city for tourists and locals alike. Its location in the heart of the French-speaking world has made it a global city with a diverse and multicultural population. In terms of transportation, Paris is served by various modes of public transport, including metro, bus, and taxi, and it is also well-connected by the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a range of technological developments that will push the boundaries of what is possible and transform the way we live, work and interact with technology. Here are some possible future trends in artificial intelligence:
    
    1. Increased sophistication of AI: As technology continues to advance, AI will become more sophisticated and capable of performing a wider range of tasks. This includes better understanding human emotions, decision-making, and decision-making ability. AI will also be able to learn from data, improve over time, and adapt to changing situations.
    
    2. Improved privacy and security: As AI becomes more integrated into our lives, there will be greater concern about privacy


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

     who

     grew

     up

     in

     [

    City

    ]

     [

    Country

    ].

     I

     have

     a

     keen

     interest

     in

     [

    Topic

    ],

     and

     I

     believe

     I

     can

     be

     a

     [

    Skill

    ]

     in

     the

     field

    .

     I

    'm

     constantly

     learning

     and

     growing

    ,

     and

     I

    'm

     always

     ready

     to

     share

     what

     I

    've

     learned

     with

     anyone

     who

     listens

    .

     My

     favorite

     way

     to

     spend

     my

     time

     is

     [

    Favorite

     Activity

    ],

     and

     I

    'm

     always

     looking

     for

     opportunities

     to

     improve

     myself

     and

     grow

     as

     a

     person

    .

     I

    'm

     [

    What

     To

     Do

     When

    ]

     and

     I

    'm

     always

     ready

     to

     help

     others

     grow

     in

     their

     own

     way

    .

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

    
    
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

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

     and

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     vibrant

     culture

    .

     The

     city

     is

     also

     a

     hub

     of

     business

     and

     politics

    ,

     and

     plays

     an

     important

     role

     in

     European

     diplomacy

    .

     Paris

     has

     a

     population

     of

     over

     two

     million

     and

     is

     considered

     to

     be

     one

     of

     the

     most

     diverse

     cities

     in

     the

     world

    .

     It

     is

     known

     for

     its

     music

    ,

     fashion

    ,

     and

     chocolate

    ,

     and

     has

     a

     rich

     cultural

     heritage

     dating

     back

     to

     the

     Roman

     Empire

    .

     It

     is

     also

     home

     to

     the

     Lou

    vre

     Museum

     and

     the

     Notre

    -D

    ame

     Cathedral

    , which

     are

     considered

     some

     of

     the

     world

    ’s

     most

     famous

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

     and

     is

     expected

     to

     continue

     to

     evolve

     and

     develop

    .

     Some

     possible

     trends

     include

    :
    


    1

    .

     Increased

     automation

    :

     AI

     will

     continue

     to

     automate

     various

     tasks

    ,

     from

     mundane

     to

     complex

     ones

    ,

     ensuring

     that

     humans

     can

     focus

     on

     more

     complex

     and

     creative

     activities

    .
    


    2

    .

     Improved

     natural

     language

     processing

    :

     As

     more

     data

     is

     available

    ,

     AI

     will

     become

     even

     more

     adept

     at

     processing

     and

     understanding

     natural

     language

    ,

     making

     it

     easier

     for

     machines

     to

     communicate

     and

     interact

     with

     humans

    .
    


    3

    .

     Enhanced

     predictive

     analytics

    :

     AI

     will

     be

     able

     to

     predict

     outcomes

     with

     greater

     accuracy

     and

     confidence

    ,

     helping

     businesses

     make

     more

     informed

     decisions

    .
    


    4

    .

     Development

     of

     super

     AI

    :

     AI

     may

     be

     able

     to

     generate

     its

     own

     intelligence

    



```python
llm.shutdown()
```
