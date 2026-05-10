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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.17it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.17it/s]


    2026-05-10 03:37:27,248 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 03:37:27] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:58,  5.24s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.37it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.37it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.37it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.37it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.37it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.37it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.37it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.37it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.37it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.13it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.13it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 11.21it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 11.21it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 11.21it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 11.21it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 11.21it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 11.21it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 11.21it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 14.87it/s]

    Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 14.87it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:01, 18.75it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:01, 18.75it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:01, 18.75it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:01, 18.75it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:06<00:01, 18.75it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:06<00:01, 18.75it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:06<00:01, 18.75it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:06<00:01, 18.75it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 24.62it/s]

    Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 24.62it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:06<00:00, 31.13it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:06<00:00, 31.13it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:06<00:00, 31.13it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:06<00:00, 31.13it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:06<00:00, 31.13it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:06<00:00, 31.13it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:06<00:00, 31.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 17.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.31it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.31it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.84it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.84it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.35 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.35 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.35 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.10 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.34 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.88it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.34 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.33 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.33 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.31 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.31 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.14 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.14 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.52it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.17 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.17 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.91it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.15 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.91it/s]Capturing num tokens (num_tokens=960 avail_mem=74.28 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.91it/s] Capturing num tokens (num_tokens=896 avail_mem=74.28 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.91it/s]Capturing num tokens (num_tokens=896 avail_mem=74.28 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.32it/s]Capturing num tokens (num_tokens=832 avail_mem=74.27 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.32it/s]Capturing num tokens (num_tokens=768 avail_mem=74.26 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.32it/s]

    Capturing num tokens (num_tokens=704 avail_mem=74.26 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.32it/s]Capturing num tokens (num_tokens=640 avail_mem=74.25 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.32it/s]Capturing num tokens (num_tokens=640 avail_mem=74.25 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.90it/s]Capturing num tokens (num_tokens=576 avail_mem=74.25 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.90it/s]Capturing num tokens (num_tokens=512 avail_mem=74.23 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.90it/s]Capturing num tokens (num_tokens=480 avail_mem=74.24 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.90it/s]Capturing num tokens (num_tokens=448 avail_mem=74.24 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.90it/s]Capturing num tokens (num_tokens=448 avail_mem=74.24 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=416 avail_mem=74.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=384 avail_mem=74.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.71it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=320 avail_mem=74.21 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=320 avail_mem=74.21 GB):  60%|██████    | 35/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=288 avail_mem=74.20 GB):  60%|██████    | 35/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=256 avail_mem=74.20 GB):  60%|██████    | 35/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=240 avail_mem=74.18 GB):  60%|██████    | 35/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=224 avail_mem=74.17 GB):  60%|██████    | 35/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=224 avail_mem=74.17 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.21it/s]Capturing num tokens (num_tokens=208 avail_mem=74.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.21it/s]Capturing num tokens (num_tokens=192 avail_mem=74.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.21it/s]

    Capturing num tokens (num_tokens=176 avail_mem=74.18 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.21it/s]Capturing num tokens (num_tokens=160 avail_mem=74.17 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.21it/s]Capturing num tokens (num_tokens=160 avail_mem=74.17 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=144 avail_mem=74.16 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=128 avail_mem=74.16 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=112 avail_mem=74.15 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=96 avail_mem=74.14 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.78it/s] Capturing num tokens (num_tokens=96 avail_mem=74.14 GB):  81%|████████  | 47/58 [00:01<00:00, 36.61it/s]Capturing num tokens (num_tokens=80 avail_mem=74.13 GB):  81%|████████  | 47/58 [00:01<00:00, 36.61it/s]Capturing num tokens (num_tokens=64 avail_mem=74.13 GB):  81%|████████  | 47/58 [00:01<00:00, 36.61it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.10 GB):  81%|████████  | 47/58 [00:01<00:00, 36.61it/s]Capturing num tokens (num_tokens=32 avail_mem=74.10 GB):  81%|████████  | 47/58 [00:01<00:00, 36.61it/s]Capturing num tokens (num_tokens=32 avail_mem=74.10 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=28 avail_mem=74.09 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=24 avail_mem=74.08 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=20 avail_mem=74.10 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=16 avail_mem=74.09 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=12 avail_mem=74.08 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=12 avail_mem=74.08 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.60it/s]Capturing num tokens (num_tokens=8 avail_mem=74.08 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.60it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.07 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.60it/s]Capturing num tokens (num_tokens=4 avail_mem=74.07 GB): 100%|██████████| 58/58 [00:01<00:00, 30.05it/s]


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
    Generated text:  Tania. My name is a surname, not a given name. My parents' first names are David and Michael. This is my mother's name. My father's name is John. David is my grandfather's name. My grandfather is also a surname. My mother's maiden name is Thompson. My father's maiden name is Brown. I am the youngest of the six children. I am from the town of Somersby, Australia. I am 15 years old. I was born in 1999. My first name is Tania and my last name is Gomes. My middle name is Josie.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she is the leader of the country. He or she has a very important job. He or she makes important decisions to solve the problems in the country. He or she is very important. He or she is called the president. The president of the United States is Donald Trump. He or she was elected to be the president in 2016. He or she has been president since 2017. President Donald Trump is a businessman. He or she likes to travel all over the world. He or she also likes to buy fancy cars. He or she has a lot of friends
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. Lyon
    C. Paris
    D. Toulouse
    Answer:
    A
    
    Among the following options, which one represents the first type of financial asset? 
    A. A business conducted by a foreign company in France
    B. A commercial bank in France
    C. A special purpose vehicle in France
    D. A government bond in France
    Answer:
    D
    
    Regarding the planning, construction, and operation of bridges and tunnels, which of the following statements is incorrect?
    A. The planning and design of bridges and tunnels must strictly follow the national laws and regulations of the People's Republic of China.
    
    ===============================
    Prompt: The future of AI is
    Generated text:  already here, and it’s going to change how we live our lives. New AI applications are being developed all the time, and a growing number of new jobs are being created to support these advancements. However, in many cases, the jobs that are created are not as well paid as the jobs that are lost. This can make it difficult to ensure that everyone in the workforce has a fair chance to move up the career ladder and advance their careers.
    To combat this issue, many companies and organizations are working to provide incentives for employees who stay in the workforce and who take on new AI applications. For example, some companies are offering bonuses or


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major center for art, literature, and music, and is home to many world-renowned museums and theaters. Paris is a popular tourist destination and a cultural hub for France and the world. It is also known for its cuisine, including its famous Parisian dishes such as croissants, escargot, and escargot. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection, risk
    


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
    Generated text:  [Name], and I am [Age]. I am a dedicated member of the [Occupation] community. I love to [How much time you spend on this hobby], and I am always up to [How much time you spend on this hobby]. I find it incredibly rewarding to [How you earn money or gain recognition for this hobby]. I also love to [What you do for fun or relieve stress]. I am [How many years old you are now], and I am currently [What you are currently doing]. What's something you're most proud of in your life, and what's something you're most proud of doing?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, a historical and cultural center. It is located in the south of France and is the country’s largest city, home to over 10 million residents and a vibrant metropolis. 
    
    The city is known for its beautiful architecture, vibrant music and dance scenes, and the annual Carnival celebration. Paris is also home to the Eiffel Tower, the Louvre Museum, Notre Dame Cathedral, and many other iconic landmarks. Despite its fame, Paris continues to be a diverse and multicultural city with over 100 languages spoken and more than 300 ethnicities represented. 
    
    The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and it is difficult to predict exactly what will happen, but there are some possible trends that are likely to shape the development of AI in the coming years. Here are some of the most promising areas:
    
    1. Increased use of AI in healthcare: As AI becomes more accessible and affordable, it is likely to have a significant impact on the healthcare industry. AI can be used to analyze medical data, detect diseases earlier, and develop personalized treatment plans. This could lead to more accurate diagnoses, improved patient outcomes, and more effective treatments.
    
    2. Enhanced privacy and security: As AI systems become more complex and sophisticated, there will be


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

     

    3

    0

    -year

    -old

     software

     engineer

     with

     experience

     in

     [

    industry

     or

     field

    ].

     I

     am

     passionate

     about

     [

    career

     goal

     or

     interest

    ],

     [

    mention

     specific

     career

     goals

     or

     interests

    ].

     I

     am

     a

     [

    general

     adjective

     describing

     your

     profession

    ].

     I

     enjoy

     [

    mention

     an

     interest

     or

     hobby

    ]

     and

     I

     am

     always

     eager

     to

     learn

     new

     skills

    .

     I

     am

     a

     [

    general

     adjective

     describing

     your

     personality

    ].

     I

     have

     a

     [

    number

    ]

     year

    -old

     dog

    ,

     [

    mention

     a

     pet

     or

     a

     pet

     of

     a

     loved

     one

    ].

     What

     brings

     you

     to

     this

     industry

     and

     how

     do

     you

     see

     it

     evolving

    ?

     I

     am

     excited

     to

     learn

     more

     about

     the

     growth

     and

     potential

     of

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Here

    's

     a

     concise

     factual

     statement

     about

     France

    's

     capital

     city

    :
    


    Paris

     is

     the

     capital

     and

     largest

     city

     of

     France

    ,

     located

     on

     the

     Se

    ine

     River

     in

     the

     central

     region

     of

     the

     country

    .

     It

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

     The

     city

     has

     a

     population

     of

     over

     

    2

     million

     people

     and

     is

     a

     major

     economic

     and

     political

     center

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     looking

     very

     promising

     with

     the

     potential

     to

     revolution

    ize

     virtually

     every

     aspect

     of

     our

     lives

    .

     Here

     are

     some

     possible

     trends

     that

     could

     emerge

     in

     the

     near

     future

    :
    


    1

    .

     Personal

    ized

     AI

    :

     As

     AI

     continues

     to

     learn

     and

     improve

    ,

     we

     can

     expect

     to

     see

     a

     rise

     in

     personalized

     AI

    .

     This

     will

     mean

     that

     machines

     will

     be

     able

     to

     learn

     and

     adapt

     to

     our

     unique

     needs

     and

     preferences

    ,

     allowing

     us

     to

     interact

     with

     them

     more

     effectively

    .
    


    2

    .

     Increased

     AI

     privacy

    :

     With

     more

     AI

     systems

     becoming

     capable

     of

     making

     decisions

     and

     taking

     action

     based

     on

     data

    ,

     we

     can

     expect

     to

     see

     a

     rise

     in

     privacy

     concerns

    .

     As

     AI

     systems

     become

     more

     integrated

     into

     our

     lives

    ,

     we

     will

     need

    



```python
llm.shutdown()
```
