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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.02it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.01it/s]


    2026-05-08 09:35:26,039 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 09:35:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  3.96it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  3.96it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  3.96it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  3.96it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  3.96it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:07,  6.01it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.22it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  9.22it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:02, 14.00it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:02, 14.00it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:04<00:02, 14.00it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:02, 14.00it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 14.00it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 14.00it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 14.00it/s]

    Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 14.00it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:05<00:01, 20.64it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 29.11it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 29.11it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 29.11it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 29.11it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 29.11it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 29.11it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 29.11it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 29.11it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 29.11it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 37.54it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 37.54it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 37.54it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 37.54it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 37.54it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 37.54it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 37.54it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 37.54it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 37.54it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 37.54it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 47.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.47 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.47 GB):   3%|▎         | 2/58 [00:00<00:03, 17.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.76 GB):   3%|▎         | 2/58 [00:00<00:03, 17.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.45 GB):   3%|▎         | 2/58 [00:00<00:03, 17.23it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.45 GB):   7%|▋         | 4/58 [00:00<00:03, 15.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.48 GB):   7%|▋         | 4/58 [00:00<00:03, 15.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.48 GB):   7%|▋         | 4/58 [00:00<00:03, 15.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.48 GB):  10%|█         | 6/58 [00:00<00:03, 14.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.49 GB):  10%|█         | 6/58 [00:00<00:03, 14.39it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.50 GB):  10%|█         | 6/58 [00:00<00:03, 14.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.50 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.51 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.51 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.51 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.40it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.52 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.40it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.54 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.54 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.54 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.57it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.54 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.54 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.52 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.57it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=72.52 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.12 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.16 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.20 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 22.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 22.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.17 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.67it/s]Capturing num tokens (num_tokens=960 avail_mem=72.18 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.67it/s] Capturing num tokens (num_tokens=896 avail_mem=72.16 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.67it/s]

    Capturing num tokens (num_tokens=896 avail_mem=72.16 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.18it/s]Capturing num tokens (num_tokens=832 avail_mem=72.17 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.18it/s]Capturing num tokens (num_tokens=768 avail_mem=72.17 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.18it/s]Capturing num tokens (num_tokens=704 avail_mem=72.16 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.18it/s]Capturing num tokens (num_tokens=640 avail_mem=72.15 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.18it/s]Capturing num tokens (num_tokens=640 avail_mem=72.15 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=576 avail_mem=72.15 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=512 avail_mem=72.13 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=480 avail_mem=72.15 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=448 avail_mem=72.12 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.31it/s]

    Capturing num tokens (num_tokens=448 avail_mem=72.12 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=416 avail_mem=72.13 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=384 avail_mem=72.12 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=352 avail_mem=72.11 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=320 avail_mem=72.10 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=320 avail_mem=72.10 GB):  60%|██████    | 35/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=288 avail_mem=72.10 GB):  60%|██████    | 35/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=256 avail_mem=72.09 GB):  60%|██████    | 35/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=240 avail_mem=72.08 GB):  60%|██████    | 35/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=224 avail_mem=72.08 GB):  60%|██████    | 35/58 [00:01<00:00, 33.10it/s]

    Capturing num tokens (num_tokens=224 avail_mem=72.08 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=208 avail_mem=72.07 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=192 avail_mem=72.07 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=176 avail_mem=72.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=160 avail_mem=72.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=160 avail_mem=72.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.36it/s]Capturing num tokens (num_tokens=144 avail_mem=72.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.36it/s]Capturing num tokens (num_tokens=128 avail_mem=72.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.36it/s]Capturing num tokens (num_tokens=112 avail_mem=72.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.36it/s]Capturing num tokens (num_tokens=96 avail_mem=72.03 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.36it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=72.03 GB):  81%|████████  | 47/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=80 avail_mem=72.03 GB):  81%|████████  | 47/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=64 avail_mem=72.02 GB):  81%|████████  | 47/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=48 avail_mem=72.01 GB):  81%|████████  | 47/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=32 avail_mem=72.00 GB):  81%|████████  | 47/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=32 avail_mem=72.00 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=28 avail_mem=71.99 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=24 avail_mem=72.01 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=20 avail_mem=72.00 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=16 avail_mem=72.00 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.44it/s]

    Capturing num tokens (num_tokens=12 avail_mem=71.99 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=12 avail_mem=71.99 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.86it/s]Capturing num tokens (num_tokens=8 avail_mem=71.98 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.86it/s] Capturing num tokens (num_tokens=4 avail_mem=71.98 GB):  97%|█████████▋| 56/58 [00:02<00:00, 37.86it/s]Capturing num tokens (num_tokens=4 avail_mem=71.98 GB): 100%|██████████| 58/58 [00:02<00:00, 28.47it/s]


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
    Generated text:  Caroline. I'm a student at a university in the United States. I like to be active, so I love running on the treadmill in the gym. I also like to read books. I enjoy writing my own stories, and I like to decorate my house with my artwork. I also love to travel and explore new places. I like to listen to music, and I like to listen to the birdsong. What is the most interesting thing you have experienced, and what makes you curious about it? Caroline is an interesting person because she is creative and loves to experiment with new things. She is also very curious and loves to learn about
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. If you were the president of the United States, what would you do?
    As the President of the United States, I would take on the responsibility and role of being the leader and manager of the country. Here are a few things I would do:
    1. Make sure that the country is running smoothly: As the leader, my first priority would be to make sure that the country is running smoothly and efficiently. This would involve implementing policies and programs that will help the country to grow and develop.
    2. Address issues that the people are concerned about: I would be responsible for addressing issues that the people
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. (Judge true or false)
    
    To determine whether the statement "The capital of France is Paris" is true or false, we need to analyze the location of Paris in the world. Paris is the capital of France, which is a country in Western Europe. The United Kingdom and the United States are the other two major countries in Western Europe.
    
    The United Kingdom is located on the mainland of the United Kingdom, which is part of the island of Great Britain. The United States is also located on the mainland of the United States, which is part of the island of North America. However, Paris, the capital of France, is situated
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the right people
    
    AI has made its mark in all sectors, from banking to transportation, but it’s also on the rise in healthcare. With more and more patients seeking more personalized and highly personalized treatments, the future of AI is in the hands of the right people.
    
    AI has made its mark in all sectors, from banking to transportation, but it’s also on the rise in healthcare. With more and more patients seeking more personalized and highly personalized treatments, the future of AI is in the hands of the right people.
    
    The healthcare industry has been in a fog of over-reliance on untrained medical professionals who lack


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art, and cuisine. Paris is a popular tourist destination and a major economic center in France. It is home to many world-renowned museums, theaters, and restaurants. The city is also known for its vibrant nightlife and cultural events. Paris is a city of contrasts, with its modern architecture and historical landmarks blending
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more prevalent in manufacturing, transportation, and other industries, where it can automate repetitive tasks and increase efficiency. This will lead to the development of new types of AI that can perform tasks that are currently done by humans, such as language translation, image recognition, and autonomous vehicles.
    
    2. Enhanced human-AI collaboration: AI is expected to become more integrated with human AI, allowing for more complex and nuanced interactions between humans and machines.
    


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
    Generated text:  [Your Name], and I am a [Type of Character] with [Your Primary Trait]. I'm passionate about [Your Primary Interest or Interest Group], and I'm always up for [Your Passion]. I enjoy [My Humor or Motivation], and I love [My Style or Personality]. I'm [Your Personality or Character Traits], and I'm always ready to learn and grow. I'm an [Your Job Title], and I'm dedicated to [Your Professionally Promised Problem or Goal]. I'm a [Your Ambition or Goal], and I'm always striving to [Your Long-term Vision or Goal]. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the south-central part of the country. It is the largest city in France and the second-largest urban area in the European Union. It is the historical and cultural center of France and is the most-visited city in the world. Paris is a major transportation hub, offering access to major highways, airports, and metro lines. It is also home to many world-renowned attractions, including the Louvre Museum, the Notre-Dame Cathedral, and the Eiffel Tower. Paris is known for its fashion, gastronomy, and cuisine, as well as its food culture, and is a major center for the arts,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but some trends are likely to continue:
    
    1. Increased use of AI in healthcare: AI is expected to play an important role in healthcare, with more AI applications being developed to help doctors diagnose diseases, predict patient outcomes, and develop personalized treatment plans.
    
    2. Greater focus on privacy and data protection: As AI becomes more advanced, there is a risk of it being used for unethical or harmful purposes, such as collecting and using personal data without permission. As a result, there may be increased focus on privacy and data protection regulations.
    
    3. Use of AI in the transportation industry: AI is expected to play a key role in


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

    Occup

    ation

    ].

     I

     have

     a

     passion

     for

     [

    Why

     is

     it

     my

     passion

    ].

     How

     would

     you

     describe

     your

     personality

     and

     what

     draws

     you

     to

     this

     career

    ?
    


    Remember

    ,

     this

     is

     a

     brief

    ,

     neutral

     introduction

    ,

     and

     I

    'd

     like

     you

     to

     focus

     on

     your

     personality

     traits

     and

     values

    .

     For

     example

    ,

     you

     could

     mention

     your

     sense

     of

     humor

    ,

     your

     love

     for

     nature

    ,

     or

     your

     approach

     to

     problem

    -solving

    .


    Sure

    ,

     here

    's

     a

     neutral

     self

    -int

    roduction

     for

     a

     fictional

     character

    :


    My

     name

     is

     [

    Name

    ],

     and

     I

    'm

     a

     [

    Occup

    ation

    ].

     I

     have

     a

     passion

     for

     [

    Why

     is

     it

     my

     passion

    ].

     I

     love

     the

     outdoors

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     beautiful

     sights

     like

     the

     Lou

    vre

     Museum

    ,

     and

     bustling

     street

     life

    .

     It

    's

     also

     a

     cultural

     and

     economic

     powerhouse

     with

     many

     world

    -ren

    owned

     museums

    ,

     theaters

    ,

     and

     a

     rich

     heritage

    .

     The

     city

     offers

     a

     blend

     of

     history

    ,

     modern

    ity

    ,

     and

     rich

     culture

    .

     Paris

     is

     a

     beautiful

    ,

     charming

    ,

     and

     vibrant

     place

     to

     visit

     and

     live

    .

     
    


    Is

     there

     any

     specific

     aspect

     of

     Paris

     that

     you

     find

     particularly

     unique

     or

     interesting

    ?

     I

    'm

     not

     sure

    .

     Can

     you

     tell

     me

     more

     about

     Paris

    ?

     Yes

    ,

     Paris

     is

     a

     unique

     and

     popular

     city

     in

     France

    ,

     known

     for

     its

     world

    -ren

    owned

     landmarks

     such

     as

     the

     E

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

    ,

     and

     there

     are

     many

     exciting

     trends

     that

     are

     shaping

     its

     development

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

     Enhanced

     human

    -machine

     collaboration

    :

     AI

     will

     continue

     to

     integrate

     seamlessly

     with

     human

     machines

    ,

     leading

     to

     more

     efficient

     and

     effective

     collaboration

     in

     fields

     like

     healthcare

    ,

     manufacturing

    ,

     and

     logistics

    .
    


    2

    .

     Autonomous

     vehicles

    :

     AI

     will

     become

     increasingly

     integrated

     into

     autonomous

     vehicles

    ,

     leading

     to

     safer

    ,

     more

     reliable

    ,

     and

     faster

     transportation

     systems

    .
    


    3

    .

     Improved

     data

     analysis

    :

     AI

     will

     enable

     more

     accurate

     and

     detailed

     analysis

     of

     data

    ,

     leading

     to

     better

     predictions

     and

     better

     decision

    -making

     in

     various

     industries

    .
    


    4

    .

     Enhanced

     security

    :

     AI

     will

     continue

     to

     improve

     the

     security

     of

     systems

     and

    



```python
llm.shutdown()
```
