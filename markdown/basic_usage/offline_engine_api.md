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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.63it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.63it/s]


    2026-05-11 01:11:29,725 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 01:11:29] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:42,  3.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:42,  3.91s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:42,  3.91s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:42,  3.91s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:42,  3.91s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]

    Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.92it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.92it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.92it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.92it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:10,  4.60it/s]

    Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:05,  7.34it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:05,  7.34it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:05,  7.34it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:05,  7.34it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:05,  7.34it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:03, 10.62it/s]

    Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:03, 10.62it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=416):  41%|████▏     | 24/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=384):  41%|████▏     | 24/58 [00:04<00:02, 15.43it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:04<00:00, 26.47it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:04<00:00, 26.47it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:04<00:00, 26.47it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:04<00:00, 26.47it/s]

    Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:04<00:00, 26.47it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:04<00:00, 26.47it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:04<00:00, 26.47it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:04<00:00, 26.47it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:04<00:00, 26.47it/s]Compiling num tokens (num_tokens=176):  57%|█████▋    | 33/58 [00:04<00:00, 26.47it/s]Compiling num tokens (num_tokens=160):  57%|█████▋    | 33/58 [00:04<00:00, 26.47it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:04<00:00, 39.45it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:04<00:00, 39.45it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:04<00:00, 39.45it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:04<00:00, 39.45it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:04<00:00, 39.45it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:04<00:00, 39.45it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:04<00:00, 39.45it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:04<00:00, 39.45it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:04<00:00, 39.45it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:04<00:00, 39.45it/s]Compiling num tokens (num_tokens=24):  74%|███████▍  | 43/58 [00:04<00:00, 39.45it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:04<00:00, 51.28it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:04<00:00, 51.28it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 51.28it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:04<00:00, 51.28it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:04<00:00, 51.28it/s] 

    Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:04<00:00, 51.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.74 GB):   2%|▏         | 1/58 [00:00<00:07,  7.37it/s]Capturing num tokens (num_tokens=7680 avail_mem=51.71 GB):   2%|▏         | 1/58 [00:00<00:07,  7.37it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=51.71 GB):   3%|▎         | 2/58 [00:00<00:07,  7.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=51.71 GB):   3%|▎         | 2/58 [00:00<00:07,  7.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=51.70 GB):   3%|▎         | 2/58 [00:00<00:07,  7.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=51.70 GB):   7%|▋         | 4/58 [00:00<00:04, 11.19it/s]Capturing num tokens (num_tokens=6144 avail_mem=51.70 GB):   7%|▋         | 4/58 [00:00<00:04, 11.19it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   7%|▋         | 4/58 [00:00<00:04, 11.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):  10%|█         | 6/58 [00:00<00:05,  8.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):  10%|█         | 6/58 [00:00<00:05,  8.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):  10%|█         | 6/58 [00:00<00:05,  8.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):  10%|█         | 6/58 [00:00<00:05,  8.74it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  10%|█         | 6/58 [00:00<00:05,  8.74it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.42it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.99 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.99 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.05it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.05it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.98 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.05it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  34%|███▍      | 20/58 [00:01<00:01, 29.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  34%|███▍      | 20/58 [00:01<00:01, 29.71it/s]Capturing num tokens (num_tokens=960 avail_mem=70.97 GB):  34%|███▍      | 20/58 [00:01<00:01, 29.71it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  34%|███▍      | 20/58 [00:01<00:01, 29.71it/s]

    Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  34%|███▍      | 20/58 [00:01<00:01, 29.71it/s]Capturing num tokens (num_tokens=768 avail_mem=70.96 GB):  34%|███▍      | 20/58 [00:01<00:01, 29.71it/s]Capturing num tokens (num_tokens=768 avail_mem=70.96 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.85it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.85it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.85it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.85it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.85it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  50%|█████     | 29/58 [00:01<00:00, 32.94it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:01<00:00, 32.94it/s]

    Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:01<00:00, 32.94it/s]Capturing num tokens (num_tokens=416 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:01<00:00, 32.94it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:01<00:00, 32.94it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:01<00:00, 32.94it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.74it/s]

    Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.78it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.58it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.58it/s]

    Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.58it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.58it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.58it/s] Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  81%|████████  | 47/58 [00:01<00:00, 30.75it/s]Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  81%|████████  | 47/58 [00:01<00:00, 30.75it/s]

    Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 30.75it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 30.75it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 30.75it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.56it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.56it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.56it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.56it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.56it/s]

    Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:02<00:00, 28.42it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:02<00:00, 28.42it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:02<00:00, 28.42it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:02<00:00, 28.42it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:02<00:00, 26.27it/s]


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
    Generated text:  Miguel Díaz, and I'm the assistant who wrote the code. Let's move on to the next topic.
    
    What is the current state of the project and what are the upcoming tasks?
    
    I apologize, but I need more context or information to provide an accurate answer. Could you please provide the name of the project you're referring to, as well as the details of the upcoming tasks? Without this information, I can't determine the current state of the project and the upcoming tasks. Please let me know the name of the project and the upcoming tasks so I can assist you better. Alternatively, if you have any specific questions about the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is in charge of the country. He is also a man with a lot of money. People can talk about the president at any time. Many people would like to see the president run again. But others think that he should not run again. Some people also think that the president should not be a man at all. Others think that the president should be a woman. Some people think that the president should be a man because of his father. Others think that the president should be a woman because of his mother. Other people think that the president should be a woman because he is a very popular man with a
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris. If you look at the map of the world, you can see that Paris is in the south of France. Paris is an important city in France. People think that Paris is the most beautiful city. As the capital, Paris is very important. Paris is famous for its long streets and tall buildings. It is the best place for people to take a walk. There are many restaurants and cafes in Paris. Many visitors like to come to Paris to visit its famous museums. The city has a beautiful park. There are many trees and flowers in it. But Paris is also very noisy. People often talk on the streets
    ===============================
    Prompt: The future of AI is
    Generated text:  the same as that of the human brain: it is constantly evolving and improving, not just in terms of the technology being used, but also in terms of the understanding of how it works.
    The future of AI is the same as that of the human brain: it is constantly evolving and improving, not just in terms of the technology being used, but also in terms of the understanding of how it works.
    The brain is a highly complex system with billions of cells that are constantly interconnected. Each cell is capable of processing information and making decisions, and the collective outputs of all cells form the brain's overall intelligence. Similarly, the output of the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Museum. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is the capital of France and the largest city in the European Union. 
    
    Paris is also known for its fashion industry, with iconic fashion houses such as Chanel, Louis Vuitton, and Dior. It is also home to the Eiffel Tower, which is considered one of the most iconic structures in the world. Paris is a city of contrasts
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the potential trends that could shape the future of AI:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to automate many of the tasks that are currently done by humans. This could lead to increased efficiency and productivity, but it could also lead to job displacement for some workers.
    
    2. Improved privacy and security: As AI becomes more advanced, it is likely to require more data to train its algorithms, which could lead to increased data privacy and security concerns.
    


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
    Generated text:  [Name], and I'm a [job title]. I'm from [location]. I've always loved to be [vocabulary word or ability], and I've been learning it for as long as I can remember. I have a strong work ethic, and I love to strive for excellence. I've been passionate about [mention a hobby or activity that makes you happy]. I believe in [mention a specific belief or principle], and I've always had a deep interest in [mention a subject or topic that fascinates you]. I'm always looking for opportunities to [mention a goal or aspiration]. I'm [your personal ranking among
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known as the “City of Love” and is a world-renowned cultural and historical center. Paris is home to the Louvre Museum, Notre-Dame Cathedral, and the Eiffel Tower, among other impressive landmarks. The city also has a diverse range of cuisine, art, and fashion, making it a popular tourist destination worldwide. Additionally, Paris is known for its culture, food, and music scene, making it a hub for cultural events, music festivals, and nightlife. The city is a major financial and business center, known for its extensive shopping, dining, and entertainment options. In summary, Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued advancements in areas such as machine learning, natural language processing, computer vision, robotics, and autonomous systems. Here are some possible future trends that are likely to shape the future of AI:
    
    1. Advancements in AI hardware: As AI becomes more complex and sophisticated, we may see a shift towards using more powerful AI hardware, such as GPUs and TPUs, to accelerate computations and improve performance.
    
    2. Increased use of AI in healthcare: As AI is used to assist doctors and healthcare providers with diagnosis and treatment, we may see a greater focus on developing AI-driven health care solutions.
    
    3. Integration of AI in education


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

    'm

     a

    /an

     [

    Age

    ]

     year

     old

    ,

     [

    Gender

    ]

     and

     [

    Country

    ].

     I

     live

     in

     [

    City

    /T

    own

    ]

     and

     I

     have

     been

     [

    Occup

    ation

     or

     Hobby

    ]

     for

     [

    Number

     of

     Years

    ].

     I

    'm

     [

    Attr

    actions

     or

     Inter

    ests

    ].

     I

     really

     enjoy

     [

    Number

     of

     Activities

    /

    Things

     You

     Do

    ]

     and

     I

     believe

     it

     makes

     me

     [

    Positive

    /

    Negative

    ].

     I

    'm

     currently

     [

    Age

    /

    Occup

    ation

     or

     Hobby

    /

    Year

    ]

     and

     I

    'm

     [

    Gender

    ],

     [

    Country

    ],

     [

    City

    /T

    own

    ].

     I

    've

     been

     [

    Number

     of

     Years

    ]

     of

     age

     and

     I

     have

     been

     [

    Occup

    ation

     or

     Hobby

    /

    Year

    ]

     in

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     located

     in

     the

     Lo

    ire

     Valley

     region

     of

     the

     central

     French

     Low

     Countries

    .

     It

     is

     the

     most

     populous

     city

     and

     largest

     city

     in

     France

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     culture

    ,

     and

     stunning

     architecture

    .

     It

     has

     a

     large

     population

     of

     over

     

    6

     million

     people

    ,

     with

     many

     French

     people

     residing

     in

     the

     surrounding

     suburbs

    .

     It

     is

     the

     official

     capital

     of

     France

    ,

     and

     its

     name

     is

     derived

     from

     the

     Latin

     "

    Par

    th

    ia

    ,"

     which

     means

     "

    Par

    th

    ian

    ".

     Paris

     is

     also

     a

     major

     transportation

     hub

    ,

     with

     major

     highways

    ,

     airports

    ,

     and

     train

     stations

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     It

     is

     known

     for

     its

     famous

     landmarks

     such

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     extremely

     bright

     and

     exciting

    ,

     and

     there

     is

     no

     doubt

     that

     it

     is

     going

     to

     play

     a

     significant

     role

     in

     many

     aspects

     of

     life

    .

     Here

     are

     some

     of

     the

     possible

     future

     trends

     in

     AI

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

     is

     already

     being

     used

     in

     healthcare

     to

     assist

     doctors

     in

     diagn

    osing

     and

     treating

     diseases

    .

     In

     the

     future

    ,

     we

     can

     expect

     AI

     to

     become

     even

     more

     advanced

     and

     to

     integrate

     seamlessly

     with

     healthcare

     systems

     to

     provide

     even

     more

     personalized

     and

     accurate

     treatment

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     finance

    :

     AI

     is

     already

     being

     used

     in

     finance

     to

     automate

     trading

    ,

     fraud

     detection

    ,

     and

     risk

     assessment

    .

     In

     the

     future

    ,

     we

     can

     expect

     AI

     to

     become

    



```python
llm.shutdown()
```
